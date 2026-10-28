#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FLUX Kontext LoRA 微调训练主脚本
===========================================

这是一个解耦、可扩展的 LoRA 训练实现：
- 所有参数通过 YAML 配置文件管理
- 模块化设计，易于添加新功能（量化、offload、MoE-LoRA等）
- 高度解耦，每个模块职责单一

使用方法：
    python train_lora.py --config config/train_lora_config.yaml
"""

import argparse
import math
import os
import sys
from pathlib import Path

# 添加本地库路径
CURRENT_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(CURRENT_DIR / "peft" / "src"),
               str(CURRENT_DIR / "diffusers" / "src"),
               str(CURRENT_DIR / "train"), str(CURRENT_DIR)]

import torch

# 导入核心模块
from core import (
    TrainingConfig,
    get_logger,
    ModelLoader,
    LoRAConfigurator,
    create_dataloader,
    create_optimizer,
    create_scheduler,
    Trainer,
    ModelSaver,
    LatentCache,
)
from core.memory import MemoryOptimizer
from diffusers import FluxKontextPipeline


def main():
    """主函数"""
    # ========================================================================
    # 1. 解析参数
    # ========================================================================
    parser = argparse.ArgumentParser(description="FLUX Kontext LoRA 训练")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_lora_config.yaml",
        help="配置文件路径"
    )
    args = parser.parse_args()
    
    # ========================================================================
    # 2. 加载配置
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"FLUX Kontext LoRA 微调训练程序（重构版）")
    print(f"{'='*60}\n")
    
    print(f"[配置] 正在加载配置文件：{args.config}")
    config = TrainingConfig.from_yaml(args.config)
    print(f"[配置] ✓ 配置加载完成\n")
    
    # ========================================================================
    # 3. 初始化日志
    # ========================================================================
    logger = get_logger("train", config.log_level)
    logger.info(f"训练任务：{config.job_name}")
    logger.info(f"设备：{config.device}")
    logger.info(f"模型路径：{config.model_path}")
    
    # ========================================================================
    # 4. 设置随机种子
    # ========================================================================
    if config.seed is not None:
        torch.manual_seed(config.seed)
        logger.info(f"随机种子：{config.seed}")
    
    # ========================================================================
    # 5. 初始化显存优化器
    # ========================================================================
    logger.info(f"\n[步骤 1/8] 初始化显存优化器")
    memory_optimizer = MemoryOptimizer(config)
    logger.info(f"✓ 显存优化器初始化完成")
    logger.info(f"  - 量化：{'启用' if config.quantize else '禁用'}")
    logger.info(f"  - 梯度检查点：{'启用' if config.gradient_checkpointing else '禁用'}")
    logger.info(f"  - VAE Offload：{'启用' if config.vae_offload else '禁用'}")
    
    # ========================================================================
    # 6. 加载模型
    # ========================================================================
    logger.info(f"\n[步骤 2/8] 加载模型")
    model_loader = ModelLoader(config, memory_optimizer)
    model_loader.set_logger(logger)
    
    transformer, vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, scheduler = model_loader.load_all()
    
    # ========================================================================
    # 7. 配置 LoRA
    # ========================================================================
    logger.info(f"\n[步骤 3/8] 配置 LoRA")
    lora_configurator = LoRAConfigurator(config)
    lora_configurator.set_logger(logger)
    
    transformer = lora_configurator.apply_lora(transformer)
    
    # 应用梯度检查点
    memory_optimizer.apply_gradient_checkpointing(transformer)
    
    # ========================================================================
    # 8. 创建 Latent 缓存（如果启用）
    # ========================================================================
    latent_cache = None
    if config.cache_latents or config.cache_to_disk:
        logger.info(f"\n[步骤 4/8] 初始化 Latent 缓存")
        latent_cache = LatentCache(
            config=config,
            vae=vae,
            device=config.device,
            dtype=config.get_dtype()
        )
        logger.info(f"✓ Latent 缓存初始化完成")
        logger.info(f"  - 模式：{'磁盘缓存' if config.cache_to_disk else '内存缓存'}")

    # ========================================================================
    # 9. 创建数据加载器
    # ========================================================================
    logger.info(f"\n[步骤 5/8] 创建数据加载器")
    dataset, train_dataloader = create_dataloader(config, latent_cache=latent_cache)
    logger.info(f"✓ 数据加载器创建完成")

    # 缓存 latents（如果启用）
    if latent_cache is not None:
        logger.info(f"\n[步骤 6/8] 缓存 Latents（首次运行可能需要较长时间）")
        latent_cache.cache_dataset(dataset, logger=logger)
    
    # ========================================================================
    # 10. 创建优化器和调度器
    # ========================================================================
    logger.info(f"\n[步骤 7/8] 创建优化器")
    
    # 收集可训练参数
    params_to_optimize = [p for p in transformer.parameters() if p.requires_grad]
    logger.info(f"可训练参数数量：{sum(p.numel() for p in params_to_optimize):,}")
    
    optimizer = create_optimizer(params_to_optimize, config)
    logger.info(f"✓ 优化器：{config.optimizer_type}")
    
    # 创建调度器
    # 有梯度累积时，学习率调度应该基于“优化器步数”而不是迭代步数
    steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    num_training_steps = steps_per_epoch * config.num_epochs
    lr_scheduler = create_scheduler(optimizer, config, num_training_steps)
    logger.info(f"✓ 学习率调度器：{config.lr_scheduler}")
    
    # ========================================================================
    # 11. 创建保存器
    # ========================================================================
    logger.info(f"\n[步骤 8/8] 初始化保存器")
    saver = ModelSaver(config, logger)
    logger.info(f"✓ 保存目录：{config.output_dir}")

    # ========================================================================
    # 11. 初始化验证采样管线（可选）
    # ========================================================================
    validation_pipeline = None
    if config.validation_enabled and not config.validation_external_infer:
        logger.info(f"[验证] 初始化 FluxKontextPipeline 以便按 epoch 采样验证")
        validation_pipeline = FluxKontextPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder_1,
            tokenizer=tokenizer_1,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
        )
        validation_pipeline.set_progress_bar_config(disable=True)
    
    # ========================================================================
    # 12. 训练
    # ========================================================================
    logger.info(f"\n[开始训练]")
    
    trainer = Trainer(config, logger)
    
    # 训练循环（生成器模式）
    for event_type, data in trainer.train(
        transformer, vae, text_encoder_1, text_encoder_2,
        tokenizer_1, tokenizer_2, scheduler, train_dataloader,
        optimizer, lr_scheduler, validation_pipeline=validation_pipeline
    ):
        if event_type == "save":
            saver.save_lora(transformer, epoch=data)
    
    # ========================================================================
    # 12. 完成
    # ========================================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"训练完成！")
    logger.info(f"LoRA 权重已保存到：{config.output_dir}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    """程序入口"""
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断] 训练被用户中断")
    except Exception as e:
        print(f"\n[错误] 训练过程中发生错误：{e}")
        import traceback
        traceback.print_exc()
        raise
