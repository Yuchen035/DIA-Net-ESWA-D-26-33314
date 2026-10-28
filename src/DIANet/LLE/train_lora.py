"""
MoE-LoRA暗光增强训练 - 主训练脚本

使用方法:
    python train_lora.py --config configs/lora_training.yaml
    python train_lora.py --config configs/lora_training.yaml --use_mock  # 使用Mock FLUX进行测试

示例:
    # 使用真实FLUX模型训练
    python train_lora.py --config configs/lora_training.yaml

    # 使用Mock FLUX模型进行快速测试
    python train_lora.py --config configs/lora_training.yaml --use_mock
"""

import argparse
import sys
import os
import torch
import yaml
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))

from core.config import TrainingConfig
from core.lora.configurator import LoRAConfigurator
from core.data.dataset import ImagePairDataset
from core.data.dataloader import create_dataloader
from core.loss.flow_matching import FlowMatchingLoss
from core.trainer import Trainer
from core.saver import ModelSaver


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LoRA训练脚本")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径 (YAML格式)"
    )

    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="使用Mock FLUX模型进行测试（无需下载真实模型）"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（覆盖配置文件中的设置）"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="设备选择（覆盖配置文件中的设置）"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="打印详细信息"
    )

    return parser.parse_args()


def load_config(config_path: str) -> TrainingConfig:
    """
    从YAML文件加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        TrainingConfig对象
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    return TrainingConfig.from_dict(config_dict)


def load_models(config: TrainingConfig, use_mock: bool = False):
    """
    加载FLUX模型或Mock模型

    Args:
        config: 训练配置
        use_mock: 是否使用Mock模型

    Returns:
        (transformer, vae, text_encoder_1, text_encoder_2,
         tokenizer_1, tokenizer_2, scheduler)
    """
    if use_mock:
        print("\n" + "=" * 70)
        print("⚠️  使用Mock FLUX模型（测试模式）")
        print("=" * 70)

        from core.mock_flux import MockFluxModelLoader

        loader = MockFluxModelLoader(
            device=config.device,
            dtype=getattr(torch, config.model.dtype)
        )
        return loader.load_all()
    else:
        print("\n" + "=" * 70)
        print("加载真实FLUX.1-dev模型（含PhotoDoodle预训练+显存优化）")
        print("=" * 70)

        # 从项目根目录导入RealFluxModelLoader
        import sys
        project_root = os.path.dirname(os.path.dirname(__file__))
        sys.path.insert(0, project_root)

        from core.model_loader import create_real_flux_loader
        from core.memory import MemoryOptimizer

        # 创建显存优化器
        memory_optimizer = MemoryOptimizer(config)
        print("\n[显存优化]")
        print(f"  - Transformer量化: {'启用 (int8)' if getattr(config, 'quantize', True) else '禁用'}")
        print(f"  - Text Encoder量化: {'启用 (int8)' if getattr(config, 'quantize_te', True) else '禁用'}")
        print(f"  - VAE Offload: 启用")
        print(f"  - Text Encoder Offload: 启用")
        print(f"  - 梯度检查点: {'启用' if getattr(config, 'gradient_checkpointing', True) else '禁用'}")

        # 创建RealFluxModelLoader
        loader = create_real_flux_loader(config.model)

        # 一键加载所有组件（含PhotoDoodle）
        print("\n[模型加载] 加载FLUX Pipeline...")
        components = loader.load_all(
            load_photodoodle=config.model.photodoodle.enabled if hasattr(config.model, 'photodoodle') else True,
            fuse_photodoodle=config.model.photodoodle.fuse_before_training if hasattr(config.model, 'photodoodle') else True,
            enable_cpu_offload=False  # 暂时禁用，我们手动控制offload
        )

        transformer, vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, scheduler = components

        # 应用offload（VAE和Text Encoders保持在CPU，不量化）
        print("\n[显存优化] 应用CPU Offload策略...")
        vae = vae.to("cpu")
        text_encoder_1 = text_encoder_1.to("cpu")
        text_encoder_2 = text_encoder_2.to("cpu")
        print("  - VAE: CPU (不量化)")
        print("  - Text Encoder 1: CPU (不量化)")
        print("  - Text Encoder 2: CPU (不量化)")

        # 应用量化优化（只量化Transformer）
        print("\n[显存优化] 应用量化...")
        if getattr(config, 'quantize', True):
            transformer = memory_optimizer.apply_quantization(transformer, "transformer")
            print("  - Transformer: 已量化 (int8)")
        else:
            print("  - Transformer: 不量化")

        # 将Transformer移到训练设备
        print("\n[模型设备] 将Transformer移到训练设备...")
        transformer = transformer.to(config.device)
        print(f"  - Transformer: {config.device}")

        return transformer, vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, scheduler


def create_optimizer_and_scheduler(config: TrainingConfig, lora_params):
    """
    创建优化器和学习率调度器

    Args:
        config: 训练配置
        lora_params: LoRA参数列表

    Returns:
        (optimizer, lr_scheduler)
    """
    # 创建优化器
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=config.train.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )

    # 计算总训练步数
    # 注意：这里假设dataloader长度已知，实际使用时需要传入
    # 简化版：使用StepLR
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.train.num_epochs * 100,  # 粗略估计
        eta_min=config.train.learning_rate * 0.1
    )

    return optimizer, lr_scheduler


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    print("=" * 70)
    print("MoE-LoRA 暗光增强训练")
    print("=" * 70)
    print(f"配置文件: {args.config}")
    print(f"使用Mock模型: {args.use_mock}")
    print("=" * 70)

    # 1. 加载配置
    print("\n【步骤1】加载配置...")
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    # 命令行参数覆盖
    if args.output_dir:
        config.save.output_dir = args.output_dir
    if args.device:
        config.device = args.device

    print(f"✓ 配置加载成功")
    print(f"  模型: {config.model.base_model}")
    print(f"  训练轮数: {config.train.num_epochs}")
    print(f"  批次大小: {config.train.batch_size}")
    print(f"  学习率: {config.train.learning_rate}")
    print(f"  输出目录: {config.save.output_dir}")
    print(f"  设备: {config.device}")

    # 2. 加载模型
    print("\n【步骤2】加载FLUX模型...")
    (transformer, vae, text_encoder_1, text_encoder_2,
     tokenizer_1, tokenizer_2, scheduler) = load_models(config, args.use_mock)
    print("✓ 模型加载成功")

    # 3. 注入LoRA
    print("\n【步骤3】注入LoRA到Transformer...")
    lora_configurator = LoRAConfigurator(
        target_modules=config.lora.target_modules,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        verbose=args.verbose
    )
    transformer, stats = lora_configurator.inject_lora(transformer)
    print(f"✓ LoRA注入完成: {stats.lora_injected} 个模块")

    # 3.5 应用梯度检查点（如果启用且不是Mock模式）
    if not args.use_mock and getattr(config, 'gradient_checkpointing', True):
        print("\n【步骤3.5】启用梯度检查点...")
        if hasattr(transformer, 'enable_gradient_checkpointing'):
            transformer.enable_gradient_checkpointing()
            print("✓ 梯度检查点已启用")
        else:
            print("⚠️  Transformer不支持梯度检查点")

    # 4. 创建数据集
    print("\n【步骤4】创建数据集...")
    dataset = ImagePairDataset(
        input_image_dir=config.data.input_image_dir,
        target_image_dir=config.data.target_image_dir,
        prompt=config.data.prompt,
        resolution=config.data.resolution,
        center_crop=config.data.center_crop,
        random_flip=config.data.random_flip,
    )

    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.device == "cuda",
    )
    print(f"✓ 数据集创建成功: {len(dataset)} 个样本")

    # 5. 创建损失函数（train的Trainer不需要传入loss_fn）
    print("\n【步骤5】创建损失函数...")
    # FlowMatchingLoss在train的Trainer内部计算
    print("✓ Flow Matching Loss（内置）")

    # 6. 创建训练器（适配train的Trainer接口）
    print("\n【步骤6】创建训练器...")
    # 创建简单的logger
    import logging
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)

    trainer = Trainer(
        config=config,
        logger=logger
    )
    print("✓ 训练器创建成功")

    # 7. 创建优化器和学习率调度器
    print("\n【步骤7】创建优化器...")
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer, lr_scheduler = create_optimizer_and_scheduler(config, lora_params)
    print(f"✓ 优化器创建成功")

    # 8. 创建模型保存器
    print("\n【步骤8】创建模型保存器...")
    saver = ModelSaver(
        output_dir=config.save.output_dir,
        save_dtype=config.save.save_dtype,
        save_total_limit=config.save.save_total_limit,
        verbose=args.verbose
    )
    print("✓ 模型保存器创建成功")

    # 9. 开始训练
    print("\n【步骤9】开始训练...")
    print("=" * 70)

    try:
        for event_type, event_data in trainer.train(
            transformer=transformer,
            vae=vae,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
            train_dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        ):
            # 处理训练事件
            if event_type == "epoch_start":
                if args.verbose:
                    print(f"\n{'='*70}")
                    print(f"开始 Epoch {event_data + 1}/{config.train.num_epochs}")
                    print(f"{'='*70}")

            elif event_type == "step":
                # 步骤日志（由tqdm处理，这里只做额外记录）
                pass

            elif event_type == "save":
                # 保存检查点
                epoch = event_data
                if epoch is not None:
                    save_path = saver.save_lora(transformer, epoch=epoch)
                    if args.verbose:
                        print(f"\n✓ Epoch {epoch} 检查点已保存: {save_path}")
                else:
                    # 最终保存
                    save_path = saver.save_lora(transformer, epoch=None)
                    if args.verbose:
                        print(f"\n✓ 最终模型已保存: {save_path}")

            elif event_type == "epoch_end":
                if args.verbose:
                    print(f"{'='*70}")
                    print(f"Epoch {event_data + 1} 完成")
                    print(f"{'='*70}")

        # 训练完成
        print("\n" + "=" * 70)
        print("✅ 训练完成!")
        print("=" * 70)
        print(f"\n保存的LoRA权重位于: {config.save.output_dir}")
        print(f"可训练参数数量: {sum(p.numel() for p in lora_params):,}")
        print("\n后续步骤:")
        print("  1. 使用inference脚本测试模型")
        print("  2. 调整超参数并继续训练")
        print("  3. 评估模型性能")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        print("正在保存当前模型...")
        save_path = saver.save_lora(transformer, epoch=None)
        print(f"✓ 模型已保存: {save_path}")
        sys.exit(0)

    except Exception as e:
        print(f"\n\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

        # 尝试保存模型
        try:
            print("\n正在保存当前模型...")
            save_path = saver.save_lora(transformer, epoch=None)
            print(f"✓ 模型已保存: {save_path}")
        except:
            print("❌ 无法保存模型")

        sys.exit(1)


if __name__ == "__main__":
    main()
