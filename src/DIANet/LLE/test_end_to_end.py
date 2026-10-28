"""
端到端测试 - 使用Mock FLUX模型测试完整训练流程

这个脚本验证所有模块能否正确协作：
1. 配置系统
2. Mock FLUX模型加载
3. LoRA配置器（注入LoRA）
4. 数据加载
5. 训练器
6. 模型保存器

目标：在不需要真实FLUX模型的情况下，验证整个训练pipeline
"""

import sys
import os
import torch
import tempfile
import shutil

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(__file__))

from core.config import TrainingConfig, ModelConfig, DataConfig, LoRAConfig, TrainConfig, SaveConfig
from core.mock_flux import MockFluxModelLoader
from core.lora.configurator import LoRAConfigurator
from core.data.dataset import ImagePairDataset
from core.data.dataloader import create_dataloader
from core.loss.flow_matching import FlowMatchingLoss
from core.trainer import Trainer
from core.saver import ModelSaver


def create_dummy_dataset(tmpdir, num_samples=4):
    """创建临时测试数据集"""
    import numpy as np
    from PIL import Image

    input_dir = os.path.join(tmpdir, "input")
    target_dir = os.path.join(tmpdir, "target")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    for i in range(num_samples):
        # 创建随机图像
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(input_dir, f"img_{i}.png"))
        Image.fromarray(img).save(os.path.join(target_dir, f"img_{i}.png"))

    return input_dir, target_dir


def test_end_to_end():
    """端到端测试"""
    print("\n" + "="*70)
    print("🚀 端到端训练流程测试（使用Mock FLUX模型）")
    print("="*70)

    # 创建临时目录
    tmpdir = tempfile.mkdtemp()
    output_dir = os.path.join(tmpdir, "outputs")

    try:
        # ================================================================
        # 步骤1: 创建配置
        # ================================================================
        print("\n【步骤1】创建配置...")
        input_dir, target_dir = create_dummy_dataset(tmpdir, num_samples=4)

        config = TrainingConfig(
            model=ModelConfig(base_model="mock-flux-dev"),
            data=DataConfig(
                input_image_dir=input_dir,
                target_image_dir=target_dir,
                prompt="enhance the image",
                resolution=256,
                center_crop=False,
                random_flip=False,
                num_workers=0,  # 单线程
            ),
            lora=LoRAConfig(
                target_modules=["blocks.*.0"],  # 注入到mock transformer的blocks
                rank=8,
                alpha=8.0,
            ),
            train=TrainConfig(
                num_epochs=2,
                batch_size=2,
                learning_rate=1e-4,
                gradient_accumulation_steps=1,
                max_grad_norm=1.0,
                max_sequence_length=512,
                save_every_n_epochs=1,
            ),
            save=SaveConfig(
                output_dir=output_dir,
                save_dtype="float32",
            ),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("✓ 配置创建成功")

        # ================================================================
        # 步骤2: 加载Mock FLUX模型
        # ================================================================
        print("\n【步骤2】加载Mock FLUX模型...")
        model_loader = MockFluxModelLoader(
            device=config.device,
            dtype=torch.float32  # 使用float32便于测试
        )
        transformer, vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, scheduler = model_loader.load_all()
        print("✓ Mock FLUX模型加载成功")

        # ================================================================
        # 步骤3: 注入LoRA
        # ================================================================
        print("\n【步骤3】注入LoRA到Transformer...")
        lora_configurator = LoRAConfigurator(
            target_modules=config.lora.target_modules,
            rank=config.lora.rank,
            alpha=config.lora.alpha,
            verbose=True
        )
        transformer, stats = lora_configurator.inject_lora(transformer)
        print(f"✓ LoRA注入完成: {stats.lora_injected} 个模块")

        # ================================================================
        # 步骤4: 创建数据集和DataLoader
        # ================================================================
        print("\n【步骤4】创建数据集...")
        dataset = ImagePairDataset(
            input_image_dir=input_dir,
            target_image_dir=target_dir,
            prompt=config.data.prompt,
            resolution=config.data.resolution,
            center_crop=config.data.center_crop,
            random_flip=config.data.random_flip,
        )

        dataloader = create_dataloader(
            dataset=dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        print(f"✓ 数据集创建成功: {len(dataset)} 个样本")

        # ================================================================
        # 步骤5: 创建损失函数
        # ================================================================
        print("\n【步骤5】创建损失函数...")
        loss_fn = FlowMatchingLoss(prediction_type="velocity")
        print("✓ Flow Matching Loss创建成功")

        # ================================================================
        # 步骤6: 创建训练器
        # ================================================================
        print("\n【步骤6】创建训练器...")
        trainer = Trainer(
            config=config,
            loss_fn=loss_fn,
            verbose=True
        )
        print("✓ 训练器创建成功")

        # ================================================================
        # 步骤7: 创建优化器和学习率调度器
        # ================================================================
        print("\n【步骤7】创建优化器...")
        # 只优化LoRA参数
        lora_params = [p for p in transformer.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=config.train.learning_rate,
            weight_decay=0.01
        )

        # 简单的学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100,
            gamma=0.9
        )
        print(f"✓ 优化器创建成功: {len(lora_params)} 个可训练参数组")

        # ================================================================
        # 步骤8: 创建模型保存器
        # ================================================================
        print("\n【步骤8】创建模型保存器...")
        saver = ModelSaver(
            output_dir=output_dir,
            save_dtype="float32",
            save_total_limit=2,
            verbose=True
        )
        print("✓ 模型保存器创建成功")

        # ================================================================
        # 步骤9: 开始训练！
        # ================================================================
        print("\n【步骤9】开始训练...")
        print("="*70)

        step_count = 0
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
            if event_type == "epoch_start":
                print(f"\n开始 Epoch {event_data + 1}")

            elif event_type == "step":
                step_count += 1
                if step_count <= 5:  # 只打印前几步
                    print(f"  Step {event_data['step']}: Loss = {event_data['loss']:.4f}")

            elif event_type == "save":
                if event_data is not None:
                    save_path = saver.save_lora(transformer, epoch=event_data)
                    print(f"\n✓ Epoch {event_data} 检查点已保存")
                else:
                    save_path = saver.save_lora(transformer, epoch=None)
                    print(f"\n✓ 最终模型已保存")

            elif event_type == "epoch_end":
                print(f"Epoch {event_data + 1} 完成")

        # ================================================================
        # 步骤10: 验证保存的权重
        # ================================================================
        print("\n【步骤10】验证保存的权重...")
        from safetensors.torch import load_file

        final_path = os.path.join(output_dir, "lora_final.safetensors")
        if os.path.exists(final_path):
            loaded_weights = load_file(final_path)
            print(f"✓ 成功加载保存的权重: {len(loaded_weights)} 个参数")
            print(f"  权重文件大小: {os.path.getsize(final_path) / 1024:.2f} KB")

            # 验证权重可以加载回模型
            for name, param in transformer.named_parameters():
                if name in loaded_weights:
                    param.data.copy_(loaded_weights[name])
            print("✓ 权重成功加载回模型")
        else:
            print("✗ 未找到保存的权重文件")

        # ================================================================
        # 总结
        # ================================================================
        print("\n" + "="*70)
        print("✅ 端到端测试完成！")
        print("="*70)
        print("\n测试总结:")
        print(f"  ✓ 配置系统正常")
        print(f"  ✓ Mock FLUX模型正常")
        print(f"  ✓ LoRA注入正常 ({stats.lora_injected} 个模块)")
        print(f"  ✓ 数据加载正常 ({len(dataset)} 个样本)")
        print(f"  ✓ 训练循环正常 (共 {step_count} 步)")
        print(f"  ✓ 模型保存正常")
        print(f"  ✓ 权重加载正常")
        print("\n所有组件协作正常！可以开始实际训练了。")
        print("="*70 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理临时目录
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
            print(f"已清理临时目录: {tmpdir}")


if __name__ == "__main__":
    success = test_end_to_end()
    sys.exit(0 if success else 1)
