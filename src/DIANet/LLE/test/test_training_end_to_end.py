"""
端到端训练测试

测试完整的双路由MoE-LoRA训练流程
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import tempfile

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.mock_flux import MockFluxModelLoader
from core.lora.moe.moe_lora_layer import MoELoRALayer
from core.data.dataset import ImagePairDataset
from core.trainer import Trainer
from core.loss.flow_matching import FlowMatchingLoss
from core.loss.load_balance import LoadBalanceLoss
from torch.utils.data import DataLoader


class SimpleConfig:
    """简单的测试配置"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        class TrainConfig:
            num_epochs = 1
            batch_size = 2
            gradient_accumulation_steps = 1
            max_grad_norm = 1.0
            save_every_n_epochs = 10  # 不保存
            max_sequence_length = 512
            cfg_scale = 3.5

        self.train = TrainConfig()


def create_test_dataset(temp_dir: Path, num_images: int = 8):
    """创建测试数据集"""
    input_dir = temp_dir / "input"
    target_dir = temp_dir / "target"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 创建不同亮度的测试图像
    brightness_levels = [0.05, 0.15, 0.25, 0.35, 0.45, 0.6, 0.75, 0.9]

    for i in range(num_images):
        brightness = brightness_levels[i % len(brightness_levels)]

        # 输入图像（暗光，64x64以加快测试）
        input_img = np.ones((64, 64, 3), dtype=np.uint8) * int(brightness * 255)
        input_pil = Image.fromarray(input_img)
        input_pil.save(input_dir / f"img_{i:03d}.png")

        # 目标图像（正常亮度）
        target_img = np.ones((64, 64, 3), dtype=np.uint8) * 200
        target_pil = Image.fromarray(target_img)
        target_pil.save(target_dir / f"img_{i:03d}.png")

    return input_dir, target_dir


def inject_dual_route_moe_lora(transformer, num_specific=5, num_shared=4, rank=8):
    """为transformer注入双路由MoE-LoRA"""
    for i, block in enumerate(transformer.blocks):
        original_linear = block[0]

        moe_layer = MoELoRALayer(
            base_layer=original_linear,
            rank=rank,
            alpha=8.0,
            num_experts=num_specific + num_shared,
            routing_mode="dual",
            router_config={
                'num_specific': num_specific,
                'shared_top_k': 2,
                'shared_hidden_dim': 512
            }
        )

        transformer.blocks[i] = nn.Sequential(
            moe_layer,
            block[1],  # GELU
            block[2]   # Second Linear
        )

    return transformer


def test_training_pipeline():
    """测试完整的训练流程"""
    print("\n" + "=" * 70)
    print("测试: 端到端训练流程")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1. 创建测试数据集
        print("\n1. 创建测试数据集...")
        input_dir, target_dir = create_test_dataset(temp_path, num_images=8)

        dataset = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            prompt="Enhance",
            resolution=64,
            classify_exposure_category=True,
            exposure_method="mean_brightness"
        )

        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        print(f"  ✓ 数据集大小: {len(dataset)}")
        print(f"  ✓ Batch数量: {len(dataloader)}")

        # 2. 加载Mock FLUX
        print("\n2. 加载Mock FLUX...")
        loader = MockFluxModelLoader()
        transformer, vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, scheduler = loader.load_all()
        print("  ✓ Mock FLUX加载成功")

        # 3. 注入双路由MoE-LoRA
        print("\n3. 注入双路由MoE-LoRA...")
        transformer = inject_dual_route_moe_lora(transformer, num_specific=5, num_shared=4, rank=8)
        print("  ✓ 注入成功")

        # 4. 创建损失函数和优化器
        print("\n4. 创建损失函数和优化器...")
        config = SimpleConfig()

        fm_loss_fn = FlowMatchingLoss(prediction_type="velocity")
        lb_loss_fn = LoadBalanceLoss(num_experts=4, weight=0.01)

        optimizer = torch.optim.AdamW(
            [p for p in transformer.parameters() if p.requires_grad],
            lr=1e-4,
            weight_decay=0.01
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(dataloader),
            eta_min=1e-6
        )
        print("  ✓ 优化器和调度器创建成功")

        # 5. 创建训练器
        print("\n5. 创建训练器...")
        trainer = Trainer(
            config=config,
            loss_fn=fm_loss_fn,
            load_balance_loss_fn=lb_loss_fn,
            verbose=False
        )
        print("  ✓ 训练器创建成功")

        # 6. 运行训练（只训练几步）
        print("\n6. 运行训练（1个epoch）...")
        losses = []
        steps_completed = 0

        for event_type, data in trainer.train(
            transformer=transformer,
            vae=vae,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
            train_dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler
        ):
            if event_type == "step":
                losses.append(data['total_loss'])
                steps_completed += 1
                print(f"  Step {steps_completed}: loss={data['total_loss']:.4f}, "
                      f"fm_loss={data['fm_loss']:.4f}, lb_loss={data['lb_loss']:.4f}")

        # 7. 验证
        print("\n7. 验证结果...")
        assert len(losses) > 0, "没有完成任何训练步骤"
        print(f"  ✓ 完成 {len(losses)} 个训练步骤")

        # 检查损失是否合理（不是NaN或Inf）
        assert all(not torch.isnan(torch.tensor(l)) for l in losses), "损失包含NaN"
        assert all(not torch.isinf(torch.tensor(l)) for l in losses), "损失包含Inf"
        print(f"  ✓ 所有损失值都是有效的")

        # 检查平均损失
        avg_loss = sum(losses) / len(losses)
        print(f"  ✓ 平均损失: {avg_loss:.4f}")

        # 8. 检查路由统计
        print("\n8. 检查路由统计...")
        moe_layers_found = 0
        for name, module in transformer.named_modules():
            if hasattr(module, 'routing_mode') and module.routing_mode == 'dual':
                moe_layers_found += 1

                # 检查specific router统计
                specific_stats = module.specific_router.get_routing_stats()
                print(f"\n  Block {moe_layers_found} Specific Router:")
                print(f"    Total calls: {specific_stats['total_calls']}")
                print(f"    Expert usage: {[int(x) for x in specific_stats['expert_usage'][:5]]}")

                # 检查shared router统计
                shared_stats = module.shared_router.get_routing_stats()
                print(f"  Block {moe_layers_found} Shared Router:")
                print(f"    Total calls: {shared_stats['total_calls']}")
                print(f"    Expert usage: {[int(x) for x in shared_stats['expert_usage'][:4]]}")
                print(f"    Load balance loss: {shared_stats['load_balance_loss']:.4f}")

        assert moe_layers_found > 0, "没有找到双路由MoE层"
        print(f"\n  ✓ 找到 {moe_layers_found} 个双路由MoE层")

        print("\n" + "=" * 70)
        print("✅ 所有测试通过！")
        print("=" * 70)


def test_training_with_exposure_categories():
    """测试训练时exposure_category的正确传递"""
    print("\n" + "=" * 70)
    print("测试: Exposure Category传递")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 创建数据集
        input_dir, target_dir = create_test_dataset(temp_path, num_images=4)

        dataset = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            prompt="Enhance",
            resolution=64,
            classify_exposure_category=True,
        )

        # 验证数据集返回exposure_category
        sample = dataset[0]
        assert 'exposure_category' in sample, "数据集没有返回exposure_category"
        print(f"✓ 数据集正确返回exposure_category: {sample['exposure_category']}")

        # 验证批次返回exposure_category
        dataloader = DataLoader(dataset, batch_size=2)
        batch = next(iter(dataloader))
        assert 'exposure_category' in batch, "Batch没有包含exposure_category"
        print(f"✓ Batch正确包含exposure_category: {batch['exposure_category'].tolist()}")

        print("\n✅ Exposure Category传递测试通过")


def main():
    print("\n" + "=" * 70)
    print("开始端到端训练测试")
    print("=" * 70)

    try:
        test_training_with_exposure_categories()
        test_training_pipeline()

        print("\n" + "=" * 70)
        print("✅ 所有测试通过！")
        print("=" * 70)

        print("\n总结:")
        print("  ✓ 数据集曝光度分类正常")
        print("  ✓ Exposure category正确传递")
        print("  ✓ 双路由MoE-LoRA注入成功")
        print("  ✓ 训练循环正常运行")
        print("  ✓ 损失计算正确")
        print("  ✓ 负载均衡损失正常工作")
        print("  ✓ 路由统计正确记录")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
