"""
简单的双路由MoE-LoRA训练脚本

使用Mock FLUX进行快速测试训练流程
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.mock_flux import MockFluxModelLoader
from core.lora.moe.moe_lora_layer import MoELoRALayer
from core.data.dataset import ImagePairDataset
from core.trainer import Trainer
from core.loss.flow_matching import FlowMatchingLoss
from core.loss.load_balance import LoadBalanceLoss


class SimpleConfig:
    """简单的配置类"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 训练配置
        class TrainConfig:
            num_epochs = 2
            batch_size = 2
            gradient_accumulation_steps = 1
            max_grad_norm = 1.0
            save_every_n_epochs = 1
            max_sequence_length = 512
            cfg_scale = 3.5

        self.train = TrainConfig()


def inject_dual_route_moe_lora(transformer, num_specific=5, num_shared=4, rank=8):
    """
    为transformer注入双路由MoE-LoRA

    Args:
        transformer: Mock FLUX transformer
        num_specific: Specific专家数量
        num_shared: Shared专家数量
        rank: LoRA rank
    """
    print(f"\n注入双路由MoE-LoRA:")
    print(f"  Specific专家: {num_specific}")
    print(f"  Shared专家: {num_shared}")
    print(f"  总专家: {num_specific + num_shared}")
    print(f"  LoRA rank: {rank}")

    num_injected = 0

    # 为每个block的第一个线性层注入MoE-LoRA
    for i, block in enumerate(transformer.blocks):
        # block是Sequential(Linear, GELU, Linear)
        original_linear = block[0]

        # 创建MoE-LoRA层
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

        # 替换原始层
        transformer.blocks[i] = nn.Sequential(
            moe_layer,
            block[1],  # GELU
            block[2]   # Second Linear
        )

        num_injected += 1

    print(f"  ✓ 成功注入 {num_injected} 个MoE-LoRA层\n")

    return transformer


def main():
    print("\n" + "=" * 70)
    print("双路由MoE-LoRA简单训练脚本")
    print("=" * 70)

    # 1. 配置
    config = SimpleConfig()
    print(f"\n设备: {config.device}")

    # 2. 加载Mock FLUX模型
    print("\n" + "=" * 70)
    print("加载Mock FLUX模型")
    print("=" * 70)
    loader = MockFluxModelLoader()
    transformer, vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, scheduler = loader.load_all()

    # 3. 注入双路由MoE-LoRA
    print("\n" + "=" * 70)
    print("注入双路由MoE-LoRA")
    print("=" * 70)
    transformer = inject_dual_route_moe_lora(transformer, num_specific=5, num_shared=4, rank=8)

    # 4. 创建测试数据集
    print("\n" + "=" * 70)
    print("创建测试数据集")
    print("=" * 70)

    # 检查是否有真实数据
    data_dir = Path("./data/train")
    if data_dir.exists() and (data_dir / "input").exists() and (data_dir / "target").exists():
        print(f"使用真实数据: {data_dir}")
        dataset = ImagePairDataset(
            input_image_dir=str(data_dir / "input"),
            target_image_dir=str(data_dir / "target"),
            prompt="Enhance the low-light image",
            resolution=256,
            classify_exposure_category=True,
            exposure_method="mean_brightness",
            exposure_thresholds=[0.1, 0.2, 0.3, 0.5]
        )
    else:
        print("⚠ 警告: 未找到真实数据，请创建测试数据集")
        print(f"  请确保以下目录存在并包含图像:")
        print(f"    {data_dir / 'input'}")
        print(f"    {data_dir / 'target'}")
        sys.exit(1)

    # 5. 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if config.device == "cuda" else False
    )

    print(f"✓ 数据集大小: {len(dataset)}")
    print(f"✓ Batch数量: {len(dataloader)}")

    # 6. 创建损失函数
    print("\n" + "=" * 70)
    print("创建损失函数")
    print("=" * 70)

    fm_loss_fn = FlowMatchingLoss(prediction_type="velocity")
    lb_loss_fn = LoadBalanceLoss(num_experts=4, weight=0.01)  # 只对4个shared专家

    print("✓ Flow Matching Loss")
    print("✓ Load Balance Loss (仅用于Shared专家)")

    # 7. 创建优化器
    print("\n" + "=" * 70)
    print("创建优化器")
    print("=" * 70)

    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01
    )

    # 简单的学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.train.num_epochs * len(dataloader),
        eta_min=1e-6
    )

    print("✓ AdamW optimizer (lr=1e-4)")
    print("✓ Cosine Annealing LR scheduler")

    # 8. 创建训练器
    print("\n" + "=" * 70)
    print("创建训练器")
    print("=" * 70)

    trainer = Trainer(
        config=config,
        loss_fn=fm_loss_fn,
        load_balance_loss_fn=lb_loss_fn,
        verbose=True
    )

    print("✓ Trainer创建成功")

    # 9. 开始训练
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)

    try:
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
            if event_type == "save":
                # 保存检查点
                epoch = data
                if epoch is not None:
                    save_path = f"checkpoints/dual_route_epoch_{epoch}.pt"
                    Path("checkpoints").mkdir(exist_ok=True)

                    # 保存MoE-LoRA权重
                    state_dict = {}
                    for name, module in transformer.named_modules():
                        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                            state_dict[f"{name}.lora_A"] = [p.data for p in module.lora_A]
                            state_dict[f"{name}.lora_B"] = [p.data for p in module.lora_B]

                    torch.save(state_dict, save_path)
                    print(f"\n✓ 保存检查点: {save_path}")

        print("\n" + "=" * 70)
        print("✅ 训练完成！")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠ 训练被用户中断")
    except Exception as e:
        print(f"\n\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
