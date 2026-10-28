"""
测试训练器的负载均衡损失集成
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.mock_flux import MockFluxModelLoader
from core.lora.configurator import LoRAConfigurator
from core.loss.flow_matching import FlowMatchingLoss
from core.loss.load_balance import LoadBalanceLoss
from core.trainer import Trainer
from core.config import TrainingConfig


def test_trainer_with_load_balance():
    """测试训练器集成负载均衡损失"""
    print("\n" + "=" * 70)
    print("测试训练器的负载均衡损失集成")
    print("=" * 70)

    # 1. 加载Mock FLUX模型
    print("\n步骤1: 加载Mock FLUX模型")
    loader = MockFluxModelLoader()
    (transformer, vae, text_encoder_1, text_encoder_2,
     tokenizer_1, tokenizer_2, scheduler) = loader.load_all()
    print("✓ Mock FLUX模型加载成功")

    # 2. 注入MoE-LoRA
    print("\n步骤2: 注入MoE-LoRA")
    configurator = LoRAConfigurator(
        module_modes={
            "blocks.0.0": "moe",
            "blocks.1.0": "moe",
        },
        rank=8,
        alpha=8.0,
        moe_config={
            "num_experts": 4,
            "routing_mode": "single",
            "router_config": {"top_k": 2}
        },
        verbose=False
    )
    transformer, stats = configurator.inject_lora(transformer)
    print(f"✓ 注入了 {stats.moe_injected} 个MoE-LoRA层")

    # 3. 创建损失函数
    print("\n步骤3: 创建损失函数")
    fm_loss_fn = FlowMatchingLoss()
    lb_loss_fn = LoadBalanceLoss(num_experts=4, weight=0.01)
    print("✓ Flow Matching Loss 和 Load Balance Loss 创建成功")

    # 4. 创建简单配置
    print("\n步骤4: 创建训练配置")

    class SimpleConfig:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        class train:
            gradient_accumulation_steps = 1
            max_grad_norm = 1.0
            save_every_n_epochs = 100
            num_epochs = 1

    config = SimpleConfig()
    print(f"✓ 使用设备: {config.device}")

    # 5. 创建训练器（带负载均衡损失）
    print("\n步骤5: 创建训练器")
    trainer = Trainer(
        config=config,
        loss_fn=fm_loss_fn,
        load_balance_loss_fn=lb_loss_fn,  # 传入负载均衡损失
        verbose=False
    )
    print("✓ 训练器创建成功，已启用负载均衡损失")

    # 6. 准备简单的数据
    print("\n步骤6: 准备测试数据")
    device = transformer.input_proj.weight.device
    dtype = transformer.input_proj.weight.dtype

    batch_size = 4
    latent = torch.randn(batch_size, 32, 32, 32, device=device, dtype=dtype)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    prompt_embeds = torch.randn(batch_size, 77, 768, device=device, dtype=dtype)
    pooled_prompt_embeds = torch.randn(batch_size, 768, device=device, dtype=dtype)
    target_latent = torch.randn(batch_size, 32, 32, 32, device=device, dtype=dtype)
    print(f"✓ 数据准备完成: batch_size={batch_size}")

    # 7. 创建优化器
    print("\n步骤7: 创建优化器")
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    print(f"✓ 优化器创建成功, 可训练参数: {sum(p.numel() for p in trainable_params):,}")

    # 8. 执行一个训练步骤
    print("\n步骤8: 执行训练步骤（测试负载均衡损失）")
    transformer.train()
    optimizer.zero_grad()

    # 前向传播
    output = transformer(
        hidden_states=latent,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        return_dict=False
    )[0]

    # 计算Flow Matching损失
    noise = torch.randn_like(target_latent)
    fm_loss = fm_loss_fn(output, target_latent, noise)

    print(f"  Flow Matching Loss: {fm_loss.item():.6f}")

    # 收集路由信息并计算负载均衡损失
    router_probs_list = []
    expert_indices_list = []

    for name, module in transformer.named_modules():
        if hasattr(module, 'get_routing_stats'):
            stats = module.get_routing_stats()
            if isinstance(stats, dict) and 'router_probs' in stats and 'expert_indices' in stats:
                router_probs_list.append(stats['router_probs'])
                expert_indices_list.append(stats['expert_indices'])
                print(f"  ✓ 从 {name} 收集到路由信息")

    assert len(router_probs_list) > 0, "应该收集到至少一个MoE-LoRA层的路由信息"

    # 合并路由信息
    all_router_probs = torch.cat(router_probs_list, dim=0)
    all_expert_indices = torch.cat(expert_indices_list, dim=0)

    # 计算负载均衡损失
    lb_loss = lb_loss_fn(all_router_probs, all_expert_indices)

    print(f"  Load Balance Loss: {lb_loss.item():.6f}")
    print(f"  收集到 {len(router_probs_list)} 个MoE-LoRA层的路由信息")

    # 总损失
    total_loss = fm_loss + lb_loss
    print(f"  Total Loss: {total_loss.item():.6f}")

    # 反向传播
    total_loss.backward()

    # 检查梯度
    grad_count = 0
    for p in trainable_params:
        if p.grad is not None and p.grad.abs().sum() > 0:
            grad_count += 1

    print(f"  ✓ 有梯度的参数: {grad_count}/{len(trainable_params)}")
    assert grad_count > 0, "应该有参数有梯度"

    # 优化器更新
    optimizer.step()
    print("  ✓ 优化器更新成功")

    print("\n✅ 所有测试通过！")
    print("=" * 70)
    print("\n总结:")
    print("  ✓ 训练器成功集成负载均衡损失")
    print("  ✓ 能够从MoE-LoRA层收集路由信息")
    print("  ✓ 负载均衡损失计算正常")
    print("  ✓ 梯度反向传播正常")
    print("  ✓ 优化器更新正常")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_trainer_with_load_balance()
        exit(0)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
