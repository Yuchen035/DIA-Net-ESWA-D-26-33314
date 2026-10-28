"""
测试双路由路由器（SpecificRouter + SharedRouter）
"""

import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.lora.moe.router import SpecificRouter, SharedRouter


def test_specific_router():
    """测试SpecificRouter"""
    print("\n" + "=" * 70)
    print("测试1: SpecificRouter（曝光度硬编码路由）")
    print("=" * 70)

    # 创建router，假设有4个曝光类别
    router = SpecificRouter(num_experts=4)
    print(f"✓ 创建SpecificRouter，专家数: 4")

    # 准备输入
    batch_size = 8
    x = torch.randn(batch_size, 256, 512)  # [batch, seq, dim]

    # 曝光类别：[0, 1, 2, 3, 0, 1, 2, 3]（4个类别）
    exposure_categories = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)

    # 前向传播
    router.train()
    weights, indices = router(x, exposure_category=exposure_categories)

    print(f"\n输入:")
    print(f"  batch_size: {batch_size}")
    print(f"  曝光类别: {exposure_categories.tolist()}")

    print(f"\n输出:")
    print(f"  weights shape: {weights.shape}")  # [batch, 1]
    print(f"  indices shape: {indices.shape}")  # [batch, 1]
    print(f"  indices: {indices.squeeze().tolist()}")

    # 验证
    assert weights.shape == (batch_size, 1)
    assert indices.shape == (batch_size, 1)
    assert torch.all(weights == 1.0), "SpecificRouter的权重应该全为1.0"
    assert torch.all(indices.squeeze() == exposure_categories), "索引应该等于曝光类别"

    print(f"\n✓ 权重全为1.0")
    print(f"✓ 索引等于曝光类别（硬编码路由）")

    # 检查统计
    stats = router.get_routing_stats()
    print(f"\n路由统计:")
    print(f"  总调用次数: {stats['total_calls']}")
    print(f"  专家使用次数: {stats['expert_usage']}")
    print(f"  专家使用率: {[f'{r:.2f}' for r in stats['expert_usage_ratio']]}")

    # 每个类别应该被调用2次
    assert stats['total_calls'] == batch_size
    assert stats['expert_usage'] == [2.0, 2.0, 2.0, 2.0]

    print("✅ SpecificRouter测试通过")


def test_shared_router():
    """测试SharedRouter"""
    print("\n" + "=" * 70)
    print("测试2: SharedRouter（共享专家动态路由）")
    print("=" * 70)

    # 创建router
    input_dim = 512
    num_experts = 4
    top_k = 2

    router = SharedRouter(
        input_dim=input_dim,
        num_experts=num_experts,
        top_k=top_k,
        hidden_dim=256
    )
    print(f"✓ 创建SharedRouter")
    print(f"  输入维度: {input_dim}")
    print(f"  专家数: {num_experts}")
    print(f"  Top-K: {top_k}")

    # 准备输入
    batch_size = 8
    seq_len = 256
    x = torch.randn(batch_size, seq_len, input_dim)

    # 前向传播
    router.train()
    weights, indices = router(x)

    print(f"\n输入:")
    print(f"  shape: {x.shape}")

    print(f"\n输出:")
    print(f"  weights shape: {weights.shape}")  # [batch, top_k]
    print(f"  indices shape: {indices.shape}")  # [batch, top_k]
    print(f"  第一个样本:")
    print(f"    indices: {indices[0].tolist()}")
    print(f"    weights: {[f'{w:.3f}' for w in weights[0].tolist()]}")
    print(f"    权重和: {weights[0].sum():.3f}")

    # 验证
    assert weights.shape == (batch_size, top_k)
    assert indices.shape == (batch_size, top_k)

    # 验证权重归一化
    weights_sum = weights.sum(dim=-1)
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum)), "权重应该归一化到1"

    print(f"\n✓ 输出形状正确")
    print(f"✓ 权重已归一化（和为1.0）")

    # 检查统计
    stats = router.get_routing_stats()
    print(f"\n路由统计:")
    print(f"  总调用次数: {stats['total_calls']}")
    print(f"  专家使用次数: {stats['expert_usage']}")
    print(f"  专家使用率: {[f'{r:.3f}' for r in stats['expert_usage_ratio']]}")
    print(f"  负载均衡损失: {stats['load_balance_loss']:.4f}")

    assert stats['total_calls'] == batch_size * top_k

    print("✅ SharedRouter测试通过")


def test_dual_route_combination():
    """测试双路由组合（Specific + Shared）"""
    print("\n" + "=" * 70)
    print("测试3: 双路由组合（Specific + Shared协同工作）")
    print("=" * 70)

    # 场景：4个specific专家 + 4个shared专家 = 8个总专家
    num_specific = 4
    num_shared = 4
    input_dim = 512

    specific_router = SpecificRouter(num_experts=num_specific)
    shared_router = SharedRouter(
        input_dim=input_dim,
        num_experts=num_shared,
        top_k=2
    )

    print(f"✓ 创建双路由器")
    print(f"  Specific专家数: {num_specific}")
    print(f"  Shared专家数: {num_shared}")
    print(f"  总专家数: {num_specific + num_shared}")

    # 准备输入
    batch_size = 4
    x = torch.randn(batch_size, 256, input_dim)
    exposure_categories = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    # Specific路由
    specific_weights, specific_indices = specific_router(
        x, exposure_category=exposure_categories
    )

    # Shared路由
    shared_weights, shared_indices = shared_router(x)

    print(f"\n输入:")
    print(f"  batch_size: {batch_size}")
    print(f"  曝光类别: {exposure_categories.tolist()}")

    print(f"\nSpecific路由输出:")
    print(f"  indices: {specific_indices.squeeze().tolist()}")
    print(f"  weights: {specific_weights.squeeze().tolist()}")

    print(f"\nShared路由输出:")
    print(f"  indices (第一个样本): {shared_indices[0].tolist()}")
    print(f"  weights (第一个样本): {[f'{w:.3f}' for w in shared_weights[0].tolist()]}")

    # 模拟组合输出（在实际MoE-LoRA层中的处理）
    print(f"\n双路由组合逻辑:")
    print(f"  1. Specific路由选择 1 个专家（基于曝光类别）")
    print(f"  2. Shared路由选择 2 个专家（动态学习）")
    print(f"  3. 总共激活 3 个专家（1 specific + 2 shared）")

    # 验证索引不冲突
    # Specific使用专家0-3，Shared应该使用专家4-7（索引偏移num_specific）
    print(f"\n✓ Specific专家索引范围: [0, {num_specific-1}]")
    print(f"✓ Shared专家索引范围: [0, {num_shared-1}] (在MoE层会偏移+{num_specific})")

    assert specific_indices.max() < num_specific
    assert shared_indices.max() < num_shared

    print("✅ 双路由组合测试通过")


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 70)
    print("测试4: 边界情况")
    print("=" * 70)

    # 测试1: SpecificRouter没有exposure_category
    print("\n子测试1: SpecificRouter缺少exposure_category参数")
    router = SpecificRouter(num_experts=3)
    x = torch.randn(4, 100, 256)

    try:
        weights, indices = router(x)  # 缺少exposure_category
        assert False, "应该抛出ValueError"
    except ValueError as e:
        print(f"  ✓ 正确抛出异常: {e}")

    # 测试2: 曝光类别超出范围
    print("\n子测试2: 曝光类别超出范围")
    exposure_categories = torch.tensor([0, 1, 5])  # 5超出范围[0-2]

    try:
        weights, indices = router(x, exposure_category=exposure_categories)
        assert False, "应该抛出ValueError"
    except ValueError as e:
        print(f"  ✓ 正确抛出异常: {str(e)[:60]}...")

    # 测试3: 重置统计
    print("\n子测试3: 重置路由统计")
    router = SharedRouter(input_dim=256, num_experts=4, top_k=2)
    router.train()

    # 执行几次前向传播
    for _ in range(3):
        x = torch.randn(4, 100, 256)
        router(x)

    stats_before = router.get_routing_stats()
    print(f"  重置前总调用: {stats_before['total_calls']}")

    router.reset_stats()
    stats_after = router.get_routing_stats()
    print(f"  重置后总调用: {stats_after['total_calls']}")

    assert stats_after['total_calls'] == 0
    print(f"  ✓ 统计已重置")

    print("✅ 边界情况测试通过")


def main():
    print("\n" + "=" * 70)
    print("开始测试双路由路由器")
    print("=" * 70)

    try:
        test_specific_router()
        test_shared_router()
        test_dual_route_combination()
        test_edge_cases()

        print("\n" + "=" * 70)
        print("✅ 所有测试通过！")
        print("=" * 70)

        print("\n总结:")
        print("  ✓ SpecificRouter（硬编码路由）正常工作")
        print("  ✓ SharedRouter（动态学习路由）正常工作")
        print("  ✓ 双路由组合逻辑正确")
        print("  ✓ 边界情况处理正常")
        print("  ✓ 路由统计功能正常")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
