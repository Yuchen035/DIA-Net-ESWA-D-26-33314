"""
Router简单测试（不依赖pytest）
"""

import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.lora.moe.router import (
    StandardRouter,
    SpecificRouter,
    SharedRouter
)


def test_standard_router():
    """测试StandardRouter"""
    print("\n=== 测试 StandardRouter ===")

    router = StandardRouter(
        input_dim=128,
        num_experts=4,
        top_k=2
    )

    # 测试3D输入
    x = torch.randn(8, 16, 128)
    weights, indices = router(x)

    print(f"输入形状: {x.shape}")
    print(f"权重形状: {weights.shape}")
    print(f"索引形状: {indices.shape}")
    print(f"权重和: {weights.sum(dim=1)}")
    print(f"索引范围: [{indices.min()}, {indices.max()}]")

    # 验证
    assert weights.shape == (8, 2), f"权重形状错误: {weights.shape}"
    assert indices.shape == (8, 2), f"索引形状错误: {indices.shape}"
    assert torch.allclose(weights.sum(dim=1), torch.ones(8), atol=1e-5), "权重未归一化"
    assert (indices >= 0).all() and (indices < 4).all(), "索引超出范围"

    # 测试路由统计
    router.reset_stats()  # 重置统计信息
    router.train()
    x = torch.randn(10, 5, 128)
    weights, indices = router(x)

    stats = router.get_routing_stats()
    print(f"\n路由统计:")
    print(f"  总调用次数: {stats['total_calls']}")
    print(f"  专家使用次数: {stats['expert_usage']}")
    print(f"  专家使用率: {[f'{r:.3f}' for r in stats['expert_usage_ratio']]}")
    print(f"  负载均衡损失: {stats['load_balance_loss']:.4f}")

    assert stats['total_calls'] == 20, f"调用次数错误: {stats['total_calls']}"

    print("✅ StandardRouter 测试通过")


def test_specific_router():
    """测试SpecificRouter"""
    print("\n=== 测试 SpecificRouter ===")

    router = SpecificRouter(num_experts=3)

    # 测试带曝光度类别
    x = torch.randn(12, 10, 128)
    exposure_category = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    weights, indices = router(x, exposure_category=exposure_category)

    print(f"输入形状: {x.shape}")
    print(f"曝光度类别: {exposure_category}")
    print(f"权重形状: {weights.shape}")
    print(f"索引形状: {indices.shape}")
    print(f"权重: {weights.squeeze(1)}")
    print(f"索引: {indices.squeeze(1)}")

    # 验证
    assert weights.shape == (12, 1), f"权重形状错误: {weights.shape}"
    assert indices.shape == (12, 1), f"索引形状错误: {indices.shape}"
    assert torch.all(weights == 1.0), "权重不全为1.0"
    assert torch.equal(indices.squeeze(1), exposure_category), "索引与类别不一致"

    # 测试路由统计
    router.reset_stats()  # 重置统计信息
    router.train()
    # 再次前向传播以产生统计
    weights, indices = router(x, exposure_category=exposure_category)

    stats = router.get_routing_stats()
    print(f"\n路由统计:")
    print(f"  总调用次数: {stats['total_calls']}")
    print(f"  专家使用次数: {stats['expert_usage']}")
    print(f"  专家使用率: {[f'{r:.3f}' for r in stats['expert_usage_ratio']]}")

    assert stats['total_calls'] == 12, f"调用次数错误: {stats['total_calls']}"
    assert stats['expert_usage'] == [4.0, 4.0, 4.0], f"专家使用次数错误: {stats['expert_usage']}"

    # 测试缺少category时报错
    try:
        router(x)
        assert False, "应该抛出ValueError"
    except ValueError as e:
        print(f"\n✅ 正确抛出错误: {e}")

    print("✅ SpecificRouter 测试通过")


def test_shared_router():
    """测试SharedRouter"""
    print("\n=== 测试 SharedRouter ===")

    router = SharedRouter(
        input_dim=128,
        num_experts=4,
        top_k=2,
        hidden_dim=256
    )

    # 测试前向传播
    x = torch.randn(8, 10, 128)
    weights, indices = router(x)

    print(f"输入形状: {x.shape}")
    print(f"权重形状: {weights.shape}")
    print(f"索引形状: {indices.shape}")
    print(f"权重和: {weights.sum(dim=1)}")
    print(f"索引范围: [{indices.min()}, {indices.max()}]")

    # 验证
    assert weights.shape == (8, 2), f"权重形状错误: {weights.shape}"
    assert indices.shape == (8, 2), f"索引形状错误: {indices.shape}"
    assert torch.allclose(weights.sum(dim=1), torch.ones(8), atol=1e-5), "权重未归一化"
    assert (indices >= 0).all() and (indices < 4).all(), "索引超出范围"

    # 测试梯度流
    x_grad = torch.randn(4, 10, 128, requires_grad=True)
    weights, indices = router(x_grad)
    loss = weights.sum()
    loss.backward()

    assert x_grad.grad is not None, "输入梯度为None"
    assert not torch.all(x_grad.grad == 0), "输入梯度全为0"

    param_count = sum(p.numel() for p in router.parameters())
    print(f"\n可学习参数数量: {param_count}")

    # 验证参数有梯度
    for name, param in router.named_parameters():
        assert param.grad is not None, f"参数 {name} 的梯度为None"

    print("✅ 梯度流测试通过")

    # 测试路由统计
    router.reset_stats()  # 重置统计信息
    router.train()
    x = torch.randn(10, 5, 128)
    weights, indices = router(x)

    stats = router.get_routing_stats()
    print(f"\n路由统计:")
    print(f"  总调用次数: {stats['total_calls']}")
    print(f"  专家使用次数: {stats['expert_usage']}")
    print(f"  专家使用率: {[f'{r:.3f}' for r in stats['expert_usage_ratio']]}")
    print(f"  负载均衡损失: {stats['load_balance_loss']:.4f}")

    print("✅ SharedRouter 测试通过")


def test_cuda_compatibility():
    """测试CUDA兼容性"""
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA不可用，跳过CUDA测试")
        return

    print("\n=== 测试 CUDA 兼容性 ===")

    # StandardRouter
    router = StandardRouter(128, 4, top_k=2).cuda()
    x = torch.randn(4, 10, 128).cuda()
    weights, indices = router(x)

    assert weights.device.type == 'cuda', "StandardRouter权重不在CUDA上"
    assert indices.device.type == 'cuda', "StandardRouter索引不在CUDA上"
    print("✅ StandardRouter CUDA测试通过")

    # SpecificRouter
    router = SpecificRouter(num_experts=3).cuda()
    x = torch.randn(4, 10, 128).cuda()
    exposure_category = torch.randint(0, 3, (4,)).cuda()
    weights, indices = router(x, exposure_category=exposure_category)

    assert weights.device.type == 'cuda', "SpecificRouter权重不在CUDA上"
    assert indices.device.type == 'cuda', "SpecificRouter索引不在CUDA上"
    print("✅ SpecificRouter CUDA测试通过")

    # SharedRouter
    router = SharedRouter(128, 4, top_k=2).cuda()
    x = torch.randn(4, 10, 128).cuda()
    weights, indices = router(x)

    assert weights.device.type == 'cuda', "SharedRouter权重不在CUDA上"
    assert indices.device.type == 'cuda', "SharedRouter索引不在CUDA上"
    print("✅ SharedRouter CUDA测试通过")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("开始Router测试")
    print("=" * 60)

    try:
        test_standard_router()
        test_specific_router()
        test_shared_router()
        test_cuda_compatibility()

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
