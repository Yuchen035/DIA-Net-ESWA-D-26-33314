"""
Router单元测试

测试所有路由器组件：
- BaseRouter抽象基类
- StandardRouter（单路由模式）
- SpecificRouter（双路由模式，硬编码）
- SharedRouter（双路由模式，可学习）
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.lora.moe.router import (
    BaseRouter,
    StandardRouter,
    SpecificRouter,
    SharedRouter
)


class TestStandardRouter:
    """测试StandardRouter"""

    def test_init(self):
        """测试初始化"""
        router = StandardRouter(
            input_dim=128,
            num_experts=4,
            top_k=2
        )

        assert router.input_dim == 128
        assert router.num_experts == 4
        assert router.top_k == 2
        assert router.gate.in_features == 128
        assert router.gate.out_features == 4

    def test_forward_2d_input(self):
        """测试2D输入"""
        router = StandardRouter(
            input_dim=128,
            num_experts=4,
            top_k=2
        )

        # 2D输入：[batch_size, input_dim] - 不推荐，需要unsqueeze
        x = torch.randn(8, 128)
        x_3d = x.unsqueeze(1)  # [8, 1, 128]

        weights, indices = router(x_3d)

        # 检查输出形状
        assert weights.shape == (8, 2), f"Expected (8, 2), got {weights.shape}"
        assert indices.shape == (8, 2), f"Expected (8, 2), got {indices.shape}"

        # 检查权重归一化
        assert torch.allclose(weights.sum(dim=1), torch.ones(8), atol=1e-5)

        # 检查索引范围
        assert (indices >= 0).all()
        assert (indices < 4).all()

    def test_forward_3d_input(self):
        """测试3D输入"""
        router = StandardRouter(
            input_dim=128,
            num_experts=4,
            top_k=2
        )

        # 3D输入：[batch_size, seq_len, input_dim]
        x = torch.randn(8, 16, 128)

        weights, indices = router(x)

        # 检查输出形状
        assert weights.shape == (8, 2)
        assert indices.shape == (8, 2)

        # 检查权重归一化
        assert torch.allclose(weights.sum(dim=1), torch.ones(8), atol=1e-5)

    def test_top_k_values(self):
        """测试不同的top_k值"""
        for top_k in [1, 2, 3]:
            router = StandardRouter(
                input_dim=64,
                num_experts=4,
                top_k=top_k
            )

            x = torch.randn(4, 10, 64)
            weights, indices = router(x)

            assert weights.shape == (4, top_k)
            assert indices.shape == (4, top_k)

    def test_noisy_gating(self):
        """测试噪声门控"""
        router = StandardRouter(
            input_dim=128,
            num_experts=4,
            top_k=2,
            use_noisy_gating=True,
            noise_epsilon=0.1
        )

        x = torch.randn(8, 10, 128)

        # 训练模式下应该有噪声
        router.train()
        weights1, indices1 = router(x)

        # 评估模式下不应该有噪声（结果应该一致）
        router.eval()
        weights2, indices2 = router(x)
        weights3, indices3 = router(x)

        # 评估模式下，相同输入应该给出相同输出
        assert torch.equal(weights2, weights3)
        assert torch.equal(indices2, indices3)

    def test_routing_stats(self):
        """测试路由统计"""
        router = StandardRouter(
            input_dim=64,
            num_experts=4,
            top_k=2
        )

        # 初始状态
        stats = router.get_routing_stats()
        assert stats['total_calls'] == 0
        assert all(u == 0.0 for u in stats['expert_usage_ratio'])

        # 训练模式下前向传播
        router.train()
        x = torch.randn(10, 5, 64)
        weights, indices = router(x)

        # 获取统计
        stats = router.get_routing_stats()
        assert stats['total_calls'] == 10 * 2  # batch_size * top_k
        assert sum(stats['expert_usage']) == 10 * 2

        # 重置统计
        router.reset_stats()
        stats = router.get_routing_stats()
        assert stats['total_calls'] == 0

    def test_device_compatibility(self):
        """测试设备兼容性"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        router = StandardRouter(
            input_dim=128,
            num_experts=4,
            top_k=2
        )
        router = router.cuda()

        x = torch.randn(4, 10, 128).cuda()
        weights, indices = router(x)

        assert weights.device.type == 'cuda'
        assert indices.device.type == 'cuda'


class TestSpecificRouter:
    """测试SpecificRouter"""

    def test_init(self):
        """测试初始化"""
        router = SpecificRouter(num_experts=3)

        assert router.num_experts == 3

    def test_forward_with_category(self):
        """测试带曝光度类别的前向传播"""
        router = SpecificRouter(num_experts=3)

        x = torch.randn(8, 10, 128)
        exposure_category = torch.randint(0, 3, (8,))

        weights, indices = router(x, exposure_category=exposure_category)

        # 检查输出形状
        assert weights.shape == (8, 1)
        assert indices.shape == (8, 1)

        # 检查权重全为1
        assert torch.all(weights == 1.0)

        # 检查索引与category一致
        assert torch.equal(indices.squeeze(1), exposure_category)

    def test_forward_without_category_raises(self):
        """测试没有提供category时应该报错"""
        router = SpecificRouter(num_experts=3)

        x = torch.randn(4, 10, 128)

        with pytest.raises(ValueError, match="requires 'exposure_category'"):
            router(x)

    def test_invalid_category_raises(self):
        """测试无效的category应该报错"""
        router = SpecificRouter(num_experts=3)

        x = torch.randn(4, 10, 128)

        # category超出范围
        exposure_category = torch.tensor([0, 1, 3, 2])  # 3超出范围[0,2]

        with pytest.raises(ValueError, match="must be in"):
            router(x, exposure_category=exposure_category)

    def test_routing_stats(self):
        """测试路由统计"""
        router = SpecificRouter(num_experts=3)

        # 训练模式
        router.train()
        x = torch.randn(12, 5, 64)

        # 平均分配给3个专家
        exposure_category = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

        weights, indices = router(x, exposure_category=exposure_category)

        # 获取统计
        stats = router.get_routing_stats()
        assert stats['total_calls'] == 12
        assert stats['expert_usage'] == [4.0, 4.0, 4.0]
        assert all(abs(ratio - 1/3) < 0.01 for ratio in stats['expert_usage_ratio'])

    def test_device_compatibility(self):
        """测试设备兼容性"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        router = SpecificRouter(num_experts=3)
        router = router.cuda()

        x = torch.randn(4, 10, 128).cuda()
        exposure_category = torch.randint(0, 3, (4,)).cuda()

        weights, indices = router(x, exposure_category=exposure_category)

        assert weights.device.type == 'cuda'
        assert indices.device.type == 'cuda'


class TestSharedRouter:
    """测试SharedRouter"""

    def test_init(self):
        """测试初始化"""
        router = SharedRouter(
            input_dim=128,
            num_experts=4,
            top_k=2,
            hidden_dim=256
        )

        assert router.input_dim == 128
        assert router.num_experts == 4
        assert router.top_k == 2

    def test_forward(self):
        """测试前向传播"""
        router = SharedRouter(
            input_dim=128,
            num_experts=4,
            top_k=2,
            hidden_dim=256
        )

        x = torch.randn(8, 10, 128)

        weights, indices = router(x)

        # 检查输出形状
        assert weights.shape == (8, 2)
        assert indices.shape == (8, 2)

        # 检查权重归一化
        assert torch.allclose(weights.sum(dim=1), torch.ones(8), atol=1e-5)

        # 检查索引范围
        assert (indices >= 0).all()
        assert (indices < 4).all()

    def test_learnable_parameters(self):
        """测试可学习参数"""
        router = SharedRouter(
            input_dim=128,
            num_experts=4,
            top_k=2
        )

        # 确保有可训练参数
        params = list(router.parameters())
        assert len(params) > 0

        # 确保参数需要梯度
        assert all(p.requires_grad for p in params)

    def test_gradient_flow(self):
        """测试梯度流"""
        router = SharedRouter(
            input_dim=64,
            num_experts=4,
            top_k=2
        )

        x = torch.randn(4, 10, 64, requires_grad=True)

        weights, indices = router(x)

        # 计算一个简单的损失
        loss = weights.sum()
        loss.backward()

        # 检查梯度
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

        # 检查router参数的梯度
        for param in router.parameters():
            assert param.grad is not None

    def test_routing_stats(self):
        """测试路由统计"""
        router = SharedRouter(
            input_dim=64,
            num_experts=4,
            top_k=2
        )

        router.train()
        x = torch.randn(10, 5, 64)
        weights, indices = router(x)

        stats = router.get_routing_stats()
        assert stats['total_calls'] == 10 * 2
        assert 'load_balance_loss' in stats

    def test_device_compatibility(self):
        """测试设备兼容性"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        router = SharedRouter(
            input_dim=128,
            num_experts=4,
            top_k=2
        )
        router = router.cuda()

        x = torch.randn(4, 10, 128).cuda()
        weights, indices = router(x)

        assert weights.device.type == 'cuda'
        assert indices.device.type == 'cuda'


class TestRouterIntegration:
    """测试Router之间的集成"""

    def test_output_format_consistency(self):
        """测试所有Router输出格式一致"""
        x = torch.randn(4, 10, 128)
        exposure_category = torch.randint(0, 3, (4,))

        # StandardRouter
        standard_router = StandardRouter(128, 4, top_k=2)
        std_weights, std_indices = standard_router(x)

        # SpecificRouter
        specific_router = SpecificRouter(num_experts=3)
        spec_weights, spec_indices = specific_router(x, exposure_category=exposure_category)

        # SharedRouter
        shared_router = SharedRouter(128, 4, top_k=2)
        shared_weights, shared_indices = shared_router(x)

        # 所有输出都应该是 (batch_size, top_k)
        assert std_weights.ndim == 2
        assert spec_weights.ndim == 2
        assert shared_weights.ndim == 2

        assert std_indices.ndim == 2
        assert spec_indices.ndim == 2
        assert shared_indices.ndim == 2


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])
