"""
负载均衡损失简单测试
"""

import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.loss.load_balance import (
    LoadBalanceLoss,
    AuxiliaryLoadBalanceLoss,
    compute_expert_utilization
)


def test_load_balance_loss():
    """测试LoadBalanceLoss"""
    print("\n=== 测试 LoadBalanceLoss ===")

    num_experts = 4
    loss_fn = LoadBalanceLoss(num_experts=num_experts, weight=0.01)

    # 创建路由概率和专家索引
    batch_size = 8
    router_probs = torch.softmax(torch.randn(batch_size, num_experts), dim=-1)
    expert_indices = torch.randint(0, num_experts, (batch_size, 2))  # top_k=2

    print(f"路由概率形状: {router_probs.shape}")
    print(f"专家索引形状: {expert_indices.shape}")

    # 计算损失
    loss = loss_fn(router_probs, expert_indices)

    print(f"负载均衡损失: {loss.item():.6f}")

    assert loss.ndim == 0, "损失应该是标量"
    assert loss.item() >= 0, "损失应该非负"

    # 测试完美均衡的情况
    # 每个专家被选中的频率相同
    perfect_indices = torch.tensor([
        [0, 1], [2, 3], [0, 1], [2, 3],
        [0, 1], [2, 3], [0, 1], [2, 3]
    ])
    perfect_probs = torch.ones(batch_size, num_experts) / num_experts

    perfect_loss = loss_fn(perfect_probs, perfect_indices)
    print(f"完美均衡损失: {perfect_loss.item():.6f}")

    # 完美均衡应该接近weight * 1.0
    assert abs(perfect_loss.item() - 0.01) < 0.001, f"完美均衡损失应接近0.01，实际: {perfect_loss.item()}"

    print("✅ LoadBalanceLoss 测试通过")


def test_auxiliary_load_balance_loss():
    """测试AuxiliaryLoadBalanceLoss"""
    print("\n=== 测试 AuxiliaryLoadBalanceLoss ===")

    num_experts = 4
    loss_fn = AuxiliaryLoadBalanceLoss(num_experts=num_experts, weight=0.01)

    # 创建路由logits
    batch_size = 16
    router_logits = torch.randn(batch_size, num_experts)

    print(f"路由logits形状: {router_logits.shape}")

    # 计算损失
    loss = loss_fn(router_logits)

    print(f"辅助负载均衡损失: {loss.item():.6f}")

    assert loss.ndim == 0, "损失应该是标量"
    assert loss.item() >= 0, "损失应该非负"

    # 测试均匀分布的情况
    uniform_logits = torch.zeros(batch_size, num_experts)  # softmax后每个专家概率都是1/4

    uniform_loss = loss_fn(uniform_logits)
    print(f"均匀分布损失: {uniform_loss.item():.6f}")

    # 均匀分布损失应该接近0
    assert uniform_loss.item() < 1e-6, f"均匀分布损失应接近0，实际: {uniform_loss.item()}"

    # 测试极不均匀的情况
    unbalanced_logits = torch.zeros(batch_size, num_experts)
    unbalanced_logits[:, 0] = 10.0  # 第一个专家概率接近1

    unbalanced_loss = loss_fn(unbalanced_logits)
    print(f"不均匀分布损失: {unbalanced_loss.item():.6f}")

    # 不均匀分布损失应该更大
    assert unbalanced_loss.item() > uniform_loss.item(), "不均匀分布损失应该大于均匀分布"

    print("✅ AuxiliaryLoadBalanceLoss 测试通过")


def test_compute_expert_utilization():
    """测试compute_expert_utilization"""
    print("\n=== 测试 compute_expert_utilization ===")

    num_experts = 4
    batch_size = 12

    # 创建专家索引：均匀分布
    expert_indices = torch.tensor([
        [0, 1], [0, 1], [0, 1],
        [2, 3], [2, 3], [2, 3],
        [0, 1], [0, 1], [0, 1],
        [2, 3], [2, 3], [2, 3]
    ])

    print(f"专家索引形状: {expert_indices.shape}")

    # 计算统计
    stats = compute_expert_utilization(expert_indices, num_experts)

    print(f"\n专家使用次数: {stats['expert_counts']}")
    print(f"专家使用率: {[f'{r:.3f}' for r in stats['utilization']]}")
    print(f"负载均衡系数: {stats['balance_score']:.4f}")
    print(f"标准差: {stats['std']:.4f}")
    print(f"最小/最大使用率: {stats['min_utilization']:.3f} / {stats['max_utilization']:.3f}")

    # 验证
    assert len(stats['expert_counts']) == num_experts
    assert len(stats['utilization']) == num_experts
    assert sum(stats['expert_counts']) == batch_size * 2  # top_k=2
    assert abs(sum(stats['utilization']) - 1.0) < 1e-6, "使用率总和应为1"

    # 完美均衡的balance_score应该接近1.0
    assert stats['balance_score'] > 0.99, f"均匀分布的均衡系数应接近1.0，实际: {stats['balance_score']}"

    # 测试不均匀分布
    unbalanced_indices = torch.zeros(12, 2, dtype=torch.long)  # 所有都选专家0
    unbalanced_stats = compute_expert_utilization(unbalanced_indices, num_experts)

    print(f"\n不均匀分布专家使用率: {[f'{r:.3f}' for r in unbalanced_stats['utilization']]}")
    print(f"不均匀分布均衡系数: {unbalanced_stats['balance_score']:.4f}")

    # 不均匀分布的balance_score应该更低
    assert unbalanced_stats['balance_score'] < stats['balance_score'], "不均匀分布的均衡系数应该更低"

    print("✅ compute_expert_utilization 测试通过")


def test_gradient_flow():
    """测试梯度流"""
    print("\n=== 测试梯度流 ===")

    num_experts = 4
    loss_fn = LoadBalanceLoss(num_experts=num_experts, weight=0.01)

    # 创建需要梯度的router_probs
    router_logits = torch.randn(8, num_experts, requires_grad=True)
    router_probs = torch.softmax(router_logits, dim=-1)

    expert_indices = torch.randint(0, num_experts, (8, 2))

    # 计算损失并反向传播
    loss = loss_fn(router_probs, expert_indices)
    loss.backward()

    # 检查梯度
    assert router_logits.grad is not None, "router_logits梯度为None"
    assert not torch.all(router_logits.grad == 0), "router_logits梯度全为0"

    print("✅ 梯度流测试通过")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("开始负载均衡损失测试")
    print("=" * 60)

    try:
        test_load_balance_loss()
        test_auxiliary_load_balance_loss()
        test_compute_expert_utilization()
        test_gradient_flow()

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
