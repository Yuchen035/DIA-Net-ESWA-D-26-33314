"""
负载均衡损失函数

用于MoE-LoRA的负载均衡，确保专家被均匀使用
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any


class LoadBalanceLoss(nn.Module):
    """
    MoE负载均衡损失

    目的：鼓励所有专家被均匀使用，避免某些专家被过度使用而其他专家被忽略

    公式：
    L_lb = num_experts * sum_i(f_i * P_i)

    其中：
    - f_i: 第i个专家被分配的token比例
    - P_i: 第i个专家的平均路由概率
    - num_experts: 专家数量

    当所有专家被均匀使用时，L_lb = 1（最优）

    Args:
        num_experts: 专家数量
        weight: 损失权重（默认0.01）
    """

    def __init__(self, num_experts: int, weight: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.weight = weight

    def forward(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        计算负载均衡损失

        Args:
            router_probs: 路由概率 [batch_size, num_experts]（softmax之后的）
            expert_indices: 选中的专家索引 [batch_size, top_k]

        Returns:
            负载均衡损失标量
        """
        batch_size = router_probs.size(0)

        # 1. 计算每个专家的平均路由概率 P_i
        # P_i = mean(router_probs[:, i])
        avg_probs = router_probs.mean(dim=0)  # [num_experts]

        # 2. 计算每个专家被选中的频率 f_i
        # f_i = (被选中的次数) / (总选择次数)
        expert_mask = torch.zeros(
            batch_size, self.num_experts,
            device=router_probs.device,
            dtype=router_probs.dtype
        )

        # 将选中的专家位置标记为1
        for i in range(batch_size):
            for j in range(expert_indices.size(1)):
                expert_idx = expert_indices[i, j]
                expert_mask[i, expert_idx] = 1.0

        # 计算频率
        frequencies = expert_mask.sum(dim=0) / (batch_size * expert_indices.size(1))  # [num_experts]

        # 3. 计算负载均衡损失
        # L_lb = num_experts * sum(f_i * P_i)
        load_balance_loss = self.num_experts * (frequencies * avg_probs).sum()

        return self.weight * load_balance_loss


class AuxiliaryLoadBalanceLoss(nn.Module):
    """
    辅助负载均衡损失（Google Switch Transformer风格）

    使用辅助损失来鼓励负载均衡，而不是直接约束路由

    Args:
        num_experts: 专家数量
        weight: 损失权重
    """

    def __init__(self, num_experts: int, weight: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.weight = weight

    def forward(
        self,
        router_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        计算辅助负载均衡损失

        Args:
            router_logits: 路由logits（softmax之前） [batch_size, num_experts]

        Returns:
            辅助负载均衡损失
        """
        # 计算每个专家的平均概率
        router_probs = torch.softmax(router_logits, dim=-1)
        avg_probs = router_probs.mean(dim=0)  # [num_experts]

        # 理想情况下，每个专家的平均概率应该是 1/num_experts
        target_prob = 1.0 / self.num_experts

        # 计算与理想分布的差异
        # 使用L2损失
        loss = ((avg_probs - target_prob) ** 2).sum()

        return self.weight * loss * self.num_experts


def compute_expert_utilization(
    expert_indices: torch.Tensor,
    num_experts: int
) -> Dict[str, Any]:
    """
    计算专家利用率统计

    Args:
        expert_indices: 选中的专家索引 [batch_size, top_k]
        num_experts: 专家总数

    Returns:
        包含统计信息的字典
    """
    batch_size = expert_indices.size(0)
    top_k = expert_indices.size(1)

    # 统计每个专家被选中的次数
    expert_counts = torch.zeros(num_experts, device=expert_indices.device)
    for i in range(num_experts):
        count = (expert_indices == i).sum().float()
        expert_counts[i] = count

    total_selections = batch_size * top_k

    # 计算使用率
    utilization = expert_counts / total_selections

    # 计算负载均衡系数（0-1，越接近1越均衡）
    # 使用熵来衡量
    eps = 1e-10
    probs = utilization + eps
    entropy = -(probs * torch.log(probs)).sum()
    max_entropy = torch.log(torch.tensor(num_experts, dtype=torch.float32))
    balance_score = entropy / max_entropy

    return {
        'expert_counts': expert_counts.cpu().tolist(),
        'utilization': utilization.cpu().tolist(),
        'balance_score': balance_score.item(),
        'std': utilization.std().item(),
        'min_utilization': utilization.min().item(),
        'max_utilization': utilization.max().item()
    }
