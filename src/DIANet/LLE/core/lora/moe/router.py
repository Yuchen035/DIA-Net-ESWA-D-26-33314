"""
MoE-LoRA路由器实现

包含三种路由器:
1. BaseRouter: 路由器抽象基类
2. StandardRouter: 标准MoE路由器（单路由模式）
3. SpecificRouter: Style-Specific路由器（双路由模式，硬编码）
4. SharedRouter: Style-Shared路由器（双路由模式，可学习）
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseRouter(nn.Module, ABC):
    """
    路由器抽象基类

    所有路由器必须实现forward方法，返回专家权重和索引
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        路由前向传播

        Args:
            x: 输入特征 [batch_size, seq_len, hidden_dim]
            **kwargs: 额外参数（如exposure_category用于双路由）

        Returns:
            weights: 专家权重 [batch_size, top_k]，已归一化
            indices: 选中的专家索引 [batch_size, top_k]
        """
        pass

    def get_routing_stats(self) -> Dict[str, Any]:
        """
        获取路由统计信息（用于监控和调试）

        Returns:
            包含路由统计信息的字典
        """
        return {}


class StandardRouter(BaseRouter):
    """
    标准MoE路由器（单路由模式）

    使用简单的线性门控网络 + Top-K选择

    Args:
        input_dim: 输入特征维度
        num_experts: 专家数量
        top_k: 选择Top-K个专家
        use_noisy_gating: 是否使用噪声门控（训练时增加探索性）
        noise_epsilon: 噪声强度
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        use_noisy_gating: bool = False,
        noise_epsilon: float = 1e-2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_noisy_gating = use_noisy_gating
        self.noise_epsilon = noise_epsilon

        # 门控网络（简单的Linear层）
        self.gate = nn.Linear(input_dim, num_experts)

        # 噪声参数（用于Noisy Top-K Gating）
        if use_noisy_gating:
            self.noise_linear = nn.Linear(input_dim, num_experts)

        # 统计信息（用于负载均衡监控）
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_calls', torch.tensor(0))

        # 缓存最近一次前向传播的路由信息（用于负载均衡损失计算）
        self.last_router_probs = None
        self.last_expert_indices = None

    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        标准路由前向传播

        Args:
            x: 输入特征 [batch_size, seq_len, hidden_dim]

        Returns:
            weights: 专家权重 [batch_size, top_k]
            indices: 选中的专家索引 [batch_size, top_k]
        """
        batch_size = x.size(0)

        # 1. 对序列维度进行mean pooling
        # [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
        x_pooled = x.mean(dim=1)

        # 2. 计算门控logits
        logits = self.gate(x_pooled)  # [batch_size, num_experts]

        # 3. 添加噪声（训练时）
        if self.use_noisy_gating and self.training:
            noise_logits = self.noise_linear(x_pooled)
            # 标准正态分布噪声
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            logits = logits + noise * self.noise_epsilon

        # 4. Softmax归一化
        probs = F.softmax(logits, dim=-1)  # [batch_size, num_experts]

        # 5. Top-K选择
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        # weights: [batch_size, top_k]
        # indices: [batch_size, top_k]

        # 6. 重新归一化权重（确保选中的专家权重和为1）
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)

        # 7. 更新统计信息（用于负载均衡）
        if self.training:
            self._update_stats(indices)
            # 缓存路由信息（用于负载均衡损失计算）
            self.last_router_probs = probs.detach()
            self.last_expert_indices = indices.detach()

        return weights, indices

    def _update_stats(self, indices: torch.Tensor):
        """
        更新专家使用统计

        Args:
            indices: 选中的专家索引 [batch_size, top_k]
        """
        with torch.no_grad():
            batch_size = indices.size(0)
            # 统计每个专家被选中的次数
            for expert_idx in range(self.num_experts):
                count = (indices == expert_idx).sum().float()
                self.expert_counts[expert_idx] += count

            self.total_calls += batch_size * self.top_k

    def get_routing_stats(self) -> Dict[str, Any]:
        """
        获取路由统计信息

        Returns:
            包含专家负载均衡信息的字典
        """
        if self.total_calls == 0:
            return {
                'expert_usage': self.expert_counts.cpu().tolist(),
                'expert_usage_ratio': [0.0] * self.num_experts,
                'load_balance_loss': 0.0
            }

        # 计算每个专家的使用比例
        usage_ratio = (self.expert_counts / self.total_calls).cpu().tolist()

        # 计算负载均衡损失（理想情况下每个专家使用1/num_experts）
        target_ratio = 1.0 / self.num_experts
        load_balance_loss = sum(
            abs(ratio - target_ratio) for ratio in usage_ratio
        ) / self.num_experts

        result = {
            'expert_usage': self.expert_counts.cpu().tolist(),
            'expert_usage_ratio': usage_ratio,
            'load_balance_loss': load_balance_loss,
            'total_calls': int(self.total_calls.item())
        }

        # 添加最近一次的路由信息（用于负载均衡损失计算）
        if self.last_router_probs is not None and self.last_expert_indices is not None:
            result['router_probs'] = self.last_router_probs
            result['expert_indices'] = self.last_expert_indices

        return result

    def reset_stats(self):
        """重置统计信息"""
        self.expert_counts.zero_()
        self.total_calls.zero_()
        self.last_router_probs = None
        self.last_expert_indices = None


class SpecificRouter(BaseRouter):
    """
    Style-Specific路由器（双路由模式）

    根据曝光度类别硬编码选择专家：
    - 0: 欠曝（过暗） -> Expert 0
    - 1: 正常曝光 -> Expert 1
    - 2: 过曝（过亮） -> Expert 2

    Args:
        num_experts: 专家数量（通常为3，对应3个曝光度类别）
    """

    def __init__(self, num_experts: int = 3):
        super().__init__()
        self.num_experts = num_experts

        # 统计信息
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_calls', torch.tensor(0))

    def forward(
        self,
        x: torch.Tensor,
        exposure_category: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Style-Specific路由（硬编码）

        Args:
            x: 输入特征 [batch_size, seq_len, hidden_dim]（未使用，仅保持接口一致）
            exposure_category: 曝光度类别 [batch_size]，取值0/1/2

        Returns:
            weights: 专家权重 [batch_size, 1]，全为1.0
            indices: 选中的专家索引 [batch_size, 1]
        """
        if exposure_category is None:
            raise ValueError(
                "SpecificRouter requires 'exposure_category' in forward() call"
            )

        batch_size = x.size(0)
        device = x.device

        # 权重全为1.0（只选择1个专家）
        weights = torch.ones(batch_size, 1, device=device, dtype=x.dtype)

        # 索引直接使用exposure_category
        indices = exposure_category.unsqueeze(1)  # [batch_size, 1]

        # 检查索引范围
        if (indices < 0).any() or (indices >= self.num_experts).any():
            raise ValueError(
                f"exposure_category must be in [0, {self.num_experts-1}], "
                f"got {indices.min().item()} to {indices.max().item()}"
            )

        # 更新统计信息
        if self.training:
            self._update_stats(indices)

        return weights, indices

    def _update_stats(self, indices: torch.Tensor):
        """更新专家使用统计"""
        with torch.no_grad():
            batch_size = indices.size(0)
            for expert_idx in range(self.num_experts):
                count = (indices == expert_idx).sum().float()
                self.expert_counts[expert_idx] += count

            self.total_calls += batch_size

    def get_routing_stats(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        if self.total_calls == 0:
            return {
                'expert_usage': self.expert_counts.cpu().tolist(),
                'expert_usage_ratio': [0.0] * self.num_experts,
                'total_calls': 0
            }

        usage_ratio = (self.expert_counts / self.total_calls).cpu().tolist()

        return {
            'expert_usage': self.expert_counts.cpu().tolist(),
            'expert_usage_ratio': usage_ratio,
            'total_calls': int(self.total_calls.item())
        }

    def reset_stats(self):
        """重置统计信息"""
        self.expert_counts.zero_()
        self.total_calls.zero_()


class SharedRouter(BaseRouter):
    """
    Style-Shared路由器（双路由模式）

    使用可学习的分类网络选择共享专家

    Args:
        input_dim: 输入特征维度
        num_experts: 专家数量
        top_k: 选择Top-K个专家
        hidden_dim: 分类器隐藏层维度
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        hidden_dim: int = 512
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # 分类器网络（比StandardRouter更复杂）
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )

        # 统计信息
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_calls', torch.tensor(0))

    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Style-Shared路由前向传播

        Args:
            x: 输入特征 [batch_size, seq_len, hidden_dim]

        Returns:
            weights: 专家权重 [batch_size, top_k]
            indices: 选中的专家索引 [batch_size, top_k]
        """
        batch_size = x.size(0)

        # 1. Mean pooling
        x_pooled = x.mean(dim=1)  # [batch_size, hidden_dim]

        # 2. 通过分类器
        logits = self.classifier(x_pooled)  # [batch_size, num_experts]

        # 3. Softmax + Top-K
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)

        # 4. 重新归一化
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)

        # 5. 更新统计信息
        if self.training:
            self._update_stats(indices)

        return weights, indices

    def _update_stats(self, indices: torch.Tensor):
        """更新专家使用统计"""
        with torch.no_grad():
            batch_size = indices.size(0)
            for expert_idx in range(self.num_experts):
                count = (indices == expert_idx).sum().float()
                self.expert_counts[expert_idx] += count

            self.total_calls += batch_size * self.top_k

    def get_routing_stats(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        if self.total_calls == 0:
            return {
                'expert_usage': self.expert_counts.cpu().tolist(),
                'expert_usage_ratio': [0.0] * self.num_experts,
                'load_balance_loss': 0.0,
                'total_calls': 0
            }

        usage_ratio = (self.expert_counts / self.total_calls).cpu().tolist()

        target_ratio = 1.0 / self.num_experts
        load_balance_loss = sum(
            abs(ratio - target_ratio) for ratio in usage_ratio
        ) / self.num_experts

        return {
            'expert_usage': self.expert_counts.cpu().tolist(),
            'expert_usage_ratio': usage_ratio,
            'load_balance_loss': load_balance_loss,
            'total_calls': int(self.total_calls.item())
        }

    def reset_stats(self):
        """重置统计信息"""
        self.expert_counts.zero_()
        self.total_calls.zero_()
