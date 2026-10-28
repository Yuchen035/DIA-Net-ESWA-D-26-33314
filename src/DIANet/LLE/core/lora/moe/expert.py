"""
LoRA Expert实现

提供独立的Expert类，用于MoE-LoRA系统中的专家模块。
每个专家是一个低秩分解：x' = x + (x @ A @ B) * scaling
"""

import torch
import torch.nn as nn
from typing import Optional


class LoRAExpert(nn.Module):
    """
    单个LoRA专家

    实现低秩分解：
    output = input + (input @ lora_A @ lora_B) * scaling

    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        rank: LoRA秩
        alpha: LoRA缩放因子（scaling = alpha / rank）
        dropout: Dropout概率（可选）
        init_method: 初始化方法 ("kaiming", "normal", "zero")
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
        init_method: str = "kaiming"
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = dropout
        self.init_method = init_method

        # LoRA矩阵
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Dropout（可选）
        if dropout > 0:
            self.dropout_layer = nn.Dropout(p=dropout)
        else:
            self.dropout_layer = None

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """初始化LoRA参数"""
        if self.init_method == "kaiming":
            # Kaiming初始化（He initialization）
            # 适用于ReLU激活函数，保持方差
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
        elif self.init_method == "normal":
            # 正态分布初始化
            nn.init.normal_(self.lora_A, std=0.02)
            nn.init.zeros_(self.lora_B)
        elif self.init_method == "zero":
            # 全零初始化（训练开始时不影响基础模型）
            nn.init.zeros_(self.lora_A)
            nn.init.zeros_(self.lora_B)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}. Expected 'kaiming', 'normal', or 'zero'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量
                - 2D: [batch_size, in_features]
                - 3D: [batch_size, seq_len, in_features]

        Returns:
            output: LoRA输出（不包含base输出）
                - 形状与输入相同
        """
        # LoRA计算: x @ A @ B * scaling
        # 注意：这里只返回LoRA部分，不包含原始输入
        # 因为base_layer的输出会在MoELoRALayer中单独计算
        lora_out = x @ self.lora_A @ self.lora_B

        # 应用scaling
        lora_out = lora_out * self.scaling

        # 应用Dropout（如果有，且在训练模式）
        if self.dropout_layer is not None and self.training:
            lora_out = self.dropout_layer(lora_out)

        return lora_out

    def extra_repr(self) -> str:
        """额外的字符串表示（用于print）"""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"alpha={self.alpha}, "
            f"scaling={self.scaling:.4f}, "
            f"dropout={self.dropout}, "
            f"init_method={self.init_method}"
        )


class ExpertEnsemble(nn.Module):
    """
    专家集合（容器）

    管理多个LoRA专家，提供统一的接口。
    用于MoE-LoRA层中的专家管理。

    Args:
        num_experts: 专家数量
        in_features: 输入特征维度
        out_features: 输出特征维度
        rank: LoRA秩
        alpha: LoRA缩放因子
        dropout: Dropout概率
        init_method: 初始化方法
    """

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
        init_method: str = "kaiming"
    ):
        super().__init__()

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # 创建专家列表
        self.experts = nn.ModuleList([
            LoRAExpert(
                in_features=in_features,
                out_features=out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                init_method=init_method
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """
        使用指定专家进行前向传播

        Args:
            x: 输入张量
            expert_idx: 专家索引

        Returns:
            output: 专家输出
        """
        if expert_idx < 0 or expert_idx >= self.num_experts:
            raise ValueError(f"Invalid expert_idx: {expert_idx}. Must be in [0, {self.num_experts})")

        return self.experts[expert_idx](x)

    def get_expert(self, expert_idx: int) -> LoRAExpert:
        """获取指定专家"""
        if expert_idx < 0 or expert_idx >= self.num_experts:
            raise ValueError(f"Invalid expert_idx: {expert_idx}. Must be in [0, {self.num_experts})")

        return self.experts[expert_idx]

    def extra_repr(self) -> str:
        """额外的字符串表示"""
        return (
            f"num_experts={self.num_experts}, "
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"alpha={self.alpha}"
        )


def create_expert_ensemble(
    num_experts: int,
    in_features: int,
    out_features: int,
    rank: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    init_method: str = "kaiming"
) -> ExpertEnsemble:
    """
    工厂函数：创建专家集合

    Args:
        num_experts: 专家数量
        in_features: 输入特征维度
        out_features: 输出特征维度
        rank: LoRA秩
        alpha: LoRA缩放因子
        dropout: Dropout概率
        init_method: 初始化方法

    Returns:
        expert_ensemble: ExpertEnsemble实例

    Examples:
        >>> experts = create_expert_ensemble(
        ...     num_experts=8,
        ...     in_features=768,
        ...     out_features=768,
        ...     rank=16,
        ...     alpha=16.0
        ... )
        >>> # 使用第3个专家
        >>> output = experts(input_tensor, expert_idx=3)
    """
    return ExpertEnsemble(
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        init_method=init_method
    )
