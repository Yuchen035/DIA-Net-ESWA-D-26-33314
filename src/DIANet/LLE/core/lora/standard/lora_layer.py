"""
标准LoRA层实现

实现标准的Low-Rank Adaptation (LoRA)层
支持Linear层的LoRA包装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..base.base_lora import BaseLoRALayer
from ..base.utils import lora_init_weights


class StandardLoRALayer(BaseLoRALayer):
    """
    标准LoRA层实现

    将基础层（通常是Linear）包装为LoRA形式：
    h = W_0 x + scaling * (B A) x

    其中：
    - W_0: 原始冻结权重 [out_features, in_features]
    - A: LoRA下投影矩阵 [in_features, rank]
    - B: LoRA上投影矩阵 [rank, out_features]
    - scaling = alpha / rank

    Examples:
        # 创建LoRA层
        base_layer = nn.Linear(1024, 2048)
        lora_layer = StandardLoRALayer(
            base_layer=base_layer,
            rank=16,
            alpha=16.0,
        )

        # 前向传播
        x = torch.randn(4, 1024)
        output = lora_layer(x)  # [4, 2048]
    """

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        init_method: str = "kaiming",
    ):
        """
        Args:
            base_layer: 被包装的基础层（通常是nn.Linear）
            rank: LoRA秩
            alpha: LoRA缩放因子
            dropout: Dropout概率
            init_method: 初始化方法 ("kaiming", "normal", "zero")

        Raises:
            ValueError: base_layer不是Linear层
            ValueError: rank或alpha非正数
        """
        super().__init__(base_layer, rank, alpha, dropout)

        # 检查base_layer类型
        if not isinstance(base_layer, nn.Linear):
            raise ValueError(
                f"StandardLoRALayer only supports nn.Linear, got {type(base_layer)}"
            )

        # 验证参数
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        # 获取基础层的输入输出维度
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # LoRA权重矩阵
        # A: [in_features, rank] - 下投影
        # B: [rank, out_features] - 上投影
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))

        # 初始化LoRA权重
        lora_init_weights(self.lora_A, self.lora_B, init_method)

        # 移动LoRA参数到与base_layer相同的device和dtype
        if base_layer.weight is not None:
            device = base_layer.weight.device
            dtype = base_layer.weight.dtype
            self.lora_A.data = self.lora_A.data.to(device=device, dtype=dtype)
            self.lora_B.data = self.lora_B.data.to(device=device, dtype=dtype)

        # Dropout层（可选）
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # 标记LoRA层已初始化
        self.is_initialized = True

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播

        h = W_0 x + scaling * dropout(x A B)

        Args:
            x: 输入张量 [batch_size, ..., in_features]
            **kwargs: 额外参数（预留，当前未使用）

        Returns:
            输出张量 [batch_size, ..., out_features]
        """
        # 1. 基础层输出（冻结权重）
        base_output = self.base_layer(x)

        # 2. LoRA路径
        # x @ A -> [batch, ..., rank]
        lora_output = self.lora_dropout(x) @ self.lora_A
        # lora_output @ B -> [batch, ..., out_features]
        lora_output = lora_output @ self.lora_B

        # 3. 缩放并合并
        # h = W_0 x + (alpha / rank) * (B A) x
        return base_output + self.scaling * lora_output

    def get_trainable_parameters(self):
        """
        获取可训练参数（仅LoRA权重）

        Returns:
            可训练参数的迭代器
        """
        yield self.lora_A
        yield self.lora_B

    def merge_weights(self) -> nn.Linear:
        """
        合并LoRA权重到基础层

        创建一个新的Linear层，权重为 W_merged = W_0 + scaling * (B @ A)^T

        Returns:
            合并后的Linear层
        """
        from ..base.utils import merge_lora_weights

        # 获取基础权重
        with torch.no_grad():
            base_weight = self.base_layer.weight.data.clone()  # [out, in]
            base_bias = (
                self.base_layer.bias.data.clone()
                if self.base_layer.bias is not None
                else None
            )

            # 合并LoRA权重
            # W_merged = W_0 + scaling * (B @ A)^T
            merged_weight = merge_lora_weights(
                base_weight, self.lora_A.data, self.lora_B.data, self.scaling
            )

            # 创建新的Linear层
            merged_layer = nn.Linear(
                self.in_features, self.out_features, bias=(base_bias is not None)
            )
            merged_layer.weight.data = merged_weight
            if base_bias is not None:
                merged_layer.bias.data = base_bias

            return merged_layer

    def reset_lora_parameters(self, init_method: str = "kaiming"):
        """
        重新初始化LoRA参数

        Args:
            init_method: 初始化方法
        """
        with torch.no_grad():
            lora_init_weights(self.lora_A, self.lora_B, init_method)

    def __repr__(self) -> str:
        return (
            f"StandardLoRALayer(\n"
            f"  in_features={self.in_features},\n"
            f"  out_features={self.out_features},\n"
            f"  rank={self.rank},\n"
            f"  alpha={self.alpha},\n"
            f"  scaling={self.scaling:.4f},\n"
            f"  dropout={self.dropout}\n"
            f")"
        )


def create_lora_layer(
    base_layer: nn.Linear,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    init_method: str = "kaiming",
) -> StandardLoRALayer:
    """
    工厂函数：创建LoRA层

    Args:
        base_layer: 基础Linear层
        rank: LoRA秩
        alpha: 缩放因子
        dropout: Dropout概率
        init_method: 初始化方法

    Returns:
        StandardLoRALayer实例

    Examples:
        base_layer = nn.Linear(512, 1024)
        lora_layer = create_lora_layer(base_layer, rank=8, alpha=8.0)
    """
    return StandardLoRALayer(
        base_layer=base_layer,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        init_method=init_method,
    )


def replace_linear_with_lora(
    module: nn.Module,
    target_modules: list,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> int:
    """
    递归替换模块中的Linear层为LoRA层

    Args:
        module: 要处理的模块
        target_modules: 目标模块名称列表（支持通配符）
        rank: LoRA秩
        alpha: 缩放因子
        dropout: Dropout概率

    Returns:
        替换的层数

    Examples:
        model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
        )
        num_replaced = replace_linear_with_lora(model, ["*"], rank=16)
        print(f"Replaced {num_replaced} layers")
    """
    import fnmatch

    num_replaced = 0

    for name, child in module.named_children():
        # 检查是否匹配目标模块
        matched = any(fnmatch.fnmatch(name, pattern) for pattern in target_modules)

        if matched and isinstance(child, nn.Linear):
            # 创建LoRA层
            lora_layer = create_lora_layer(
                base_layer=child, rank=rank, alpha=alpha, dropout=dropout
            )
            # 替换
            setattr(module, name, lora_layer)
            num_replaced += 1
        else:
            # 递归处理子模块
            num_replaced += replace_linear_with_lora(
                child, target_modules, rank, alpha, dropout
            )

    return num_replaced
