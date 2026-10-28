"""
基础LoRA层抽象类

定义所有LoRA层的通用接口
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseLoRALayer(nn.Module, ABC):
    """
    LoRA层抽象基类

    所有LoRA实现（标准LoRA、MoE-LoRA）都应继承此类
    """

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        """
        Args:
            base_layer: 被包装的基础层（冻结）
            rank: LoRA秩
            alpha: LoRA缩放因子
            dropout: Dropout概率
        """
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.scaling = alpha / rank if rank > 0 else 0.0

        # 冻结基础层
        for param in self.base_layer.parameters():
            param.requires_grad = False

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量
            **kwargs: 额外参数（如曝光类别等）

        Returns:
            输出张量
        """
        pass

    @abstractmethod
    def get_trainable_parameters(self):
        """
        获取可训练参数

        Returns:
            可训练参数的迭代器
        """
        pass
