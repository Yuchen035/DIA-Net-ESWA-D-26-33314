"""
Flow Matching Loss实现

MVP版本：
- 简单的MSE loss
- 不考虑时间步加权
- 支持velocity预测和noise预测两种模式

Flow Matching框架：
- 给定干净样本 x_1 和噪声 x_0
- 在时间步 t ∈ [0, 1] 上进行线性插值: x_t = (1-t)*x_0 + t*x_1
- 目标速度: v = x_1 - x_0
- 模型预测速度场: v_θ(x_t, t, condition)
- 损失: L = ||v_θ(x_t, t, c) - v||²
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FlowMatchingLoss(nn.Module):
    """
    Flow Matching Loss

    MVP版本：简单的MSE loss，不考虑时间步加权
    """

    def __init__(
        self,
        prediction_type: str = "velocity",
        reduction: str = "mean",
    ):
        """
        Args:
            prediction_type: 预测类型
                - "velocity": 预测速度场 v = x_1 - x_0 (默认)
                - "noise": 预测噪声 x_0
            reduction: 损失归约方式 ("mean", "sum", "none")
        """
        super().__init__()
        self.prediction_type = prediction_type
        self.reduction = reduction

        if prediction_type not in ["velocity", "noise"]:
            raise ValueError(
                f"prediction_type must be 'velocity' or 'noise', got {prediction_type}"
            )

    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算Flow Matching损失

        Args:
            model_output: 模型预测 [B, C, H, W]
                - 如果prediction_type="velocity"，这是预测的速度场
                - 如果prediction_type="noise"，这是预测的噪声
            target: 目标latent（干净样本 x_1）[B, C, H, W]
            noise: 噪声样本（x_0）[B, C, H, W]
                - 仅当prediction_type="velocity"时需要
                - 用于计算目标速度 v = target - noise

        Returns:
            loss: 标量损失值
        """
        if self.prediction_type == "velocity":
            # 速度预测模式
            if noise is None:
                raise ValueError("noise is required when prediction_type='velocity'")

            # 目标速度: v = x_1 - x_0
            target_velocity = target - noise

            # MSE loss
            loss = F.mse_loss(model_output, target_velocity, reduction=self.reduction)

        elif self.prediction_type == "noise":
            # 噪声预测模式
            if noise is None:
                raise ValueError("noise is required when prediction_type='noise'")

            # 目标就是噪声
            loss = F.mse_loss(model_output, noise, reduction=self.reduction)

        return loss

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"prediction_type={self.prediction_type}, "
            f"reduction={self.reduction})"
        )


def compute_flow_matching_loss(
    model_output: torch.Tensor,
    target: torch.Tensor,
    noise: torch.Tensor,
    prediction_type: str = "velocity",
    reduction: str = "mean",
) -> torch.Tensor:
    """
    计算Flow Matching损失的便捷函数

    Args:
        model_output: 模型预测 [B, C, H, W]
        target: 目标latent（干净样本 x_1）[B, C, H, W]
        noise: 噪声样本（x_0）[B, C, H, W]
        prediction_type: 预测类型 ("velocity" or "noise")
        reduction: 损失归约方式 ("mean", "sum", "none")

    Returns:
        loss: 标量损失值

    Example:
        >>> model_output = torch.randn(2, 4, 32, 32)
        >>> target = torch.randn(2, 4, 32, 32)
        >>> noise = torch.randn(2, 4, 32, 32)
        >>> loss = compute_flow_matching_loss(model_output, target, noise)
    """
    loss_fn = FlowMatchingLoss(prediction_type=prediction_type, reduction=reduction)
    return loss_fn(model_output, target, noise)
