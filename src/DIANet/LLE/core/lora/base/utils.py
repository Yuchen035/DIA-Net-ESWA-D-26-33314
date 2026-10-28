"""
LoRA工具函数

包含初始化、合并等工具函数
"""

import torch
import torch.nn as nn
import math


def lora_init_weights(lora_A: nn.Parameter, lora_B: nn.Parameter, init_method: str = "kaiming"):
    """
    初始化LoRA权重

    Args:
        lora_A: LoRA的A矩阵 [in_features, rank]
        lora_B: LoRA的B矩阵 [rank, out_features]
        init_method: 初始化方法 ("kaiming", "normal", "zero")
    """
    if init_method == "kaiming":
        # Kaiming初始化A，B初始化为0
        nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
        nn.init.zeros_(lora_B)
    elif init_method == "normal":
        # 正态分布初始化A，B初始化为0
        nn.init.normal_(lora_A, std=0.02)
        nn.init.zeros_(lora_B)
    elif init_method == "zero":
        # 全部初始化为0
        nn.init.zeros_(lora_A)
        nn.init.zeros_(lora_B)
    else:
        raise ValueError(f"Unknown init method: {init_method}")


def merge_lora_weights(
    base_weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float = 1.0
) -> torch.Tensor:
    """
    合并LoRA权重到基础权重

    W_merged = W_base + scaling * (A @ B)^T

    Args:
        base_weight: 基础权重 [out_features, in_features]
        lora_A: LoRA的A矩阵 [in_features, rank]
        lora_B: LoRA的B矩阵 [rank, out_features]
        scaling: 缩放因子

    Returns:
        合并后的权重 [out_features, in_features]
    """
    # lora_A @ lora_B = [in_features, rank] @ [rank, out_features] = [in_features, out_features]
    # 转置得到 [out_features, in_features]
    lora_weight = (lora_A @ lora_B).t()  # [out_features, in_features]
    merged_weight = base_weight + scaling * lora_weight
    return merged_weight
