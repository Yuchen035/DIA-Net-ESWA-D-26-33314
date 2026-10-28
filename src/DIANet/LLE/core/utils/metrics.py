"""
评估指标实现（PSNR, SSIM等）
"""

import torch
import numpy as np
from typing import Union


def calculate_psnr(
    img1: Union[torch.Tensor, np.ndarray],
    img2: Union[torch.Tensor, np.ndarray],
    max_value: float = 1.0
) -> float:
    """
    计算PSNR (Peak Signal-to-Noise Ratio)

    Args:
        img1: 第一张图像
        img2: 第二张图像
        max_value: 图像的最大值（默认1.0，如果是[0,255]则设为255.0）

    Returns:
        PSNR值（dB）
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(
    img1: Union[torch.Tensor, np.ndarray],
    img2: Union[torch.Tensor, np.ndarray],
    max_value: float = 1.0
) -> float:
    """
    计算SSIM (Structural Similarity Index)

    简化版本实现，仅用于基本评估

    Args:
        img1: 第一张图像
        img2: 第二张图像
        max_value: 图像的最大值

    Returns:
        SSIM值（0-1之间）
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    # 简化的SSIM实现
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim)
