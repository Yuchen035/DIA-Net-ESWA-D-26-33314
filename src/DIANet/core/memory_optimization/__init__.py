# -*- coding: utf-8 -*-
"""
显存优化模块 (Memory Optimization)
========================================

统一管理所有显存优化策略：
- 精度管理（PrecisionManager）
- 量化（Quantization）
- CPU卸载（Offload）
- Latent缓存（Cache）
- 梯度检查点（GradientCheckpoint）
- 8-bit优化器（Optimizer）
- 显存优化管理器（Manager）
"""

from .precision_manager import PrecisionManager, create_precision_manager
from .quantization import Quantizer, create_quantizer
from .offload import Offloader, create_offloader
from .cache import LatentCache, create_latent_cache
from .gradient_checkpoint import GradientCheckpoint, create_gradient_checkpoint
from .optimizer import create_optimizer, get_optimizer_info, OptimizerFactory
from .manager import MemoryOptimizationManager, create_memory_optimization_manager

__all__ = [
    # 精度管理
    "PrecisionManager",
    "create_precision_manager",
    # 量化
    "Quantizer",
    "create_quantizer",
    # Offload
    "Offloader",
    "create_offloader",
    # 缓存
    "LatentCache",
    "create_latent_cache",
    # 梯度检查点
    "GradientCheckpoint",
    "create_gradient_checkpoint",
    # 优化器
    "create_optimizer",
    "get_optimizer_info",
    "OptimizerFactory",
    # 管理器
    "MemoryOptimizationManager",
    "create_memory_optimization_manager",
]
