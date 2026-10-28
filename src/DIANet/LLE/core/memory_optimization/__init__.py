"""
显存优化模块（统一管理）

包含：
- precision_manager: 精度管理器（核心）
- quantization: 量化（int8/fp8）
- offload: VAE/Text Encoder offload
- cache: Latent缓存（内存/磁盘）
- gradient_checkpoint: 梯度检查点
- optimizer: 8-bit优化器封装
- manager: 显存优化管理器（统一调度）
"""

from .precision_manager import PrecisionManager
from .manager import MemoryOptimizationManager

__all__ = [
    'PrecisionManager',
    'MemoryOptimizationManager',
]
