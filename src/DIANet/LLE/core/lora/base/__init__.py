"""
LoRA基础模块

提供基础LoRA抽象类和工具函数
"""

from .base_lora import BaseLoRALayer
from .utils import lora_init_weights, merge_lora_weights

__all__ = [
    'BaseLoRALayer',
    'lora_init_weights',
    'merge_lora_weights',
]
