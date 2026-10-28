# -*- coding: utf-8 -*-
"""
显存优化管理器 (MemoryOptimizationManager)
========================================

核心功能：
1. 统一管理所有显存优化策略
2. 提供简单的API：prepare_model(), create_optimizer()
3. 整合：精度管理、量化、Offload、缓存、梯度检查点、8-bit优化器

设计原则：
- 单一入口：所有优化策略通过Manager统一管理
- 可插拔：每个优化策略可以独立启用/禁用
- 易用性：简化API，隐藏实现细节
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Iterable
from loguru import logger

from .precision_manager import PrecisionManager, create_precision_manager
from .quantization import Quantizer, create_quantizer
from .offload import Offloader, create_offloader
from .cache import LatentCache, create_latent_cache
from .gradient_checkpoint import GradientCheckpoint, create_gradient_checkpoint
from .optimizer import create_optimizer as create_optimizer_func


class MemoryOptimizationManager:
    """显存优化管理器 - 统一管理所有优化策略"""

    def __init__(self, config):
        """
        初始化显存优化管理器

        参数：
            config: 配置对象，包含以下字段：
                - precision: 精度管理配置
                - quantization: 量化配置
                - offload: Offload配置
                - cache: 缓存配置
                - gradient_checkpoint: 梯度检查点配置
                - optimizer: 优化器配置
        """
        self.config = config

        # 创建各个优化组件
        self.precision_mgr = create_precision_manager(
            getattr(config, 'precision', config)
        )

        self.quantizer = create_quantizer(
            getattr(config, 'quantization', config)
        )

        self.offloader = create_offloader(
            getattr(config, 'offload', config)
        )

        self.gradient_checkpoint = create_gradient_checkpoint(
            getattr(config, 'gradient_checkpoint', config)
        )

        # 缓存需要VAE，稍后设置
        self.cache: Optional[LatentCache] = None

        logger.info("[显存优化] 显存优化管理器初始化完成")

    def prepare_models(
        self,
        transformer: nn.Module,
        vae: nn.Module,
        text_encoder_1: nn.Module,
        text_encoder_2: Optional[nn.Module] = None
    ) -> Tuple[nn.Module, nn.Module, nn.Module, Optional[nn.Module]]:
        """
        准备模型：应用所有显存优化策略

        参数：
            transformer: Transformer模型
            vae: VAE模型
            text_encoder_1: Text Encoder 1
            text_encoder_2: Text Encoder 2（可选）

        返回：
            (优化后的transformer, vae, text_encoder_1, text_encoder_2)
        """
        logger.info("[显存优化] 开始应用显存优化策略...")

        # 1. 量化
        if self.quantizer.enabled:
            transformer = self.quantizer.quantize_transformer(transformer)
            if self.quantizer.quantize_text_encoder:
                text_encoder_1 = self.quantizer.quantize_text_encoder(text_encoder_1)
                if text_encoder_2 is not None:
                    text_encoder_2 = self.quantizer.quantize_text_encoder(text_encoder_2)

        # 2. 梯度检查点
        if self.gradient_checkpoint.enabled:
            transformer = self.gradient_checkpoint.apply_to_transformer(transformer)

        # 3. Offload
        if self.offloader.enabled:
            if self.offloader.should_offload_vae:
                vae = self.offloader.offload_vae(vae)

            if self.offloader.should_offload_text_encoder:
                text_encoder_1 = self.offloader.offload_text_encoder(
                    text_encoder_1, name="text_encoder_1"
                )
                if text_encoder_2 is not None:
                    text_encoder_2 = self.offloader.offload_text_encoder(
                        text_encoder_2, name="text_encoder_2"
                    )

        logger.info("[显存优化] ✓ 显存优化策略应用完成")

        return transformer, vae, text_encoder_1, text_encoder_2

    def create_cache(
        self,
        vae: nn.Module,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ) -> Optional[LatentCache]:
        """
        创建Latent缓存

        参数：
            vae: VAE模型
            device: 设备
            dtype: 数据类型

        返回：
            LatentCache实例，如果未启用返回None
        """
        cache_config = getattr(self.config, 'cache', None)

        if cache_config is None or not getattr(cache_config, 'enabled', False):
            return None

        self.cache = create_latent_cache(cache_config, vae, device, dtype)
        logger.info("[显存优化] Latent缓存已创建")

        return self.cache

    def create_optimizer(
        self,
        params: Iterable[nn.Parameter],
        use_8bit: Optional[bool] = None
    ) -> torch.optim.Optimizer:
        """
        创建优化器（标准或8-bit）

        参数：
            params: 模型参数
            use_8bit: 是否使用8-bit优化器（可选，覆盖配置）

        返回：
            优化器实例
        """
        optimizer_config = getattr(self.config, 'optimizer', self.config)

        return create_optimizer_func(params, optimizer_config, use_8bit)

    def get_optimization_info(self) -> Dict[str, Any]:
        """
        获取优化信息摘要

        返回：
            优化信息字典
        """
        info = {
            "precision": {
                "training_dtype": str(self.precision_mgr.training_dtype),
                "lora_dtype": str(self.precision_mgr.lora_dtype),
                "log_conversions": self.precision_mgr.log_conversions,
            },
            "quantization": self.quantizer.get_quantization_info(
                nn.Module()  # Dummy model for structure
            ) if self.quantizer.enabled else {"enabled": False},
            "offload": self.offloader.get_offload_info(),
            "gradient_checkpoint": {
                "enabled": self.gradient_checkpoint.enabled,
            },
            "cache": self.cache.get_cache_info() if self.cache else {"enabled": False},
        }

        return info

    def print_optimization_summary(self):
        """打印优化摘要"""
        info = self.get_optimization_info()

        print("\n" + "=" * 80)
        print("显存优化摘要")
        print("=" * 80)

        # 精度管理
        print(f"\n[精度管理]")
        print(f"  训练dtype: {info['precision']['training_dtype']}")
        print(f"  LoRA dtype: {info['precision']['lora_dtype']}")
        print(f"  记录转换: {info['precision']['log_conversions']}")

        # 量化
        print(f"\n[量化]")
        if info['quantization']['enabled']:
            print(f"  启用: ✓")
            print(f"  量化类型: {info['quantization'].get('qtype', 'N/A')}")
        else:
            print(f"  启用: ✗")

        # Offload
        print(f"\n[Offload]")
        if info['offload']['enabled']:
            print(f"  启用: ✓")
            print(f"  VAE: {'✓' if info['offload']['offload_vae'] else '✗'}")
            print(f"  Text Encoder: {'✓' if info['offload']['offload_text_encoder'] else '✗'}")
            print(f"  已offload: {', '.join(info['offload']['offloaded_models'])}")
            if 'estimated_memory_saved_gb' in info['offload']:
                print(f"  预计节省显存: {info['offload']['estimated_memory_saved_gb']:.2f} GB")
        else:
            print(f"  启用: ✗")

        # 梯度检查点
        print(f"\n[梯度检查点]")
        print(f"  启用: {'✓' if info['gradient_checkpoint']['enabled'] else '✗'}")

        # 缓存
        print(f"\n[Latent缓存]")
        if info['cache']['enabled']:
            print(f"  启用: ✓")
            print(f"  模式: {info['cache'].get('mode', 'N/A')}")
            if 'cached_count' in info['cache']:
                print(f"  缓存数量: {info['cache']['cached_count']}")
            if 'memory_usage_mb' in info['cache']:
                print(f"  内存占用: {info['cache']['memory_usage_mb']:.2f} MB")
            elif 'disk_usage_mb' in info['cache']:
                print(f"  磁盘占用: {info['cache']['disk_usage_mb']:.2f} MB")
        else:
            print(f"  启用: ✗")

        print("=" * 80 + "\n")

    def cleanup(self):
        """清理资源"""
        if self.cache:
            self.cache.clear_cache()

        self.offloader.clear()

        logger.info("[显存优化] 资源清理完成")


def create_memory_optimization_manager(config) -> MemoryOptimizationManager:
    """
    工厂函数：创建显存优化管理器

    参数：
        config: 配置对象

    返回：
        MemoryOptimizationManager实例
    """
    return MemoryOptimizationManager(config)
