# -*- coding: utf-8 -*-
"""
梯度检查点模块 (GradientCheckpoint)
========================================

功能：
1. 启用梯度检查点以节省显存
2. 释放中间激活值，仅保留必要的用于反向传播
3. 显存节省：40-60%（代价是增加约20-30%计算时间）

原理：
- 前向传播时不保存中间激活值
- 反向传播时重新计算中间激活值
- 用计算时间换显存空间

适用场景：
- 显存不足以训练大模型
- 可以接受略微降低训练速度
"""
import torch
import torch.nn as nn
from typing import Optional
from loguru import logger


class GradientCheckpoint:
    """梯度检查点管理器"""

    def __init__(self, config):
        """
        初始化梯度检查点管理器

        参数：
            config: 配置对象，包含以下字段：
                - enabled: 是否启用梯度检查点
        """
        self.config = config
        self.enabled = getattr(config, 'enabled', False)

    def apply(self, model: nn.Module, model_type: str = "transformer") -> nn.Module:
        """
        应用梯度检查点到模型

        参数：
            model: 模型
            model_type: 模型类型（用于日志）

        返回：
            应用梯度检查点后的模型
        """
        if not self.enabled:
            return model

        logger.info(f"[梯度检查点] 正在为{model_type}启用梯度检查点...")

        # 检查模型是否支持梯度检查点
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
            logger.info(f"[梯度检查点] ✓ {model_type}梯度检查点已启用")
        elif hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info(f"[梯度检查点] ✓ {model_type}梯度检查点已启用")
        else:
            logger.warning(f"[梯度检查点] {model_type}不支持梯度检查点")

        return model

    def apply_to_transformer(self, transformer: nn.Module) -> nn.Module:
        """
        应用梯度检查点到Transformer

        参数：
            transformer: Transformer模型

        返回：
            应用梯度检查点后的Transformer
        """
        return self.apply(transformer, model_type="Transformer")

    def disable(self, model: nn.Module) -> nn.Module:
        """
        禁用梯度检查点

        参数：
            model: 模型

        返回：
            禁用梯度检查点后的模型
        """
        if hasattr(model, 'disable_gradient_checkpointing'):
            model.disable_gradient_checkpointing()
            logger.info("[梯度检查点] 梯度检查点已禁用")
        elif hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
            logger.info("[梯度检查点] 梯度检查点已禁用")

        return model

    def is_enabled(self, model: nn.Module) -> bool:
        """
        检查模型是否启用了梯度检查点

        参数：
            model: 模型

        返回：
            是否启用梯度检查点
        """
        # 检查是否有gradient_checkpointing属性
        if hasattr(model, 'gradient_checkpointing'):
            return model.gradient_checkpointing
        elif hasattr(model.config, 'gradient_checkpointing') if hasattr(model, 'config') else False:
            return model.config.gradient_checkpointing

        return False


def create_gradient_checkpoint(config) -> GradientCheckpoint:
    """
    工厂函数：创建梯度检查点管理器

    参数：
        config: 配置对象

    返回：
        GradientCheckpoint实例
    """
    return GradientCheckpoint(config)
