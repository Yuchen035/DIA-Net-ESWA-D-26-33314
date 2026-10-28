# -*- coding: utf-8 -*-
"""
8-bit优化器模块 (8-bit Optimizer)
========================================

功能：
1. 集成bitsandbytes的8-bit AdamW优化器
2. 显存节省：75%优化器状态显存
3. 性能几乎无损失

原理：
- 优化器状态（momentum, variance）使用8-bit存储
- 计算时动态反量化到fp32
- 显著减少优化器显存占用

支持的优化器：
- AdamW8bit: 8-bit AdamW
- Adam8bit: 8-bit Adam
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Iterable
from loguru import logger


class OptimizerFactory:
    """优化器工厂 - 创建标准或8-bit优化器"""

    @staticmethod
    def create_optimizer(
        params: Iterable[nn.Parameter],
        optimizer_type: str = "adamw",
        learning_rate: float = 1e-4,
        use_8bit: bool = False,
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        创建优化器

        参数：
            params: 模型参数
            optimizer_type: 优化器类型 ("adamw", "adam", "sgd")
            learning_rate: 学习率
            use_8bit: 是否使用8-bit优化器
            **kwargs: 其他优化器参数
                - betas: (beta1, beta2) for Adam/AdamW
                - eps: epsilon
                - weight_decay: 权重衰减

        返回：
            优化器实例
        """
        optimizer_type = optimizer_type.lower()

        # 提取参数
        betas = kwargs.get('betas', (0.9, 0.999))
        eps = kwargs.get('eps', 1e-8)
        weight_decay = kwargs.get('weight_decay', 0.01)

        # 创建优化器
        if use_8bit:
            return OptimizerFactory._create_8bit_optimizer(
                params, optimizer_type, learning_rate, betas, eps, weight_decay
            )
        else:
            return OptimizerFactory._create_standard_optimizer(
                params, optimizer_type, learning_rate, betas, eps, weight_decay
            )

    @staticmethod
    def _create_8bit_optimizer(
        params,
        optimizer_type: str,
        learning_rate: float,
        betas: tuple,
        eps: float,
        weight_decay: float
    ) -> torch.optim.Optimizer:
        """
        创建8-bit优化器

        参数：
            params: 模型参数
            optimizer_type: 优化器类型
            learning_rate: 学习率
            betas: Adam beta参数
            eps: epsilon
            weight_decay: 权重衰减

        返回：
            8-bit优化器实例
        """
        try:
            import bitsandbytes as bnb

            logger.info(f"[优化器] 创建8-bit {optimizer_type.upper()}优化器")

            if optimizer_type == "adamw":
                optimizer = bnb.optim.AdamW8bit(
                    params,
                    lr=learning_rate,
                    betas=betas,
                    eps=eps,
                    weight_decay=weight_decay,
                )
            elif optimizer_type == "adam":
                optimizer = bnb.optim.Adam8bit(
                    params,
                    lr=learning_rate,
                    betas=betas,
                    eps=eps,
                    weight_decay=weight_decay,
                )
            else:
                logger.warning(f"[优化器] 8-bit {optimizer_type}不支持，使用标准优化器")
                return OptimizerFactory._create_standard_optimizer(
                    params, optimizer_type, learning_rate, betas, eps, weight_decay
                )

            logger.info(f"[优化器] ✓ 8-bit {optimizer_type.upper()}优化器创建成功")
            return optimizer

        except ImportError:
            logger.warning("[优化器] bitsandbytes未安装，使用标准优化器")
            logger.warning("[优化器] 安装命令：pip install bitsandbytes")
            return OptimizerFactory._create_standard_optimizer(
                params, optimizer_type, learning_rate, betas, eps, weight_decay
            )

    @staticmethod
    def _create_standard_optimizer(
        params,
        optimizer_type: str,
        learning_rate: float,
        betas: tuple,
        eps: float,
        weight_decay: float
    ) -> torch.optim.Optimizer:
        """
        创建标准优化器

        参数：
            params: 模型参数
            optimizer_type: 优化器类型
            learning_rate: 学习率
            betas: Adam beta参数
            eps: epsilon
            weight_decay: 权重衰减

        返回：
            标准优化器实例
        """
        logger.info(f"[优化器] 创建标准{optimizer_type.upper()}优化器")

        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"不支持的优化器类型：{optimizer_type}")

        return optimizer


def create_optimizer(
    params: Iterable[nn.Parameter],
    config,
    use_8bit: Optional[bool] = None
) -> torch.optim.Optimizer:
    """
    工厂函数：从配置创建优化器

    参数：
        params: 模型参数
        config: 配置对象，包含以下字段：
            - optimizer_type: 优化器类型
            - learning_rate: 学习率
            - use_8bit: 是否使用8-bit优化器
            - adam_beta1, adam_beta2: Adam参数
            - adam_epsilon: epsilon
            - adam_weight_decay: 权重衰减
        use_8bit: 覆盖配置中的use_8bit（可选）

    返回：
        优化器实例
    """
    optimizer_type = getattr(config, 'optimizer_type', 'adamw')
    learning_rate = getattr(config, 'learning_rate', 1e-4)

    # 确定是否使用8-bit
    if use_8bit is None:
        use_8bit = getattr(config, 'use_8bit', False)

    # 提取Adam参数
    betas = (
        getattr(config, 'adam_beta1', 0.9),
        getattr(config, 'adam_beta2', 0.999)
    )
    eps = getattr(config, 'adam_epsilon', 1e-8)
    weight_decay = getattr(config, 'adam_weight_decay', 0.01)

    return OptimizerFactory.create_optimizer(
        params=params,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        use_8bit=use_8bit,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )


def get_optimizer_info(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """
    获取优化器信息

    参数：
        optimizer: 优化器

    返回：
        优化器信息字典
    """
    info = {
        "type": optimizer.__class__.__name__,
        "param_groups": len(optimizer.param_groups),
    }

    # 获取学习率
    if optimizer.param_groups:
        info["learning_rate"] = optimizer.param_groups[0]['lr']

    # 检查是否为8-bit优化器
    try:
        import bitsandbytes as bnb
        info["is_8bit"] = isinstance(optimizer, (bnb.optim.AdamW8bit, bnb.optim.Adam8bit))
    except ImportError:
        info["is_8bit"] = False

    return info
