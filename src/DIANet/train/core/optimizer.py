# -*- coding: utf-8 -*-
"""优化器模块 - 工厂模式"""
import torch

def create_optimizer(params, config):
    """创建优化器（工厂模式）"""
    opt_type = config.optimizer_type.lower()
    
    if opt_type == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.adam_weight_decay,
        )
    elif opt_type == "adamw8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                params,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
                weight_decay=config.adam_weight_decay,
            )
        except ImportError as exc:
            raise RuntimeError(
                "AdamW8bit was requested but bitsandbytes is unavailable. "
                "Resolve the runtime dependency, or explicitly select adamw in "
                "a new configuration; the optimizer is not substituted automatically."
            ) from exc
    else:
        raise ValueError(f"不支持的优化器：{opt_type}")
    
    return optimizer

def create_scheduler(optimizer, config, num_training_steps):
    """创建学习率调度器"""
    scheduler_type = config.lr_scheduler.lower()
    
    if scheduler_type == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps
        )
    elif scheduler_type == "constant_with_warmup":
        from transformers import get_constant_schedule_with_warmup
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.lr_warmup_steps
        )
    else:
        raise ValueError(f"不支持的调度器：{scheduler_type}")
    
    return scheduler
