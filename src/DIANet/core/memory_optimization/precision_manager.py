# -*- coding: utf-8 -*-
"""
精度管理器 (PrecisionManager)
========================================

核心功能：
1. 统一管理所有精度转换
2. 语义化API - 每个转换操作都有明确目的
3. 可追踪 - 记录所有转换操作，便于调试

设计原则：
- 冻结权重：bf16 → int8 (量化) 或保持 bf16
- LoRA权重：fp32 (训练需要高精度梯度)
- 激活值：bf16 (节省计算和显存)
- 梯度：fp32 (避免小梯度丢失)
- 优化器状态：int8 (8-bit AdamW)
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Any
from collections import defaultdict
from loguru import logger


class PrecisionManager:
    """精度管理器 - 统一管理所有精度转换"""

    def __init__(self, config):
        """
        初始化精度管理器

        参数：
            config: 配置对象，包含以下字段：
                - training_dtype: 训练时的主数据类型 (通常为 bfloat16)
                - lora_dtype: LoRA权重数据类型 (通常为 float32)
                - log_conversions: 是否记录精度转换
        """
        self.config = config

        # 精度配置
        self.training_dtype = getattr(config, 'training_dtype', torch.bfloat16)
        self.lora_dtype = getattr(config, 'lora_dtype', torch.float32)
        self.log_conversions = getattr(config, 'log_conversions', False)

        # 转换统计（用于摘要）
        self.conversion_stats = defaultdict(lambda: {
            'from_dtype': None,
            'to_dtype': None,
            'purpose': '',
            'count': 0
        })

    def prepare_frozen_weight(
        self,
        weight: torch.Tensor,
        component_name: str
    ) -> torch.Tensor:
        """
        准备冻结权重：bf16 → int8 (如果启用量化) 或保持 bf16

        参数：
            weight: 权重张量
            component_name: 组件名称（用于日志）

        返回：
            转换后的权重
        """
        from_dtype = weight.dtype
        to_dtype = self.training_dtype

        # 如果启用量化，权重会在量化模块中处理，这里只转换到训练dtype
        if weight.dtype != to_dtype:
            weight = weight.to(to_dtype)

            purpose = "准备冻结权重 (后续可能量化)"
            self._log_conversion(component_name, from_dtype, to_dtype, purpose)

        return weight

    def prepare_lora_weight(
        self,
        weight: torch.Tensor,
        layer_name: str
    ) -> torch.Tensor:
        """
        准备LoRA权重：任意精度 → fp32（保证梯度精度）

        参数：
            weight: LoRA权重张量
            layer_name: 层名称（用于日志）

        返回：
            转换为fp32的权重
        """
        from_dtype = weight.dtype
        to_dtype = self.lora_dtype

        if weight.dtype != to_dtype:
            weight = weight.to(to_dtype)

            purpose = "LoRA训练需要fp32保证梯度精度"
            self._log_conversion(f"{layer_name}.lora_weight", from_dtype, to_dtype, purpose)

        return weight

    def prepare_activation(
        self,
        activation: torch.Tensor,
        stage: str = "forward"
    ) -> torch.Tensor:
        """
        准备激活值：任意精度 → bf16（节省计算和显存）

        参数：
            activation: 激活值张量
            stage: 阶段标识（用于日志）

        返回：
            转换为bf16的激活值
        """
        from_dtype = activation.dtype
        to_dtype = self.training_dtype

        if activation.dtype != to_dtype:
            activation = activation.to(to_dtype)

            purpose = "激活值使用bf16加速计算"
            self._log_conversion(f"activation.{stage}", from_dtype, to_dtype, purpose)

        return activation

    def lora_forward_cast(
        self,
        input_tensor: torch.Tensor,
        lora_weight: torch.Tensor,
        layer_name: str
    ) -> Tuple[torch.Tensor, torch.dtype]:
        """
        LoRA前向传播时的精度转换：input(bf16) → fp32

        参数：
            input_tensor: 输入张量 (bf16)
            lora_weight: LoRA权重 (fp32)
            layer_name: 层名称（用于日志）

        返回：
            (转换后的输入, 原始dtype)
        """
        original_dtype = input_tensor.dtype
        target_dtype = lora_weight.dtype  # 匹配LoRA权重精度

        if input_tensor.dtype != target_dtype:
            input_tensor = input_tensor.to(target_dtype)

            purpose = f"匹配LoRA权重精度({target_dtype})进行计算"
            self._log_conversion(f"{layer_name}.lora_input", original_dtype, target_dtype, purpose)

        return input_tensor, original_dtype

    def lora_output_cast(
        self,
        lora_output: torch.Tensor,
        layer_name: str
    ) -> torch.Tensor:
        """
        LoRA输出转换回激活值精度：lora_output(fp32) → bf16

        参数：
            lora_output: LoRA输出 (fp32)
            layer_name: 层名称（用于日志）

        返回：
            转换为bf16的输出
        """
        from_dtype = lora_output.dtype
        to_dtype = self.training_dtype

        if lora_output.dtype != to_dtype:
            lora_output = lora_output.to(to_dtype)

            purpose = "降回bf16与主干网络激活值匹配"
            self._log_conversion(f"{layer_name}.lora_output", from_dtype, to_dtype, purpose)

        return lora_output

    def prepare_gradient(
        self,
        gradient: torch.Tensor,
        param_name: str
    ) -> torch.Tensor:
        """
        准备梯度：任意精度 → fp32（避免小梯度丢失）

        参数：
            gradient: 梯度张量
            param_name: 参数名称（用于日志）

        返回：
            转换为fp32的梯度
        """
        from_dtype = gradient.dtype
        to_dtype = torch.float32

        if gradient.dtype != to_dtype:
            gradient = gradient.to(to_dtype)

            purpose = "梯度使用fp32避免小梯度丢失"
            self._log_conversion(f"{param_name}.gradient", from_dtype, to_dtype, purpose)

        return gradient

    def cast_model_to_dtype(
        self,
        model: nn.Module,
        dtype: torch.dtype,
        exclude_patterns: Optional[list] = None
    ) -> nn.Module:
        """
        将模型转换到指定精度（排除某些模块）

        参数：
            model: 模型
            dtype: 目标数据类型
            exclude_patterns: 排除的模块名称模式列表（fnmatch格式）

        返回：
            转换后的模型
        """
        import fnmatch

        exclude_patterns = exclude_patterns or []

        for name, param in model.named_parameters():
            # 检查是否在排除列表中
            should_exclude = any(
                fnmatch.fnmatch(name, pattern)
                for pattern in exclude_patterns
            )

            if not should_exclude and param.dtype != dtype:
                param.data = param.data.to(dtype)

        return model

    def _log_conversion(
        self,
        name: str,
        from_dtype: torch.dtype,
        to_dtype: torch.dtype,
        purpose: str
    ):
        """
        记录精度转换（内部方法）

        参数：
            name: 转换对象名称
            from_dtype: 源数据类型
            to_dtype: 目标数据类型
            purpose: 转换目的
        """
        # 更新统计
        self.conversion_stats[name]['from_dtype'] = from_dtype
        self.conversion_stats[name]['to_dtype'] = to_dtype
        self.conversion_stats[name]['purpose'] = purpose
        self.conversion_stats[name]['count'] += 1

        # 日志输出
        if self.log_conversions:
            logger.debug(
                f"[精度转换] {name}: {from_dtype} → {to_dtype} | 目的: {purpose}"
            )

    def print_conversion_summary(self):
        """打印精度转换摘要"""
        print("\n" + "=" * 80)
        print("精度转换摘要")
        print("=" * 80)

        if not self.conversion_stats:
            print("未记录到精度转换")
            return

        # 按名称排序
        sorted_stats = sorted(self.conversion_stats.items())

        for name, stats in sorted_stats:
            from_dtype = stats['from_dtype']
            to_dtype = stats['to_dtype']
            purpose = stats['purpose']
            count = stats['count']

            # 格式化输出
            print(f"{name:50} | {str(from_dtype):20} → {str(to_dtype):20} | {purpose}")
            if count > 1:
                print(f"{'':50} | 转换次数: {count}")

        print("=" * 80 + "\n")

    def get_conversion_stats(self) -> Dict[str, Any]:
        """
        获取转换统计数据

        返回：
            转换统计字典
        """
        return dict(self.conversion_stats)

    def reset_stats(self):
        """重置统计数据"""
        self.conversion_stats.clear()


def create_precision_manager(config) -> PrecisionManager:
    """
    工厂函数：创建精度管理器

    参数：
        config: 配置对象

    返回：
        PrecisionManager实例
    """
    return PrecisionManager(config)
