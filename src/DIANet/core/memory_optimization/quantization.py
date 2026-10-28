# -*- coding: utf-8 -*-
"""
量化模块 (Quantization)
========================================

功能：
1. INT8/FP8量化支持
2. 集成optimum-quanto库
3. 减少冻结权重显存占用约50%

支持的量化类型：
- int8: 8位整数量化 (~50%显存节省)
- fp8/float8: 8位浮点量化 (~50%显存节省)
"""
import torch
import torch.nn as nn
from typing import Optional, Literal
from loguru import logger


class Quantizer:
    """量化器 - 负责模型权重量化"""

    def __init__(self, config):
        """
        初始化量化器

        参数：
            config: 配置对象，包含以下字段：
                - enabled: 是否启用量化
                - qtype: 量化类型 ("int8", "fp8", "float8")
                - quantize_transformer: 是否量化transformer
                - quantize_text_encoder: 是否量化text encoder
        """
        self.config = config
        self.enabled = getattr(config, 'enabled', False)
        self.qtype = getattr(config, 'qtype', 'int8').lower()
        self.quantize_transformer = getattr(config, 'transformer', True)
        self.quantize_text_encoder = getattr(config, 'text_encoder', False)

        # 检查quanto库是否可用
        self.quanto_available = self._check_quanto_available()

    def _check_quanto_available(self) -> bool:
        """检查optimum-quanto是否可用"""
        try:
            import optimum.quanto
            return True
        except ImportError:
            if self.enabled:
                logger.warning("[量化] optimum-quanto未安装，量化功能不可用")
                logger.warning("[量化] 安装命令：pip install optimum-quanto")
            return False

    def quantize_model(
        self,
        model: nn.Module,
        model_type: Literal["transformer", "text_encoder", "vae"] = "transformer"
    ) -> nn.Module:
        """
        量化模型

        参数：
            model: 要量化的模型
            model_type: 模型类型（用于判断是否需要量化）

        返回：
            量化后的模型
        """
        # 检查是否需要量化
        if not self.enabled:
            return model

        if not self.quanto_available:
            logger.warning(f"[量化] optimum-quanto不可用，跳过{model_type}量化")
            return model

        # 根据模型类型判断是否量化
        should_quantize = False
        if model_type == "transformer" and self.quantize_transformer:
            should_quantize = True
        elif model_type == "text_encoder" and self.quantize_text_encoder:
            should_quantize = True

        if not should_quantize:
            return model

        try:
            from optimum.quanto import quantize, freeze, qint8, qfloat8

            # 获取量化类型
            qtype_map = {
                "int8": qint8,
                "fp8": qfloat8,
                "float8": qfloat8,  # 别名
            }
            qtype = qtype_map.get(self.qtype, qint8)

            logger.info(f"[量化] 正在量化 {model_type} 为 {self.qtype}...")

            # 量化模型（参考diffusers实现）
            # 1. 先移到CPU避免GPU峰值
            original_device = next(model.parameters()).device
            model = model.to("cpu")

            # 2. 量化权重
            quantize(model, weights=qtype)

            # 3. 冻结量化权重
            freeze(model)

            # 4. 移回原设备
            model = model.to(original_device)

            logger.info(f"[量化] ✓ {model_type} 量化完成")

        except Exception as e:
            logger.warning(f"[量化] 量化过程出错：{e}")
            logger.warning(f"[量化] 跳过量化，使用原始模型")

        return model

    def quantize_transformer(self, transformer: nn.Module) -> nn.Module:
        """
        量化Transformer模型

        参数：
            transformer: Transformer模型

        返回：
            量化后的模型
        """
        return self.quantize_model(transformer, model_type="transformer")

    def quantize_text_encoder(self, text_encoder: nn.Module) -> nn.Module:
        """
        量化Text Encoder模型

        参数：
            text_encoder: Text Encoder模型

        返回：
            量化后的模型
        """
        return self.quantize_model(text_encoder, model_type="text_encoder")

    def is_quantized(self, model: nn.Module) -> bool:
        """
        检查模型是否已量化

        参数：
            model: 模型

        返回：
            是否已量化
        """
        if not self.quanto_available:
            return False

        try:
            from optimum.quanto import QModuleMixin

            # 检查是否有量化模块
            for module in model.modules():
                if isinstance(module, QModuleMixin):
                    return True
            return False
        except ImportError:
            return False

    def get_quantization_info(self, model: nn.Module) -> dict:
        """
        获取量化信息

        参数：
            model: 模型

        返回：
            量化信息字典
        """
        info = {
            "quantized": self.is_quantized(model),
            "qtype": self.qtype if self.is_quantized(model) else None,
            "enabled": self.enabled,
        }

        # 计算量化模块数量
        if info["quantized"]:
            try:
                from optimum.quanto import QModuleMixin
                quantized_modules = sum(
                    1 for m in model.modules() if isinstance(m, QModuleMixin)
                )
                info["quantized_modules"] = quantized_modules
            except ImportError:
                pass

        return info


def create_quantizer(config) -> Quantizer:
    """
    工厂函数：创建量化器

    参数：
        config: 配置对象

    返回：
        Quantizer实例
    """
    return Quantizer(config)
