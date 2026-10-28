# -*- coding: utf-8 -*-
"""
显存优化模块 - 可插拔的优化策略
"""
import torch
from typing import Optional

class MemoryOptimizer:
    """显存优化器 - 统一管理所有优化策略"""
    
    def __init__(self, config):
        """
        初始化显存优化器
        
        参数：
            config: 训练配置对象
        """
        self.config = config
        self.quantizer = None
        
    def apply_quantization(self, model, model_type: str = "transformer"):
        """
        应用量化策略
        
        参数：
            model: 要量化的模型
            model_type: 模型类型（transformer 或 text_encoder）
        
        返回：
            量化后的模型
        """
        # 判断是否需要量化
        should_quantize = (
            (model_type == "transformer" and self.config.quantize) or
            (model_type == "text_encoder" and self.config.quantize_te)
        )
        
        if not should_quantize:
            return model
        
        try:
            # 导入 optimum.quanto（参考 diffusers 库的实现）
            from optimum.quanto import quantize, freeze, qint8, qfloat8

            # 获取量化类型（直接使用 quanto 的类型常量）
            qtype_map = {
                "int8": qint8,
                "fp8": qfloat8,
                "float8": qfloat8,  # 兼容别名
            }
            qtype = qtype_map.get(self.config.qtype.lower(), qint8)

            print(f"[显存优化] 正在量化 {model_type} 为 {self.config.qtype}...")

            # 量化模型（参考 diffusers 的正确用法），先移到 CPU 避免 GPU 峰值
            model = model.to("cpu")
            quantize(model, weights=qtype)
            freeze(model)

            print(f"[显存优化] ✓ {model_type} 量化完成")
            return model

        except ImportError as e:
            print(f"[警告] optimum.quanto 未安装，跳过量化")
            print(f"[警告] 安装命令：pip install optimum-quanto")
            return model
        except Exception as e:
            print(f"[警告] 量化过程出错：{e}")
            print(f"[警告] 跳过量化，使用原始模型")
            return model
    
    def apply_gradient_checkpointing(self, model):
        """
        应用梯度检查点
        
        参数：
            model: 要应用梯度检查点的模型
        """
        if not self.config.gradient_checkpointing:
            return
        
        print(f"[显存优化] 启用梯度检查点...")
        
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
            print(f"[显存优化] ✓ 梯度检查点已启用")
        else:
            print(f"[警告] 模型不支持梯度检查点")
    
    def move_to_device(self, model, device: str, keep_on_cpu: bool = False):
        """
        将模型移动到设备（支持 offload）
        
        参数：
            model: 模型
            device: 目标设备
            keep_on_cpu: 是否保持在 CPU（用于 offload）
        
        返回：
            移动后的模型
        """
        if keep_on_cpu:
            return model.to("cpu")
        else:
            return model.to(device)
