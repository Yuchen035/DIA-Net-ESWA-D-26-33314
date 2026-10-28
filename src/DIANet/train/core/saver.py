# -*- coding: utf-8 -*-
"""模型保存模块"""
import os
import torch

from adapter_checkpoint import save_adapter_checkpoint

class ModelSaver:
    """模型保存器"""
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _resolve_save_dtype(self):
        """将配置中的保存精度字符串转换为 torch.dtype"""
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        save_dtype = self.config.save_dtype.lower()
        if save_dtype not in dtype_map:
            raise ValueError(f"不支持的保存精度：{self.config.save_dtype}")
        return dtype_map[save_dtype]
    
    def save_lora(self, transformer, epoch=None):
        """保存 LoRA 权重"""
        if epoch is not None:
            save_path = os.path.join(self.config.output_dir, f"lora_epoch_{epoch}.safetensors")
            self._log(f"[保存] 正在保存 Epoch {epoch} 的 LoRA 权重...")
        else:
            save_path = os.path.join(self.config.output_dir, "lora_final.safetensors")
            self._log(f"[保存] 正在保存最终 LoRA 权重...")
        
        save_dtype = self._resolve_save_dtype()

        # v02 工程协议：raw LoRA/专家/路由键 + 完整配置 JSON，禁止猜测旧格式。
        save_adapter_checkpoint(
            transformer, save_path, save_dtype=save_dtype,
            base_model_reference=self.config.model_path,
        )
        
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        self._log(f"[保存] ✓ LoRA 权重已保存：{save_path}")
        self._log(f"[保存] 必需配置和校验文件：{save_path}.json")
        self._log(f"[保存] 文件大小：{file_size_mb:.2f} MB")
