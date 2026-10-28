# -*- coding: utf-8 -*-
"""模型加载模块 - 解耦的模型加载器"""
import os
import sys
from pathlib import Path
import torch
from typing import Tuple

# 添加本地库路径
CURRENT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(CURRENT_DIR / "diffusers" / "src"))
sys.path.insert(0, str(CURRENT_DIR / "peft" / "src"))

from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTokenizer, CLIPTextModel, T5TokenizerFast, T5EncoderModel

class ModelLoader:
    """模型加载器 - 负责加载所有模型组件"""
    
    def __init__(self, config, memory_optimizer):
        self.config = config
        self.memory_optimizer = memory_optimizer
        self.logger = None
    
    def set_logger(self, logger):
        self.logger = logger
    
    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
    
    def load_all(self) -> Tuple:
        """加载所有模型组件"""
        self._log(f"[模型加载] 开始加载 FLUX Kontext 模型")
        self._log(f"[模型加载] 模型路径：{self.config.model_path}")
        
        dtype = self.config.get_dtype()
        device = self.config.device
        
        # 1. 加载 Transformer
        self._log(f"[模型加载] 正在加载 Transformer...")
        transformer = FluxTransformer2DModel.from_pretrained(
            self.config.model_path,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        
        # 应用量化（如果启用）
        transformer = self.memory_optimizer.apply_quantization(transformer, "transformer")
        transformer = self.memory_optimizer.move_to_device(transformer, device)
        
        # 2. 加载 VAE
        self._log(f"[模型加载] 正在加载 VAE...")
        vae = AutoencoderKL.from_pretrained(
            self.config.model_path,
            subfolder="vae",
            torch_dtype=dtype,
        )
        
        # VAE offload（如果启用）
        vae_device = "cpu" if self.config.vae_offload else device
        vae = vae.to(vae_device, dtype=dtype)
        
        # 3. 加载 Text Encoders
        self._log(f"[模型加载] 正在加载 Text Encoders...")
        text_encoder_1 = CLIPTextModel.from_pretrained(
            self.config.model_path,
            subfolder="text_encoder",
            torch_dtype=dtype,
        )
        tokenizer_1 = CLIPTokenizer.from_pretrained(
            self.config.model_path,
            subfolder="tokenizer",
        )
        
        text_encoder_2 = T5EncoderModel.from_pretrained(
            self.config.model_path,
            subfolder="text_encoder_2",
            torch_dtype=dtype,
        )
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            self.config.model_path,
            subfolder="tokenizer_2",
        )
        
        # 应用量化（如果启用）
        text_encoder_2 = self.memory_optimizer.apply_quantization(text_encoder_2, "text_encoder")
        
        # Text Encoder offload（如果启用）
        te_device = "cpu" if self.config.te_offload else device
        text_encoder_1 = text_encoder_1.to(te_device, dtype=dtype)
        text_encoder_2 = text_encoder_2.to(te_device, dtype=dtype)
        
        # 4. 加载 Scheduler
        self._log(f"[模型加载] 正在加载 Scheduler...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.model_path,
            subfolder="scheduler",
        )
        
        self._log(f"[模型加载] ✓ 所有组件加载完成")
        
        return transformer, vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, scheduler
