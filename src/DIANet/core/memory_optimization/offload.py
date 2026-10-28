# -*- coding: utf-8 -*-
"""
Offload模块 (CPU卸载)
========================================

功能：
1. VAE offload到CPU (~3GB显存节省)
2. Text Encoder offload到CPU (~5GB显存节省)
3. 按需移动到GPU进行计算
4. 计算完成后自动移回CPU

使用场景：
- 训练时不需要梯度的组件（VAE、Text Encoder）
- 需要时临时移到GPU，用完立即释放
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from contextlib import contextmanager
from loguru import logger


class Offloader:
    """Offload管理器 - 负责模型在CPU/GPU之间的移动"""

    def __init__(self, config):
        """
        初始化Offload管理器

        参数：
            config: 配置对象，包含以下字段：
                - enabled: 是否启用offload
                - vae: 是否offload VAE
                - text_encoder: 是否offload text encoder
                - device: GPU设备
        """
        self.config = config
        self.enabled = getattr(config, 'enabled', False)
        self.should_offload_vae = getattr(config, 'vae', False)
        self.should_offload_text_encoder = getattr(config, 'text_encoder', False)
        self.device = getattr(config, 'device', 'cuda')

        # 记录offload的模型
        self.offloaded_models: Dict[str, nn.Module] = {}

    def offload_vae(self, vae: nn.Module) -> nn.Module:
        """
        Offload VAE到CPU

        参数：
            vae: VAE模型

        返回：
            offload后的VAE
        """
        if not self.enabled or not self.should_offload_vae:
            return vae

        logger.info("[Offload] 正在将VAE移至CPU...")
        vae = vae.to("cpu")
        vae.eval()  # 设置为评估模式
        self.offloaded_models["vae"] = vae
        logger.info("[Offload] ✓ VAE已移至CPU")

        return vae

    def offload_text_encoder(
        self,
        text_encoder: nn.Module,
        name: str = "text_encoder"
    ) -> nn.Module:
        """
        Offload Text Encoder到CPU

        参数：
            text_encoder: Text Encoder模型
            name: 模型名称（用于区分text_encoder_1和text_encoder_2）

        返回：
            offload后的Text Encoder
        """
        if not self.enabled or not self.should_offload_text_encoder:
            return text_encoder

        logger.info(f"[Offload] 正在将{name}移至CPU...")
        text_encoder = text_encoder.to("cpu")
        text_encoder.eval()  # 设置为评估模式
        self.offloaded_models[name] = text_encoder
        logger.info(f"[Offload] ✓ {name}已移至CPU")

        return text_encoder

    @contextmanager
    def temporary_load(self, model_name: str):
        """
        临时加载模型到GPU（上下文管理器）

        用法：
        ```python
        with offloader.temporary_load("vae"):
            # VAE在GPU上
            output = vae(input)
        # VAE自动移回CPU
        ```

        参数：
            model_name: 模型名称
        """
        model = self.offloaded_models.get(model_name)

        if model is None:
            # 模型未offload，直接使用
            yield
            return

        try:
            # 移到GPU
            original_device = next(model.parameters()).device
            if original_device != torch.device(self.device):
                model.to(self.device)

            yield

        finally:
            # 移回CPU
            if model is not None:
                model.to("cpu")
                # 清空GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def encode_with_vae(
        self,
        vae: nn.Module,
        images: torch.Tensor,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        使用VAE编码图像（自动处理offload）

        参数：
            vae: VAE模型
            images: 图像张量
            return_dict: 是否返回字典

        返回：
            latent张量或字典
        """
        # 检查VAE是否在GPU
        vae_device = next(vae.parameters()).device
        images_device = images.device

        # 如果VAE在CPU且图像在GPU，需要临时移动
        if vae_device.type == "cpu" and images_device.type == "cuda":
            with self.temporary_load("vae"):
                images = images.to(self.device)
                with torch.no_grad():
                    output = vae.encode(images, return_dict=return_dict)
                # 移回原设备
                if return_dict:
                    output.latent_dist.mean = output.latent_dist.mean.to(images_device)
                    output.latent_dist.std = output.latent_dist.std.to(images_device)
                else:
                    output = output.to(images_device)
                return output
        else:
            # 正常编码
            with torch.no_grad():
                return vae.encode(images, return_dict=return_dict)

    def decode_with_vae(
        self,
        vae: nn.Module,
        latents: torch.Tensor,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        使用VAE解码latents（自动处理offload）

        参数：
            vae: VAE模型
            latents: latent张量
            return_dict: 是否返回字典

        返回：
            图像张量或字典
        """
        # 检查VAE是否在GPU
        vae_device = next(vae.parameters()).device
        latents_device = latents.device

        # 如果VAE在CPU且latents在GPU，需要临时移动
        if vae_device.type == "cpu" and latents_device.type == "cuda":
            with self.temporary_load("vae"):
                latents = latents.to(self.device)
                with torch.no_grad():
                    output = vae.decode(latents, return_dict=return_dict)
                # 移回原设备
                if return_dict:
                    output.sample = output.sample.to(latents_device)
                else:
                    output = output.to(latents_device)
                return output
        else:
            # 正常解码
            with torch.no_grad():
                return vae.decode(latents, return_dict=return_dict)

    def encode_prompt_with_text_encoder(
        self,
        text_encoder: nn.Module,
        tokenizer,
        prompt: str,
        device: Optional[str] = None
    ) -> torch.Tensor:
        """
        使用Text Encoder编码prompt（自动处理offload）

        参数：
            text_encoder: Text Encoder模型
            tokenizer: tokenizer
            prompt: 文本prompt
            device: 目标设备

        返回：
            文本embedding
        """
        device = device or self.device

        # Tokenize
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # 检查Text Encoder是否在GPU
        te_device = next(text_encoder.parameters()).device
        text_encoder_name = "text_encoder"  # 可能需要根据实际名称调整

        # 如果Text Encoder在CPU，需要临时移动
        if te_device.type == "cpu":
            with self.temporary_load(text_encoder_name):
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                with torch.no_grad():
                    embeddings = text_encoder(**text_inputs)[0]
                # 移回目标设备
                embeddings = embeddings.to(device)
                return embeddings
        else:
            # 正常编码
            text_inputs = {k: v.to(te_device) for k, v in text_inputs.items()}
            with torch.no_grad():
                embeddings = text_encoder(**text_inputs)[0]
            return embeddings.to(device)

    def get_offload_info(self) -> Dict[str, Any]:
        """
        获取offload信息

        返回：
            offload信息字典
        """
        info = {
            "enabled": self.enabled,
            "offload_vae": self.should_offload_vae,
            "offload_text_encoder": self.should_offload_text_encoder,
            "offloaded_models": list(self.offloaded_models.keys()),
        }

        # 计算显存节省
        memory_saved = 0
        for name, model in self.offloaded_models.items():
            model_memory = sum(
                p.numel() * p.element_size() for p in model.parameters()
            )
            memory_saved += model_memory

        info["estimated_memory_saved_gb"] = memory_saved / (1024 ** 3)

        return info

    def clear(self):
        """清空offload记录"""
        self.offloaded_models.clear()


def create_offloader(config) -> Offloader:
    """
    工厂函数：创建Offload管理器

    参数：
        config: 配置对象

    返回：
        Offloader实例
    """
    return Offloader(config)
