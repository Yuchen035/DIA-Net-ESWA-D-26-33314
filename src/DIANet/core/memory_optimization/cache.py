# -*- coding: utf-8 -*-
"""
Latent缓存模块 (LatentCache)
========================================

功能：
1. 预编码所有图像为latents（通过VAE）
2. 保存到磁盘或内存
3. 训练时直接加载latents，跳过VAE编码
4. 性能提升：2-3倍训练速度

使用场景：
- Few-Shot训练（数据量小，可以全部缓存）
- 重复训练同一数据集
- VAE编码开销大的场景
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
from safetensors.torch import save_file, load_file
from collections import OrderedDict
from loguru import logger


class LatentCache:
    """Latent缓存管理器"""

    def __init__(self, config, vae: Optional[nn.Module] = None, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        """
        初始化缓存管理器

        参数：
            config: 配置对象，包含以下字段：
                - enabled: 是否启用缓存
                - mode: 缓存模式 ("memory" 或 "disk")
                - disk_path: 磁盘缓存路径
            vae: VAE模型（用于编码）
            device: 计算设备
            dtype: 数据类型
        """
        self.config = config
        self.enabled = getattr(config, 'enabled', False)
        self.mode = getattr(config, 'mode', 'memory')
        self.disk_path = Path(getattr(config, 'disk_path', './cache'))
        self.vae = vae
        self.device = device
        self.dtype = dtype

        # 内存缓存
        self.latent_cache: Dict[str, torch.Tensor] = {}

        # 创建磁盘缓存目录
        if self.enabled and self.mode == "disk":
            self.disk_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"[Cache] 磁盘缓存目录: {self.disk_path}")

    def get_cache_key(self, image_path: Path, is_input: bool) -> str:
        """
        获取缓存key

        参数：
            image_path: 图像路径
            is_input: 是否为输入图像（否则为目标图像）

        返回：
            缓存key
        """
        img_type = "input" if is_input else "target"
        return f"{image_path.stem}_{img_type}"

    def get_cache_path(self, image_path: Path, is_input: bool) -> Path:
        """
        获取磁盘缓存文件路径

        参数：
            image_path: 图像路径
            is_input: 是否为输入图像

        返回：
            缓存文件路径
        """
        cache_key = self.get_cache_key(image_path, is_input)
        return self.disk_path / f"{cache_key}.safetensors"

    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        编码图像为latent

        参数：
            image_tensor: 图像张量 (C, H, W)，范围 [-1, 1]

        返回：
            latent张量
        """
        if self.vae is None:
            raise ValueError("VAE模型未设置，无法编码图像")

        # 记录VAE原始设备
        vae_device_original = next(self.vae.parameters()).device

        # 临时移动VAE到GPU
        if vae_device_original != torch.device(self.device):
            self.vae.to(self.device)

        # 添加batch维度
        image_tensor = image_tensor.unsqueeze(0).to(self.device, dtype=self.dtype)

        # 编码
        with torch.no_grad():
            latent_dist = self.vae.encode(image_tensor).latent_dist
            latent = latent_dist.sample()

            # Flux VAE的scaling和shift
            if hasattr(self.vae.config, 'scaling_factor'):
                latent = (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # 移除batch维度
        latent = latent.squeeze(0).cpu()

        # 移回原设备
        if vae_device_original != torch.device(self.device):
            self.vae.to(vae_device_original)

        return latent

    def cache_dataset(self, dataset, show_progress: bool = True):
        """
        缓存整个数据集的latents

        参数：
            dataset: 数据集对象（需要有image_paths和target_dir属性）
            show_progress: 是否显示进度条
        """
        if not self.enabled:
            logger.info("[Cache] 缓存功能未启用，跳过")
            return

        logger.info(f"[Cache] 开始缓存latents")
        logger.info(f"[Cache] 模式：{self.mode}")
        logger.info(f"[Cache] 图像数量：{len(dataset)}")

        # 遍历数据集
        iterator = tqdm(range(len(dataset)), desc="缓存latents") if show_progress else range(len(dataset))

        for idx in iterator:
            # 获取图像路径
            image_path = dataset.image_paths[idx]
            cache_key_input = self.get_cache_key(image_path, is_input=True)
            cache_key_target = self.get_cache_key(image_path, is_input=False)

            # 检查是否已缓存
            if self.is_cached(image_path, is_input=True) and self.is_cached(image_path, is_input=False):
                continue

            # 加载图像（使用数据集的加载方法）
            try:
                sample = dataset[idx]
                input_image = sample['input_image']  # PIL Image
                target_image = sample['target_image']  # PIL Image

                # 转换为张量（假设数据集有transforms）
                from torchvision import transforms
                to_tensor = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])

                input_tensor = to_tensor(input_image)
                target_tensor = to_tensor(target_image)

                # 编码
                input_latent = self.encode_image(input_tensor)
                target_latent = self.encode_image(target_tensor)

                # 保存
                self.save_latent(image_path, input_latent, is_input=True)
                self.save_latent(image_path, target_latent, is_input=False)

            except Exception as e:
                logger.error(f"[Cache] 缓存失败 {image_path}: {e}")
                continue

        logger.info(f"[Cache] ✓ 缓存完成")

    def save_latent(self, image_path: Path, latent: torch.Tensor, is_input: bool):
        """
        保存latent到缓存

        参数：
            image_path: 图像路径
            latent: latent张量
            is_input: 是否为输入图像
        """
        if self.mode == "memory":
            # 保存到内存
            cache_key = self.get_cache_key(image_path, is_input)
            self.latent_cache[cache_key] = latent

        elif self.mode == "disk":
            # 保存到磁盘
            cache_path = self.get_cache_path(image_path, is_input)
            save_file(
                OrderedDict([("latent", latent)]),
                str(cache_path)
            )

    def load_latent(self, image_path: Path, is_input: bool) -> Optional[torch.Tensor]:
        """
        从缓存加载latent

        参数：
            image_path: 图像路径
            is_input: 是否为输入图像

        返回：
            latent张量，如果未缓存返回None
        """
        if not self.enabled:
            return None

        if self.mode == "memory":
            # 从内存加载
            cache_key = self.get_cache_key(image_path, is_input)
            return self.latent_cache.get(cache_key)

        elif self.mode == "disk":
            # 从磁盘加载
            cache_path = self.get_cache_path(image_path, is_input)
            if cache_path.exists():
                state_dict = load_file(str(cache_path))
                return state_dict["latent"]
            return None

    def is_cached(self, image_path: Path, is_input: bool) -> bool:
        """
        检查是否已缓存

        参数：
            image_path: 图像路径
            is_input: 是否为输入图像

        返回：
            是否已缓存
        """
        if not self.enabled:
            return False

        if self.mode == "memory":
            cache_key = self.get_cache_key(image_path, is_input)
            return cache_key in self.latent_cache

        elif self.mode == "disk":
            cache_path = self.get_cache_path(image_path, is_input)
            return cache_path.exists()

        return False

    def clear_cache(self):
        """清空缓存"""
        if self.mode == "memory":
            self.latent_cache.clear()
            logger.info("[Cache] 内存缓存已清空")

        elif self.mode == "disk":
            # 删除磁盘缓存文件
            if self.disk_path.exists():
                for cache_file in self.disk_path.glob("*.safetensors"):
                    cache_file.unlink()
                logger.info("[Cache] 磁盘缓存已清空")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息

        返回：
            缓存信息字典
        """
        info = {
            "enabled": self.enabled,
            "mode": self.mode,
        }

        if self.mode == "memory":
            info["cached_count"] = len(self.latent_cache)
            # 计算内存占用
            memory_bytes = sum(
                latent.numel() * latent.element_size()
                for latent in self.latent_cache.values()
            )
            info["memory_usage_mb"] = memory_bytes / (1024 ** 2)

        elif self.mode == "disk":
            info["cache_dir"] = str(self.disk_path)
            if self.disk_path.exists():
                cache_files = list(self.disk_path.glob("*.safetensors"))
                info["cached_count"] = len(cache_files)
                # 计算磁盘占用
                disk_bytes = sum(f.stat().st_size for f in cache_files)
                info["disk_usage_mb"] = disk_bytes / (1024 ** 2)

        return info


def create_latent_cache(config, vae: Optional[nn.Module] = None, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> LatentCache:
    """
    工厂函数：创建Latent缓存管理器

    参数：
        config: 配置对象
        vae: VAE模型
        device: 设备
        dtype: 数据类型

    返回：
        LatentCache实例
    """
    return LatentCache(config, vae, device, dtype)
