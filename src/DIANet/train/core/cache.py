# -*- coding: utf-8 -*-
"""
Latent 缓存模块 - 用于加速训练
==========================================

功能：
1. 预编码所有图像为 latents（通过 VAE）
2. 保存到磁盘或内存
3. 训练时直接加载 latents，跳过 VAE 编码

性能提升：2-3 倍训练速度
"""
import os
import torch
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import save_file, load_file
from collections import OrderedDict
import hashlib
import json


class LatentCache:
    """Latent 缓存管理器"""

    def __init__(self, config, vae, device, dtype):
        """
        初始化缓存管理器

        参数：
            config: 训练配置
            vae: VAE 模型
            device: 设备
            dtype: 数据类型
        """
        self.config = config
        self.vae = vae
        self.device = device
        self.dtype = dtype
        self.cache_dir = Path(config.cache_dir)
        self.cache_to_disk = config.cache_to_disk
        self.cache_to_memory = config.cache_latents and not config.cache_to_disk

        # 内存缓存
        self.latent_cache = {}

        if self.cache_to_disk:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, image_path: Path, is_input: bool) -> Path:
        """
        获取缓存文件路径

        参数：
            image_path: 图像路径
            is_input: 是否为输入图像（否则为目标图像）

        返回：
            缓存文件路径
        """
        # 使用图像路径的哈希作为缓存文件名
        img_type = "input" if is_input else "target"
        cache_filename = f"{image_path.stem}_{img_type}.safetensors"
        return self.cache_dir / cache_filename

    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        编码图像为 latent

        参数：
            image_tensor: 图像张量 (C, H, W)，范围 [-1, 1]

        返回：
            latent 张量
        """
        # 临时移动 VAE 到 GPU（如果需要）
        vae_device_original = next(self.vae.parameters()).device
        if vae_device_original != torch.device(self.device):
            self.vae.to(self.device)

        # 添加 batch 维度
        image_tensor = image_tensor.unsqueeze(0).to(self.device, dtype=self.dtype)

        # 编码
        with torch.no_grad():
            latent = self.vae.encode(image_tensor).latent_dist.sample()
            latent = (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # 移除 batch 维度
        latent = latent.squeeze(0)

        # 移回原设备
        if vae_device_original != torch.device(self.device):
            self.vae.to(vae_device_original)

        return latent.cpu()

    def cache_dataset(self, dataset, logger=None):
        """
        缓存整个数据集的 latents

        参数：
            dataset: 数据集对象
            logger: 日志记录器
        """
        def _log(msg):
            if logger:
                logger.info(msg)
            else:
                print(msg)

        _log(f"\n[Cache Latents] 开始缓存 latents")
        _log(f"[Cache Latents] 模式：{'磁盘' if self.cache_to_disk else '内存'}")
        _log(f"[Cache Latents] 图像数量：{len(dataset)}")

        if self.cache_to_disk:
            _log(f"[Cache Latents] 缓存目录：{self.cache_dir}")

        # 遍历数据集
        for idx in tqdm(range(len(dataset)), desc="缓存 latents"):
            # 获取原始图像路径
            image_path = dataset.image_paths[idx]

            # 检查缓存
            input_cache_path = self.get_cache_path(image_path, is_input=True)
            target_cache_path = self.get_cache_path(image_path, is_input=False)

            # 缓存 key（用于内存缓存）
            cache_key_input = f"{idx}_input"
            cache_key_target = f"{idx}_target"

            # 如果已缓存，跳过
            if self.cache_to_disk:
                if input_cache_path.exists() and target_cache_path.exists():
                    continue
            elif cache_key_input in self.latent_cache and cache_key_target in self.latent_cache:
                continue

            # 加载图像（不使用 DataLoader 的变换）
            from PIL import Image
            import torchvision.transforms as transforms

            input_path = dataset.image_paths[idx]
            target_path = dataset.target_dir / input_path.name

            input_image = Image.open(input_path).convert("RGB").resize(
                (dataset.resolution, dataset.resolution)
            )
            target_image = Image.open(target_path).convert("RGB").resize(
                (dataset.resolution, dataset.resolution)
            )

            # 转换为张量（范围 [-1, 1]）
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            input_tensor = transform(input_image)
            target_tensor = transform(target_image)

            # 编码
            input_latent = self.encode_image(input_tensor)
            target_latent = self.encode_image(target_tensor)

            # 保存
            if self.cache_to_disk:
                # 保存到磁盘
                save_file(
                    OrderedDict([("latent", input_latent)]),
                    str(input_cache_path)
                )
                save_file(
                    OrderedDict([("latent", target_latent)]),
                    str(target_cache_path)
                )
            else:
                # 保存到内存
                self.latent_cache[cache_key_input] = input_latent
                self.latent_cache[cache_key_target] = target_latent

        _log(f"[Cache Latents] ✓ 缓存完成")

    def load_latent(self, image_path: Path, is_input: bool, idx: int) -> torch.Tensor:
        """
        加载缓存的 latent

        参数：
            image_path: 图像路径
            is_input: 是否为输入图像
            idx: 数据集索引

        返回：
            latent 张量
        """
        if self.cache_to_disk:
            # 从磁盘加载
            cache_path = self.get_cache_path(image_path, is_input)
            state_dict = load_file(str(cache_path))
            return state_dict["latent"]
        else:
            # 从内存加载
            cache_key = f"{idx}_{'input' if is_input else 'target'}"
            return self.latent_cache[cache_key]
