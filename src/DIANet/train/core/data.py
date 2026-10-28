# -*- coding: utf-8 -*-
"""数据加载模块"""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

class FluxKontextDataset(Dataset):
    """FLUX Kontext 数据集（支持 Latent 缓存）"""

    def __init__(self, config, latent_cache=None):
        """
        初始化数据集

        参数：
            config: 训练配置
            latent_cache: Latent 缓存管理器（可选）
        """
        self.config = config
        self.input_dir = Path(config.input_image_dir)
        self.target_dir = Path(config.target_image_dir)
        self.prompt = config.prompt
        self.resolution = config.resolution
        self.latent_cache = latent_cache  # 缓存管理器
        self.use_cache = (latent_cache is not None) and (config.cache_latents or config.cache_to_disk)

        # 获取图像列表
        self.image_paths = sorted(list(self.input_dir.glob("*.png")) + list(self.input_dir.glob("*.jpg")))

        if len(self.image_paths) == 0:
            raise ValueError(f"在 {self.input_dir} 中没有找到图像")

        print(f"[数据集] 加载了 {len(self.image_paths)} 对训练图像")
        if self.use_cache:
            print(f"[数据集] 启用 Latent 缓存模式")

        # 图像变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        获取数据项

        如果启用缓存，直接返回 latents；否则返回图像张量
        """
        if self.use_cache:
            # 从缓存加载 latents
            input_path = self.image_paths[idx]
            input_latent = self.latent_cache.load_latent(input_path, is_input=True, idx=idx)
            target_latent = self.latent_cache.load_latent(input_path, is_input=False, idx=idx)

            return {
                "input_latent": input_latent,   # 已编码的 latent
                "target_latent": target_latent, # 已编码的 latent
                "prompt": self.prompt,
            }
        else:
            # 加载图像（原有逻辑）
            input_path = self.image_paths[idx]
            input_image = Image.open(input_path).convert("RGB")

            target_path = self.target_dir / input_path.name
            if not target_path.exists():
                raise FileNotFoundError(f"找不到目标图像：{target_path}")
            target_image = Image.open(target_path).convert("RGB")

            # 调整大小
            input_image = input_image.resize((self.resolution, self.resolution))
            target_image = target_image.resize((self.resolution, self.resolution))

            # 转换为张量
            input_tensor = self.transform(input_image)
            target_tensor = self.transform(target_image)

            return {
                "input_image": input_tensor,
                "target_image": target_tensor,
                "prompt": self.prompt,
            }

def create_dataloader(config, latent_cache=None) -> DataLoader:
    """
    创建数据加载器

    参数：
        config: 训练配置
        latent_cache: Latent 缓存管理器（可选）

    返回：
        DataLoader 对象
    """
    dataset = FluxKontextDataset(config, latent_cache=latent_cache)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return dataset, dataloader
