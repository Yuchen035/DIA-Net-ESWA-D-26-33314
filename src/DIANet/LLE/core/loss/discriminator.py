"""
判别器网络 - 用于对抗训练

支持两种模式：
- SingleRouteDiscriminator: 单路由模式（图像 + 时间步）
- DualRouteDiscriminator: 双路由模式（图像 + 曝光类别 + 时间步）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SinusoidalPositionEmbedding(nn.Module):
    """
    正弦位置编码 - 用于时间步embedding
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] 时间步，范围 [0, 1]

        Returns:
            embeddings: [B, dim] 位置编码
        """
        device = timesteps.device
        dtype = timesteps.dtype
        half_dim = self.dim // 2

        # 计算频率
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)

        # 时间步扩展到 [B, half_dim]
        emb = timesteps[:, None] * emb[None, :]

        # sin和cos拼接
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb


class SingleRouteDiscriminator(nn.Module):
    """
    单路由模式判别器

    输入: (image, timestep)
    输出: 真实性得分 [B, 1]

    用于标准MoE-LoRA训练的对抗学习
    """

    def __init__(
        self,
        image_channels: int = 3,
        hidden_dim: int = 512,
        num_layers: int = 3,
        image_size: int = 512
    ):
        """
        初始化单路由判别器

        Args:
            image_channels: 输入图像通道数（默认3=RGB）
            hidden_dim: 隐藏层维度
            num_layers: CNN层数
            image_size: 输入图像大小
        """
        super().__init__()

        self.image_channels = image_channels
        self.hidden_dim = hidden_dim

        # 图像编码器（CNN）
        encoder_layers = []
        in_ch = image_channels
        out_ch = 64

        for i in range(num_layers):
            encoder_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_ch = out_ch
            out_ch = min(out_ch * 2, hidden_dim)

        # 自适应池化到固定大小
        encoder_layers.append(nn.AdaptiveAvgPool2d(1))

        self.image_encoder = nn.Sequential(*encoder_layers)

        # 时间步embedding（Sinusoidal）
        self.time_embed = SinusoidalPositionEmbedding(hidden_dim)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(in_ch + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 分类头
        self.head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, image: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            image: [B, C, H, W] 输入图像，范围 [-1, 1]
            timestep: [B] 时间步，范围 [0, 1]

        Returns:
            logits: [B, 1] 真实性得分（未经sigmoid）
        """
        # 图像特征提取
        img_feat = self.image_encoder(image)  # [B, C, 1, 1]
        img_feat = img_feat.squeeze(-1).squeeze(-1)  # [B, C]

        # 时间步embedding
        time_feat = self.time_embed(timestep)  # [B, hidden_dim]

        # 融合
        fused_feat = torch.cat([img_feat, time_feat], dim=-1)  # [B, C + hidden_dim]
        fused_feat = self.fusion(fused_feat)  # [B, hidden_dim // 2]

        # 分类
        logits = self.head(fused_feat)  # [B, 1]

        return logits


class DualRouteDiscriminator(nn.Module):
    """
    双路由模式判别器

    输入: (image, exposure_category, timestep)
    输出: 真实性得分 [B, 1]

    用于双路由MoE-LoRA训练的对抗学习，考虑曝光度类别信息
    """

    def __init__(
        self,
        image_channels: int = 3,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_categories: int = 5,
        image_size: int = 512
    ):
        """
        初始化双路由判别器

        Args:
            image_channels: 输入图像通道数（默认3=RGB）
            hidden_dim: 隐藏层维度
            num_layers: CNN层数
            num_categories: 曝光度类别数（默认5）
            image_size: 输入图像大小
        """
        super().__init__()

        self.image_channels = image_channels
        self.hidden_dim = hidden_dim
        self.num_categories = num_categories

        # 图像编码器（CNN）- 与SingleRoute相同
        encoder_layers = []
        in_ch = image_channels
        out_ch = 64

        for i in range(num_layers):
            encoder_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_ch = out_ch
            out_ch = min(out_ch * 2, hidden_dim)

        encoder_layers.append(nn.AdaptiveAvgPool2d(1))

        self.image_encoder = nn.Sequential(*encoder_layers)

        # 时间步embedding
        self.time_embed = SinusoidalPositionEmbedding(hidden_dim)

        # 【新增】曝光类别embedding
        self.category_embed = nn.Embedding(num_categories, hidden_dim)

        # 融合层（输入维度变为: img_feat + time_feat + category_feat）
        fusion_input_dim = in_ch + hidden_dim + hidden_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 分类头
        self.head = nn.Linear(hidden_dim // 2, 1)

    def forward(
        self,
        image: torch.Tensor,
        category: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            image: [B, C, H, W] 输入图像，范围 [-1, 1]
            category: [B] 曝光度类别ID，范围 [0, num_categories-1]
            timestep: [B] 时间步，范围 [0, 1]

        Returns:
            logits: [B, 1] 真实性得分（未经sigmoid）
        """
        # 图像特征提取
        img_feat = self.image_encoder(image)  # [B, C, 1, 1]
        img_feat = img_feat.squeeze(-1).squeeze(-1)  # [B, C]

        # 时间步embedding
        time_feat = self.time_embed(timestep)  # [B, hidden_dim]

        # 曝光类别embedding
        cat_feat = self.category_embed(category)  # [B, hidden_dim]

        # 融合
        fused_feat = torch.cat([img_feat, time_feat, cat_feat], dim=-1)  # [B, fusion_input_dim]
        fused_feat = self.fusion(fused_feat)  # [B, hidden_dim // 2]

        # 分类
        logits = self.head(fused_feat)  # [B, 1]

        return logits


def create_discriminator(config: dict, routing_mode: str = "single") -> nn.Module:
    """
    工厂函数：根据配置创建判别器

    Args:
        config: 判别器配置字典
        routing_mode: 路由模式 ("single" or "dual")

    Returns:
        discriminator: 判别器实例

    Examples:
        >>> config = {
        ...     'image_channels': 3,
        ...     'hidden_dim': 512,
        ...     'num_layers': 3,
        ...     'num_categories': 5  # 双路由需要
        ... }
        >>> disc = create_discriminator(config, routing_mode="dual")
    """
    if routing_mode == "single":
        return SingleRouteDiscriminator(
            image_channels=config.get('image_channels', 3),
            hidden_dim=config.get('hidden_dim', 512),
            num_layers=config.get('num_layers', 3),
            image_size=config.get('image_size', 512)
        )
    elif routing_mode == "dual":
        if 'num_categories' not in config:
            raise ValueError("DualRouteDiscriminator requires 'num_categories' in config")

        return DualRouteDiscriminator(
            image_channels=config.get('image_channels', 3),
            hidden_dim=config.get('hidden_dim', 512),
            num_layers=config.get('num_layers', 3),
            num_categories=config['num_categories'],
            image_size=config.get('image_size', 512)
        )
    else:
        raise ValueError(f"Unknown routing_mode: {routing_mode}. Expected 'single' or 'dual'")
