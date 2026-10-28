"""
对抗损失实现

支持多种对抗损失类型：
- Hinge Loss (推荐，稳定性好)
- GAN Loss (传统BCE loss)
- WGAN Loss (Wasserstein距离)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


class AdversarialLoss(nn.Module):
    """
    对抗损失类

    支持生成器损失和判别器损失的计算
    """

    def __init__(
        self,
        discriminator: nn.Module,
        loss_type: Literal["hinge", "gan", "wgan"] = "hinge",
        routing_mode: Literal["single", "dual"] = "single"
    ):
        """
        初始化对抗损失

        Args:
            discriminator: 判别器网络
            loss_type: 损失类型
                - "hinge": Hinge loss (推荐，SNGAN/StyleGAN使用)
                - "gan": 传统GAN loss (BCE)
                - "wgan": Wasserstein loss
            routing_mode: 路由模式 ("single" or "dual")
        """
        super().__init__()

        self.discriminator = discriminator
        self.loss_type = loss_type
        self.routing_mode = routing_mode

    def generator_loss(
        self,
        fake_images: torch.Tensor,
        timesteps: torch.Tensor,
        categories: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        生成器损失（欺骗判别器）

        目标：让判别器认为假图像是真的

        Args:
            fake_images: [B, C, H, W] 生成的假图像
            timesteps: [B] 时间步
            categories: [B] 曝光类别 (双路由模式需要)

        Returns:
            g_loss: 生成器损失（标量）
        """
        # 判别器对假图像的判断
        if self.routing_mode == "dual":
            if categories is None:
                raise ValueError("categories is required for dual routing mode")
            fake_logits = self.discriminator(fake_images, categories, timesteps)
        else:
            fake_logits = self.discriminator(fake_images, timesteps)

        # 计算损失
        if self.loss_type == "hinge":
            # Hinge loss: 最大化 fake_logits
            # 生成器希望 fake_logits 尽可能大（即判别器认为是真的）
            g_loss = -fake_logits.mean()

        elif self.loss_type == "gan":
            # GAN loss (BCE): log(D(G(z)))
            # 生成器希望 D(fake) = 1
            g_loss = F.binary_cross_entropy_with_logits(
                fake_logits,
                torch.ones_like(fake_logits)
            )

        elif self.loss_type == "wgan":
            # WGAN loss: -E[D(G(z))]
            # 生成器希望 D(fake) 尽可能大
            g_loss = -fake_logits.mean()

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return g_loss

    def discriminator_loss(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        timesteps: torch.Tensor,
        categories: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        判别器损失（区分真假）

        目标：
        - 真图像 -> 高分 (1)
        - 假图像 -> 低分 (0)

        Args:
            real_images: [B, C, H, W] 真实图像
            fake_images: [B, C, H, W] 生成的假图像
            timesteps: [B] 时间步
            categories: [B] 曝光类别 (双路由模式需要)

        Returns:
            d_loss: 判别器损失（标量）
        """
        # 判别器对真图像的判断
        if self.routing_mode == "dual":
            if categories is None:
                raise ValueError("categories is required for dual routing mode")
            real_logits = self.discriminator(real_images, categories, timesteps)
            fake_logits = self.discriminator(fake_images.detach(), categories, timesteps)
        else:
            real_logits = self.discriminator(real_images, timesteps)
            fake_logits = self.discriminator(fake_images.detach(), timesteps)

        # 计算损失
        if self.loss_type == "hinge":
            # Hinge loss:
            # D_loss_real = max(0, 1 - D(real))
            # D_loss_fake = max(0, 1 + D(fake))
            d_loss_real = torch.relu(1.0 - real_logits).mean()
            d_loss_fake = torch.relu(1.0 + fake_logits).mean()
            d_loss = d_loss_real + d_loss_fake

        elif self.loss_type == "gan":
            # GAN loss (BCE):
            # D_loss_real = -log(D(real))
            # D_loss_fake = -log(1 - D(fake))
            d_loss_real = F.binary_cross_entropy_with_logits(
                real_logits,
                torch.ones_like(real_logits)
            )
            d_loss_fake = F.binary_cross_entropy_with_logits(
                fake_logits,
                torch.zeros_like(fake_logits)
            )
            d_loss = d_loss_real + d_loss_fake

        elif self.loss_type == "wgan":
            # WGAN loss: E[D(fake)] - E[D(real)]
            # 判别器希望真图像得分高，假图像得分低
            d_loss = fake_logits.mean() - real_logits.mean()

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return d_loss


def create_adversarial_loss(
    discriminator: nn.Module,
    loss_type: str = "hinge",
    routing_mode: str = "single"
) -> AdversarialLoss:
    """
    工厂函数：创建对抗损失

    Args:
        discriminator: 判别器网络
        loss_type: 损失类型 ("hinge", "gan", "wgan")
        routing_mode: 路由模式 ("single", "dual")

    Returns:
        adv_loss: AdversarialLoss实例

    Examples:
        >>> from core.loss.discriminator import create_discriminator
        >>> disc = create_discriminator({'hidden_dim': 512}, routing_mode="dual")
        >>> adv_loss = create_adversarial_loss(disc, loss_type="hinge", routing_mode="dual")
    """
    return AdversarialLoss(
        discriminator=discriminator,
        loss_type=loss_type,
        routing_mode=routing_mode
    )
