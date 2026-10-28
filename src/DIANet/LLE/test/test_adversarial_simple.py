"""
对抗学习组件简单测试

测试阶段4的核心功能：
1. SingleRouteDiscriminator
2. DualRouteDiscriminator
3. AdversarialLoss (hinge/gan/wgan)
"""

import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.loss.discriminator import SingleRouteDiscriminator, DualRouteDiscriminator, create_discriminator
from core.loss.adversarial import AdversarialLoss, create_adversarial_loss


def test_single_route_discriminator():
    """测试1: 单路由判别器"""
    print("\n" + "=" * 70)
    print("测试1: 单路由判别器")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建判别器
    disc = SingleRouteDiscriminator(
        image_channels=3,
        hidden_dim=256,
        num_layers=3
    ).to(device)

    print(f"  创建SingleRouteDiscriminator")
    print(f"    参数量: {sum(p.numel() for p in disc.parameters()):,}")

    # 测试前向传播
    batch_size = 4
    images = torch.randn(batch_size, 3, 64, 64).to(device)
    timesteps = torch.rand(batch_size).to(device)

    logits = disc(images, timesteps)

    assert logits.shape == (batch_size, 1), f"输出形状错误: {logits.shape}"
    print(f"  ✓ 前向传播成功: {images.shape} -> {logits.shape}")
    print(f"    输出范围: [{logits.min():.3f}, {logits.max():.3f}]")

    print("✓ 单路由判别器测试通过")


def test_dual_route_discriminator():
    """测试2: 双路由判别器"""
    print("\n" + "=" * 70)
    print("测试2: 双路由判别器")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建判别器
    disc = DualRouteDiscriminator(
        image_channels=3,
        hidden_dim=256,
        num_layers=3,
        num_categories=5
    ).to(device)

    print(f"  创建DualRouteDiscriminator")
    print(f"    参数量: {sum(p.numel() for p in disc.parameters()):,}")
    print(f"    曝光类别数: 5")

    # 测试前向传播
    batch_size = 4
    images = torch.randn(batch_size, 3, 64, 64).to(device)
    categories = torch.randint(0, 5, (batch_size,)).to(device)
    timesteps = torch.rand(batch_size).to(device)

    logits = disc(images, categories, timesteps)

    assert logits.shape == (batch_size, 1), f"输出形状错误: {logits.shape}"
    print(f"  ✓ 前向传播成功: images{images.shape} + categories{categories.shape} -> {logits.shape}")
    print(f"    输出范围: [{logits.min():.3f}, {logits.max():.3f}]")

    print("✓ 双路由判别器测试通过")


def test_adversarial_loss_hinge():
    """测试3: Hinge对抗损失"""
    print("\n" + "=" * 70)
    print("测试3: Hinge对抗损失")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建判别器和对抗损失
    disc = SingleRouteDiscriminator(hidden_dim=256, num_layers=2).to(device)
    adv_loss = AdversarialLoss(disc, loss_type="hinge", routing_mode="single")

    batch_size = 4
    real_images = torch.randn(batch_size, 3, 64, 64).to(device)
    fake_images = torch.randn(batch_size, 3, 64, 64).to(device)
    timesteps = torch.rand(batch_size).to(device)

    # 测试判别器损失
    d_loss = adv_loss.discriminator_loss(real_images, fake_images, timesteps)
    assert d_loss.ndim == 0, "判别器损失应该是标量"
    assert d_loss.item() >= 0, "Hinge loss应该非负"
    print(f"  ✓ 判别器损失 (Hinge): {d_loss.item():.4f}")

    # 测试生成器损失
    g_loss = adv_loss.generator_loss(fake_images, timesteps)
    assert g_loss.ndim == 0, "生成器损失应该是标量"
    print(f"  ✓ 生成器损失 (Hinge): {g_loss.item():.4f}")

    print("✓ Hinge对抗损失测试通过")


def test_adversarial_loss_gan():
    """测试4: GAN对抗损失"""
    print("\n" + "=" * 70)
    print("测试4: GAN对抗损失")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    disc = SingleRouteDiscriminator(hidden_dim=256, num_layers=2).to(device)
    adv_loss = AdversarialLoss(disc, loss_type="gan", routing_mode="single")

    batch_size = 4
    real_images = torch.randn(batch_size, 3, 64, 64).to(device)
    fake_images = torch.randn(batch_size, 3, 64, 64).to(device)
    timesteps = torch.rand(batch_size).to(device)

    d_loss = adv_loss.discriminator_loss(real_images, fake_images, timesteps)
    g_loss = adv_loss.generator_loss(fake_images, timesteps)

    print(f"  ✓ 判别器损失 (GAN): {d_loss.item():.4f}")
    print(f"  ✓ 生成器损失 (GAN): {g_loss.item():.4f}")

    print("✓ GAN对抗损失测试通过")


def test_dual_route_adversarial():
    """测试5: 双路由对抗损失"""
    print("\n" + "=" * 70)
    print("测试5: 双路由对抗损失")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建双路由判别器
    disc = DualRouteDiscriminator(
        hidden_dim=256,
        num_layers=2,
        num_categories=5
    ).to(device)

    adv_loss = AdversarialLoss(disc, loss_type="hinge", routing_mode="dual")

    batch_size = 4
    real_images = torch.randn(batch_size, 3, 64, 64).to(device)
    fake_images = torch.randn(batch_size, 3, 64, 64).to(device)
    categories = torch.randint(0, 5, (batch_size,)).to(device)
    timesteps = torch.rand(batch_size).to(device)

    print(f"  曝光类别: {categories.tolist()}")

    # 测试损失
    d_loss = adv_loss.discriminator_loss(real_images, fake_images, timesteps, categories)
    g_loss = adv_loss.generator_loss(fake_images, timesteps, categories)

    print(f"  ✓ 判别器损失: {d_loss.item():.4f}")
    print(f"  ✓ 生成器损失: {g_loss.item():.4f}")

    print("✓ 双路由对抗损失测试通过")


def test_factory_functions():
    """测试6: 工厂函数"""
    print("\n" + "=" * 70)
    print("测试6: 工厂函数")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试create_discriminator
    config = {
        'image_channels': 3,
        'hidden_dim': 256,
        'num_layers': 2,
        'num_categories': 5
    }

    disc_single = create_discriminator(config, routing_mode="single")
    disc_dual = create_discriminator(config, routing_mode="dual")

    print(f"  ✓ create_discriminator (single): {type(disc_single).__name__}")
    print(f"  ✓ create_discriminator (dual): {type(disc_dual).__name__}")

    # 测试create_adversarial_loss
    adv_loss_single = create_adversarial_loss(disc_single, loss_type="hinge", routing_mode="single")
    adv_loss_dual = create_adversarial_loss(disc_dual, loss_type="gan", routing_mode="dual")

    print(f"  ✓ create_adversarial_loss (single, hinge): loss_type={adv_loss_single.loss_type}")
    print(f"  ✓ create_adversarial_loss (dual, gan): loss_type={adv_loss_dual.loss_type}")

    print("✓ 工厂函数测试通过")


def main():
    print("\n" + "=" * 70)
    print("对抗学习组件测试（阶段4验收）")
    print("=" * 70)

    try:
        test_single_route_discriminator()
        test_dual_route_discriminator()
        test_adversarial_loss_hinge()
        test_adversarial_loss_gan()
        test_dual_route_adversarial()
        test_factory_functions()

        print("\n" + "=" * 70)
        print("✅ 阶段4所有对抗学习组件测试通过！")
        print("=" * 70)

        print("\n总结:")
        print("  ✓ SingleRouteDiscriminator正常工作")
        print("  ✓ DualRouteDiscriminator正常工作")
        print("  ✓ Hinge对抗损失正常工作")
        print("  ✓ GAN对抗损失正常工作")
        print("  ✓ 双路由对抗损失正常工作")
        print("  ✓ 工厂函数正常工作")
        print("=" * 70)

        print("\n✨ 阶段4（对抗学习）核心组件已完成！\n")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
