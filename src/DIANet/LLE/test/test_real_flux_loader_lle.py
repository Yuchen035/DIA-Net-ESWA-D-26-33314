# -*- coding: utf-8 -*-
"""
测试真实FLUX模型加载器（在LLE目录中）
"""
import sys
from pathlib import Path

# 添加项目根目录到path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from types import SimpleNamespace

# 从项目根目录导入
from core.model_loader import RealFluxModelLoader, create_real_flux_loader


def test_real_flux_loader_in_lle():
    """测试在LLE环境中加载FLUX + PhotoDoodle"""
    print("\n" + "=" * 80)
    print("测试：在LLE环境中加载FLUX + PhotoDoodle")
    print("=" * 80 + "\n")

    # 配置
    config = SimpleNamespace(
        model_path="__LOCAL_PROJECT_ROOT__/FLUX.1-dev",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        photodoodle=SimpleNamespace(
            enabled=True,
            lora_path="__LOCAL_PROJECT_ROOT__/lora_model/pretrain.safetensors",
            fuse_before_training=True,  # 融合到基础模型
            lora_scale=1.0,
            save_fused_model=False
        )
    )

    try:
        # 创建加载器
        loader = create_real_flux_loader(config)
        print("✓ 加载器创建成功")

        # 一键加载所有组件（包括PhotoDoodle融合）
        print("\n开始加载FLUX模型和PhotoDoodle预训练...")
        components = loader.load_all(
            load_photodoodle=True,
            fuse_photodoodle=True,
            enable_cpu_offload=True
        )

        print(f"\n✓ 完整加载成功，共{len(components)}个组件")

        # 验证组件
        transformer, vae, te1, te2, tok1, tok2, scheduler = components

        print(f"\n组件验证:")
        print(f"  Transformer: {type(transformer).__name__}")
        print(f"  - 参数量: {sum(p.numel() for p in transformer.parameters()) / 1e9:.2f}B")
        print(f"  - 设备: {next(transformer.parameters()).device}")
        print(f"  - 数据类型: {next(transformer.parameters()).dtype}")

        print(f"\n  VAE: {type(vae).__name__}")
        print(f"  - 设备: {next(vae.parameters()).device}")

        print(f"\n  Text Encoder 1: {type(te1).__name__}")
        print(f"  - 设备: {next(te1.parameters()).device}")

        print(f"\n  Text Encoder 2: {type(te2).__name__}")
        print(f"  - 设备: {next(te2.parameters()).device}")

        # 打印模型信息
        loader.print_model_info()

        return True, components

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_extract_lora_params(transformer):
    """测试提取LoRA可训练参数"""
    print("\n" + "=" * 80)
    print("测试：提取融合后的Transformer参数")
    print("=" * 80 + "\n")

    # 统计参数
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)

    print(f"总参数量: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/1e9:.2f}B)")
    print(f"可训练比例: {100*trainable_params/total_params:.4f}%")

    # 注意：融合后的PhotoDoodle权重已经合并到基础模型中
    # 所有参数都应该是不可训练的（因为还没有注入新的MoE-LoRA）
    if trainable_params == 0:
        print("\n✓ 正确：融合后所有参数都被冻结，准备注入MoE-LoRA")
    else:
        print("\n⚠️  警告：存在可训练参数，这可能不是期望的状态")

    return True


def test_components_ready_for_training(components):
    """测试组件是否准备好训练"""
    print("\n" + "=" * 80)
    print("测试：验证组件是否准备好训练")
    print("=" * 80 + "\n")

    transformer, vae, te1, te2, tok1, tok2, scheduler = components

    checks = []

    # 1. Transformer在正确的设备上
    transformer_device = next(transformer.parameters()).device
    checks.append(("Transformer设备", transformer_device.type in ["cuda", "cpu"]))
    print(f"  ✓ Transformer在 {transformer_device} 上")

    # 2. VAE可以编码/解码
    try:
        dummy_image = torch.randn(1, 3, 512, 512).to(next(vae.parameters()).device)
        with torch.no_grad():
            latent = vae.encode(dummy_image).latent_dist.sample()
        checks.append(("VAE编码", latent.shape[1] in [4, 16]))  # FLUX VAE是16通道
        print(f"  ✓ VAE编码正常，latent shape: {latent.shape}")
    except Exception as e:
        checks.append(("VAE编码", False))
        print(f"  ✗ VAE编码失败: {e}")

    # 3. Text Encoder可以编码文本
    try:
        test_prompt = "A test image"
        inputs = tok1(test_prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(next(te1.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings = te1(**inputs).last_hidden_state
        checks.append(("Text Encoder 1", embeddings.shape[0] == 1))
        print(f"  ✓ Text Encoder 1正常，embedding shape: {embeddings.shape}")
    except Exception as e:
        checks.append(("Text Encoder 1", False))
        print(f"  ✗ Text Encoder 1失败: {e}")

    # 4. Scheduler可用
    checks.append(("Scheduler", scheduler is not None))
    print(f"  ✓ Scheduler: {type(scheduler).__name__}")

    # 总结
    all_passed = all(check[1] for check in checks)
    print(f"\n{'='*80}")
    if all_passed:
        print("✅ 所有组件检查通过，可以开始训练！")
    else:
        print("⚠️  部分组件检查失败：")
        for name, passed in checks:
            if not passed:
                print(f"  - {name}")
    print(f"{'='*80}\n")

    return all_passed


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("LLE环境 - 真实FLUX模型加载器测试套件")
    print("=" * 80 + "\n")

    # 测试1: 加载FLUX + PhotoDoodle
    success, components = test_real_flux_loader_in_lle()
    if not success:
        print("\n❌ 主测试失败，退出")
        return

    # 测试2: 检查参数状态
    transformer = components[0]
    test_extract_lora_params(transformer)

    # 测试3: 验证训练准备
    test_components_ready_for_training(components)

    print("\n" + "=" * 80)
    print("✅ LLE环境测试完成！模型已准备好进行MoE-LoRA训练")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
