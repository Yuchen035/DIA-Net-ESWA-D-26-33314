"""
推理管道测试

测试双路由MoE-LoRA推理流程
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import tempfile

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.mock_flux import MockFluxModelLoader
from core.lora.moe.moe_lora_layer import MoELoRALayer
from core.inference import InferencePipeline, create_inference_pipeline_from_checkpoint


def create_test_checkpoint(temp_dir: Path, num_blocks: int = 4):
    """创建测试检查点文件"""
    checkpoint_path = temp_dir / "test_checkpoint.pt"

    # 创建简单的state dict
    state_dict = {}

    # 为每个block创建lora权重
    for i in range(num_blocks):
        # 假设lora_A和lora_B各有9个专家（5 specific + 4 shared）
        num_experts = 9
        in_features = 256
        out_features = 256
        rank = 8

        lora_A_weights = [
            torch.randn(in_features, rank) * 0.01  # Shape: (in_features, rank)
            for _ in range(num_experts)
        ]
        lora_B_weights = [
            torch.randn(rank, out_features) * 0.01  # Shape: (rank, out_features)
            for _ in range(num_experts)
        ]

        state_dict[f"blocks.{i}.0.lora_A"] = lora_A_weights
        state_dict[f"blocks.{i}.0.lora_B"] = lora_B_weights

    # 保存
    torch.save(state_dict, checkpoint_path)
    return checkpoint_path


def test_inference_pipeline_creation():
    """测试推理管道创建"""
    print("\n" + "=" * 70)
    print("测试: 推理管道创建")
    print("=" * 70)

    # 加载Mock FLUX
    loader = MockFluxModelLoader(device="cuda" if torch.cuda.is_available() else "cpu")
    (transformer, vae, text_encoder_1, text_encoder_2,
     tokenizer_1, tokenizer_2, scheduler) = loader.load_all()

    # 创建推理管道
    pipeline = InferencePipeline(
        transformer=transformer,
        vae=vae,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        scheduler=scheduler,
        num_specific=5,
        num_shared=4,
        rank=8
    )

    assert pipeline is not None
    assert not pipeline.lora_injected
    assert not pipeline.lora_loaded

    print("✓ 推理管道创建成功")


def test_lora_injection():
    """测试LoRA注入"""
    print("\n" + "=" * 70)
    print("测试: LoRA注入")
    print("=" * 70)

    # 加载Mock FLUX
    loader = MockFluxModelLoader(device="cuda" if torch.cuda.is_available() else "cpu")
    (transformer, vae, text_encoder_1, text_encoder_2,
     tokenizer_1, tokenizer_2, scheduler) = loader.load_all()

    # 创建推理管道
    pipeline = InferencePipeline(
        transformer=transformer,
        vae=vae,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        scheduler=scheduler
    )

    # 注入LoRA
    pipeline.inject_dual_route_moe_lora(verbose=True)

    assert pipeline.lora_injected

    # 验证MoE层
    moe_count = 0
    for name, module in pipeline.transformer.named_modules():
        if hasattr(module, 'routing_mode') and module.routing_mode == 'dual':
            moe_count += 1

    assert moe_count == 4, f"期望4个MoE层，实际找到{moe_count}个"

    print(f"✓ LoRA注入成功，找到 {moe_count} 个MoE层")


def test_checkpoint_loading():
    """测试检查点加载"""
    print("\n" + "=" * 70)
    print("测试: 检查点加载")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 创建测试检查点
        checkpoint_path = create_test_checkpoint(temp_path)
        print(f"✓ 创建测试检查点: {checkpoint_path}")

        # 加载Mock FLUX
        loader = MockFluxModelLoader(device="cuda" if torch.cuda.is_available() else "cpu")
        (transformer, vae, text_encoder_1, text_encoder_2,
         tokenizer_1, tokenizer_2, scheduler) = loader.load_all()

        # 使用便捷函数创建管道
        pipeline = create_inference_pipeline_from_checkpoint(
            checkpoint_path=checkpoint_path,
            transformer=transformer,
            vae=vae,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
            verbose=True
        )

        assert pipeline.lora_injected
        assert pipeline.lora_loaded

        print("✓ 检查点加载成功")


def test_exposure_classification():
    """测试曝光度分类"""
    print("\n" + "=" * 70)
    print("测试: 曝光度分类")
    print("=" * 70)

    # 加载Mock FLUX
    loader = MockFluxModelLoader(device="cuda" if torch.cuda.is_available() else "cpu")
    (transformer, vae, text_encoder_1, text_encoder_2,
     tokenizer_1, tokenizer_2, scheduler) = loader.load_all()

    pipeline = InferencePipeline(
        transformer=transformer,
        vae=vae,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        scheduler=scheduler
    )

    # 创建不同亮度的测试图像
    test_cases = [
        (0.05, 0, "extremely_dark"),
        (0.15, 1, "very_dark"),
        (0.25, 2, "dark"),
        (0.4, 3, "moderate"),
        (0.7, 4, "normal")
    ]

    for brightness, expected_category, name in test_cases:
        # 创建图像
        img_array = np.ones((64, 64, 3), dtype=np.uint8) * int(brightness * 255)
        img = Image.fromarray(img_array)

        # 分类
        category = pipeline.classify_image_exposure(img)

        print(f"  亮度 {brightness:.2f} -> 类别 {category} ({name})")
        assert category == expected_category, f"期望类别{expected_category}，实际{category}"

    print("✓ 曝光度分类测试通过")


def test_single_image_inference():
    """测试单张图像推理"""
    print("\n" + "=" * 70)
    print("测试: 单张图像推理")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 创建测试图像
        img_array = np.ones((64, 64, 3), dtype=np.uint8) * 50  # 暗光图像
        test_image = Image.fromarray(img_array)
        test_image_path = temp_path / "test_input.png"
        test_image.save(test_image_path)

        # 创建测试检查点
        checkpoint_path = create_test_checkpoint(temp_path)

        # 加载Mock FLUX
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loader = MockFluxModelLoader(device=device)
        (transformer, vae, text_encoder_1, text_encoder_2,
         tokenizer_1, tokenizer_2, scheduler) = loader.load_all()

        # 创建管道
        pipeline = create_inference_pipeline_from_checkpoint(
            checkpoint_path=checkpoint_path,
            transformer=transformer,
            vae=vae,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
            device=device,
            verbose=False
        )

        # 运行推理
        print("\n运行推理...")
        enhanced = pipeline.enhance(
            image=test_image_path,
            prompt="Enhance",
            resolution=64,
            verbose=True
        )

        # 验证输出
        assert isinstance(enhanced, Image.Image)
        assert enhanced.size == (64, 64)

        print(f"✓ 单张图像推理成功")
        print(f"  输入大小: {test_image.size}")
        print(f"  输出大小: {enhanced.size}")


def test_batch_inference():
    """测试批量推理"""
    print("\n" + "=" * 70)
    print("测试: 批量推理")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 创建多张测试图像
        num_images = 6
        image_paths = []

        for i in range(num_images):
            brightness = 0.1 + i * 0.1
            img_array = np.ones((64, 64, 3), dtype=np.uint8) * int(brightness * 255)
            img = Image.fromarray(img_array)
            img_path = temp_path / f"test_{i}.png"
            img.save(img_path)
            image_paths.append(img_path)

        # 创建测试检查点
        checkpoint_path = create_test_checkpoint(temp_path)

        # 加载Mock FLUX
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loader = MockFluxModelLoader(device=device)
        (transformer, vae, text_encoder_1, text_encoder_2,
         tokenizer_1, tokenizer_2, scheduler) = loader.load_all()

        # 创建管道
        pipeline = create_inference_pipeline_from_checkpoint(
            checkpoint_path=checkpoint_path,
            transformer=transformer,
            vae=vae,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
            device=device,
            verbose=False
        )

        # 批量推理
        print(f"\n批量推理 {num_images} 张图像...")
        enhanced_images = pipeline.enhance_batch(
            images=image_paths,
            prompt="Enhance",
            resolution=64,
            batch_size=2,
            verbose=True
        )

        # 验证输出
        assert len(enhanced_images) == num_images
        assert all(isinstance(img, Image.Image) for img in enhanced_images)
        assert all(img.size == (64, 64) for img in enhanced_images)

        print(f"✓ 批量推理成功")
        print(f"  处理了 {len(enhanced_images)} 张图像")


def main():
    print("\n" + "=" * 70)
    print("开始推理管道测试")
    print("=" * 70)

    try:
        # 运行所有测试
        test_inference_pipeline_creation()
        test_lora_injection()
        test_checkpoint_loading()
        test_exposure_classification()
        test_single_image_inference()
        test_batch_inference()

        print("\n" + "=" * 70)
        print("✅ 所有测试通过！")
        print("=" * 70)

        print("\n总结:")
        print("  ✓ 推理管道创建正常")
        print("  ✓ LoRA注入正常")
        print("  ✓ 检查点加载正常")
        print("  ✓ 曝光度分类正确")
        print("  ✓ 单张图像推理成功")
        print("  ✓ 批量推理成功")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
