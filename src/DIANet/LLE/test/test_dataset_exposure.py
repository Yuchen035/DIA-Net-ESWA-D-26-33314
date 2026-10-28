"""
测试数据集曝光度分类功能
"""

import torch
import numpy as np
import sys
from pathlib import Path
from PIL import Image
import tempfile
import shutil

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data.dataset import ImagePairDataset
from core.utils.exposure import get_category_name


def create_test_images(temp_dir: Path, num_images: int = 8):
    """创建测试图像（不同亮度）"""
    input_dir = temp_dir / "input"
    target_dir = temp_dir / "target"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 创建不同亮度的测试图像
    brightness_levels = [0.05, 0.15, 0.25, 0.35, 0.45, 0.6, 0.75, 0.9]

    for i in range(num_images):
        brightness = brightness_levels[i % len(brightness_levels)]

        # 输入图像（暗光）
        input_img = np.ones((128, 128, 3), dtype=np.uint8) * int(brightness * 255)
        input_pil = Image.fromarray(input_img)
        input_pil.save(input_dir / f"img_{i:03d}.png")

        # 目标图像（正常亮度）
        target_img = np.ones((128, 128, 3), dtype=np.uint8) * 200
        target_pil = Image.fromarray(target_img)
        target_pil.save(target_dir / f"img_{i:03d}.png")

    return input_dir, target_dir


def test_dataset_without_exposure():
    """测试1: 不启用曝光度分类（向后兼容）"""
    print("\n" + "=" * 70)
    print("测试1: 数据集不启用曝光度分类（向后兼容）")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir, target_dir = create_test_images(temp_path, num_images=4)

        # 创建数据集（不启用曝光度分类）
        dataset = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            prompt="Enhance the low-light image",
            resolution=64,
            classify_exposure_category=False
        )

        print(f"✓ 数据集创建成功，共 {len(dataset)} 个样本")

        # 获取一个样本
        sample = dataset[0]

        print(f"\n样本内容:")
        print(f"  input_image shape: {sample['input_image'].shape}")
        print(f"  target_image shape: {sample['target_image'].shape}")
        print(f"  prompt: {sample['prompt']}")
        print(f"  exposure_category 存在: {'exposure_category' in sample}")

        # 验证
        assert 'input_image' in sample
        assert 'target_image' in sample
        assert 'prompt' in sample
        assert 'exposure_category' not in sample, "不应该包含exposure_category"

        print("\n✅ 向后兼容性测试通过")


def test_dataset_with_exposure():
    """测试2: 启用曝光度分类"""
    print("\n" + "=" * 70)
    print("测试2: 数据集启用曝光度分类")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir, target_dir = create_test_images(temp_path, num_images=8)

        # 创建数据集（启用曝光度分类）
        dataset = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            prompt="Enhance the low-light image",
            resolution=64,
            classify_exposure_category=True,
            exposure_method="mean_brightness",
            exposure_thresholds=[0.1, 0.2, 0.3, 0.5]
        )

        print(f"✓ 数据集创建成功，共 {len(dataset)} 个样本")

        # 收集所有样本的曝光类别
        categories = []
        for i in range(len(dataset)):
            sample = dataset[i]
            category = sample['exposure_category']
            categories.append(category)

        print(f"\n所有样本的曝光类别:")
        for i, cat in enumerate(categories):
            print(f"  样本{i}: 类别 {cat} ({get_category_name(cat)})")

        # 验证第一个样本
        sample0 = dataset[0]
        print(f"\n第一个样本详情:")
        print(f"  input_image shape: {sample0['input_image'].shape}")
        print(f"  target_image shape: {sample0['target_image'].shape}")
        print(f"  prompt: {sample0['prompt']}")
        print(f"  exposure_category: {sample0['exposure_category']} ({get_category_name(sample0['exposure_category'])})")

        # 验证
        assert 'input_image' in sample0
        assert 'target_image' in sample0
        assert 'prompt' in sample0
        assert 'exposure_category' in sample0, "应该包含exposure_category"
        assert isinstance(sample0['exposure_category'], int), "exposure_category应该是int"
        assert 0 <= sample0['exposure_category'] <= 4, "曝光类别应该在0-4之间"

        print("\n✅ 曝光度分类测试通过")


def test_batch_loading():
    """测试3: 批量加载（与DataLoader配合）"""
    print("\n" + "=" * 70)
    print("测试3: 批量加载（与DataLoader配合）")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir, target_dir = create_test_images(temp_path, num_images=8)

        # 创建数据集
        dataset = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            prompt="Enhance the low-light image",
            resolution=64,
            classify_exposure_category=True,
        )

        # 创建DataLoader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        print(f"✓ DataLoader创建成功，batch_size=4")

        # 获取一个batch
        batch = next(iter(dataloader))

        print(f"\nBatch内容:")
        print(f"  input_image shape: {batch['input_image'].shape}")
        print(f"  target_image shape: {batch['target_image'].shape}")
        print(f"  prompt 数量: {len(batch['prompt'])}")
        print(f"  exposure_category: {batch['exposure_category']}")
        print(f"  exposure_category shape: {batch['exposure_category'].shape}")

        # 验证
        assert batch['input_image'].shape == (4, 3, 64, 64)
        assert batch['target_image'].shape == (4, 3, 64, 64)
        assert len(batch['prompt']) == 4
        assert batch['exposure_category'].shape == (4,)
        assert batch['exposure_category'].dtype == torch.long or batch['exposure_category'].dtype == torch.int64

        print("\n✅ 批量加载测试通过")


def test_different_methods():
    """测试4: 不同曝光度分类方法"""
    print("\n" + "=" * 70)
    print("测试4: 不同曝光度分类方法")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir, target_dir = create_test_images(temp_path, num_images=4)

        methods = ["mean_brightness", "histogram", "percentile"]

        for method in methods:
            print(f"\n使用方法: {method}")

            dataset = ImagePairDataset(
                input_image_dir=str(input_dir),
                target_image_dir=str(target_dir),
                prompt="Enhance",
                resolution=64,
                classify_exposure_category=True,
                exposure_method=method
            )

            # 获取分类结果
            categories = [dataset[i]['exposure_category'] for i in range(len(dataset))]
            print(f"  分类结果: {categories}")

            # 验证所有类别都在有效范围内
            assert all(0 <= cat <= 4 for cat in categories), f"方法 {method} 产生了无效类别"

        print("\n✅ 不同方法测试通过")


def test_custom_thresholds():
    """测试5: 自定义曝光度阈值"""
    print("\n" + "=" * 70)
    print("测试5: 自定义曝光度阈值")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir, target_dir = create_test_images(temp_path, num_images=4)

        # 使用自定义阈值（3类）
        custom_thresholds = [0.3, 0.6]

        dataset = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            prompt="Enhance",
            resolution=64,
            classify_exposure_category=True,
            exposure_thresholds=custom_thresholds
        )

        print(f"✓ 使用自定义阈值: {custom_thresholds}")
        print(f"  类别数: {len(custom_thresholds) + 1}")

        # 获取分类结果
        categories = [dataset[i]['exposure_category'] for i in range(len(dataset))]
        print(f"  分类结果: {categories}")

        # 验证类别范围
        max_category = len(custom_thresholds)
        assert all(0 <= cat <= max_category for cat in categories), \
            f"类别应该在 [0, {max_category}] 范围内"

        print("\n✅ 自定义阈值测试通过")


def main():
    print("\n" + "=" * 70)
    print("开始测试数据集曝光度分类功能")
    print("=" * 70)

    try:
        test_dataset_without_exposure()
        test_dataset_with_exposure()
        test_batch_loading()
        test_different_methods()
        test_custom_thresholds()

        print("\n" + "=" * 70)
        print("✅ 所有测试通过！")
        print("=" * 70)

        print("\n总结:")
        print("  ✓ 向后兼容性（不启用曝光度分类）正常")
        print("  ✓ 曝光度分类功能正常工作")
        print("  ✓ DataLoader批量加载正常")
        print("  ✓ 三种分类方法都正常工作")
        print("  ✓ 自定义阈值功能正常")
        print("  ✓ exposure_category正确返回int类型")
        print("==" * 70)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
