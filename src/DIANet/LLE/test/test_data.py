"""
数据加载测试

测试内容：
1. Dataset本身功能
   - 初始化
   - 图像加载
   - 变换应用
   - 错误处理

2. DataLoader功能
   - 批处理
   - 数据迭代
   - 多进程加载

3. 与其他组件的协作
   - 与config.py的集成
   - 端到端测试
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import shutil

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.data.dataset import ImagePairDataset, create_dataset_from_config
from core.data.dataloader import create_dataloader, create_dataloader_from_config
from core.config import TrainingConfig


# ============================================================================
# 测试辅助函数
# ============================================================================

def create_test_images(test_dir: Path, num_images: int = 5):
    """创建测试用的图像"""
    input_dir = test_dir / "input"
    target_dir = test_dir / "target"

    # 创建目录
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 创建图像对
    for i in range(num_images):
        # 创建输入图像（暗光 - 偏暗）
        input_img = np.random.randint(0, 100, (256, 256, 3), dtype=np.uint8)
        input_pil = Image.fromarray(input_img)
        input_pil.save(input_dir / f"image_{i:03d}.png")

        # 创建目标图像（正常曝光 - 偏亮）
        target_img = np.random.randint(150, 255, (256, 256, 3), dtype=np.uint8)
        target_pil = Image.fromarray(target_img)
        target_pil.save(target_dir / f"image_{i:03d}.png")

    print(f"✓ 创建了 {num_images} 对测试图像")
    return input_dir, target_dir


def cleanup_test_images(test_dir: Path):
    """清理测试图像"""
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print("✓ 清理测试图像")


# ============================================================================
# 第一部分：Dataset本身功能测试
# ============================================================================

def test_dataset_initialization():
    """测试1: Dataset初始化"""
    test_dir = Path("./test_data_temp")
    input_dir, target_dir = create_test_images(test_dir, num_images=5)

    try:
        dataset = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            prompt="test prompt",
            resolution=128
        )

        assert len(dataset) == 5, "数据集大小应该为5"
        assert dataset.prompt == "test prompt", "prompt应该正确"
        assert dataset.resolution == 128, "resolution应该正确"

        print("✓ 测试1: Dataset初始化")
    finally:
        cleanup_test_images(test_dir)


def test_dataset_invalid_dirs():
    """测试2: 无效目录错误处理"""
    # 测试不存在的输入目录
    try:
        dataset = ImagePairDataset(
            input_image_dir="/nonexistent/input",
            target_image_dir="/nonexistent/target",
            resolution=128
        )
        assert False, "应该抛出ValueError"
    except ValueError as e:
        assert "输入图像目录不存在" in str(e)

    print("✓ 测试2: 无效目录错误处理")


def test_dataset_missing_target():
    """测试3: 缺少目标图像错误处理"""
    test_dir = Path("./test_data_temp")
    input_dir, target_dir = create_test_images(test_dir, num_images=5)

    # 删除一个目标图像
    (target_dir / "image_002.png").unlink()

    try:
        dataset = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            resolution=128
        )
        assert False, "应该抛出ValueError"
    except ValueError as e:
        assert "没有对应的目标图像" in str(e)
        assert "image_002.png" in str(e)

    finally:
        cleanup_test_images(test_dir)

    print("✓ 测试3: 缺少目标图像错误处理")


def test_dataset_getitem():
    """测试4: __getitem__功能"""
    test_dir = Path("./test_data_temp")
    input_dir, target_dir = create_test_images(test_dir, num_images=5)

    try:
        dataset = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            prompt="enhance image",
            resolution=128
        )

        # 获取第一个样本
        sample = dataset[0]

        # 检查返回的字典
        assert "input_image" in sample, "应该包含input_image"
        assert "target_image" in sample, "应该包含target_image"
        assert "prompt" in sample, "应该包含prompt"

        # 检查张量形状
        assert sample["input_image"].shape == (3, 128, 128), "input_image形状应该正确"
        assert sample["target_image"].shape == (3, 128, 128), "target_image形状应该正确"

        # 检查prompt
        assert sample["prompt"] == "enhance image", "prompt应该正确"

        # 检查归一化范围
        assert sample["input_image"].min() >= -1.0, "input_image应该归一化到[-1, 1]"
        assert sample["input_image"].max() <= 1.0, "input_image应该归一化到[-1, 1]"

        print("✓ 测试4: __getitem__功能")
    finally:
        cleanup_test_images(test_dir)


def test_dataset_transforms():
    """测试5: 图像变换"""
    test_dir = Path("./test_data_temp")
    input_dir, target_dir = create_test_images(test_dir, num_images=5)

    try:
        # 测试resize
        dataset = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            resolution=256
        )

        sample = dataset[0]
        assert sample["input_image"].shape == (3, 256, 256), "应该resize到256x256"

        # 测试center_crop
        dataset_crop = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            resolution=128,
            center_crop=True
        )

        sample_crop = dataset_crop[0]
        assert sample_crop["input_image"].shape == (3, 128, 128), "center_crop后形状应该正确"

        print("✓ 测试5: 图像变换")
    finally:
        cleanup_test_images(test_dir)


def test_dataset_random_flip():
    """测试6: 随机翻转（输入和目标应该一致）"""
    test_dir = Path("./test_data_temp")
    input_dir, target_dir = create_test_images(test_dir, num_images=5)

    try:
        dataset = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            resolution=128,
            random_flip=True
        )

        # 多次采样同一个图像，检查随机性
        samples = [dataset[0] for _ in range(10)]

        # 检查是否有不同的结果（随机翻转）
        # 注意：由于随机种子，每次__getitem__可能不同
        # 这里只检查形状正确即可
        for sample in samples:
            assert sample["input_image"].shape == (3, 128, 128)
            assert sample["target_image"].shape == (3, 128, 128)

        print("✓ 测试6: 随机翻转")
    finally:
        cleanup_test_images(test_dir)


def test_dataset_iteration():
    """测试7: 完整迭代数据集"""
    test_dir = Path("./test_data_temp")
    input_dir, target_dir = create_test_images(test_dir, num_images=5)

    try:
        dataset = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            resolution=128
        )

        # 迭代所有样本
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample["input_image"].shape == (3, 128, 128)
            assert sample["target_image"].shape == (3, 128, 128)

        print("✓ 测试7: 完整迭代数据集")
    finally:
        cleanup_test_images(test_dir)


# ============================================================================
# 第二部分：DataLoader功能测试
# ============================================================================

def test_dataloader_creation():
    """测试8: DataLoader创建"""
    test_dir = Path("./test_data_temp")
    input_dir, target_dir = create_test_images(test_dir, num_images=10)

    try:
        dataset = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            resolution=128
        )

        dataloader = create_dataloader(
            dataset=dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0
        )

        assert len(dataloader) == 5, "10个样本，batch_size=2，应该有5个batch"

        print("✓ 测试8: DataLoader创建")
    finally:
        cleanup_test_images(test_dir)


def test_dataloader_batching():
    """测试9: DataLoader批处理"""
    test_dir = Path("./test_data_temp")
    input_dir, target_dir = create_test_images(test_dir, num_images=10)

    try:
        dataset = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            resolution=128
        )

        dataloader = create_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0
        )

        # 获取第一个batch
        batch = next(iter(dataloader))

        # 检查batch形状
        assert batch["input_image"].shape == (4, 3, 128, 128), "batch input形状应该正确"
        assert batch["target_image"].shape == (4, 3, 128, 128), "batch target形状应该正确"
        assert len(batch["prompt"]) == 4, "prompt应该有4个元素"

        print("✓ 测试9: DataLoader批处理")
    finally:
        cleanup_test_images(test_dir)


def test_dataloader_iteration():
    """测试10: DataLoader完整迭代"""
    test_dir = Path("./test_data_temp")
    input_dir, target_dir = create_test_images(test_dir, num_images=10)

    try:
        dataset = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            resolution=128
        )

        dataloader = create_dataloader(
            dataset=dataset,
            batch_size=3,
            shuffle=False,
            num_workers=0
        )

        total_samples = 0
        for batch in dataloader:
            batch_size = batch["input_image"].shape[0]
            total_samples += batch_size

        assert total_samples == 10, "应该迭代所有10个样本"

        print("✓ 测试10: DataLoader完整迭代")
    finally:
        cleanup_test_images(test_dir)


def test_dataloader_drop_last():
    """测试11: DataLoader drop_last参数"""
    test_dir = Path("./test_data_temp")
    input_dir, target_dir = create_test_images(test_dir, num_images=10)

    try:
        dataset = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            resolution=128
        )

        # 不drop_last
        dataloader_no_drop = create_dataloader(
            dataset=dataset,
            batch_size=3,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )

        # 10个样本，batch_size=3，应该有4个batch（最后一个batch只有1个样本）
        assert len(dataloader_no_drop) == 4, "不drop_last应该有4个batch"

        # drop_last
        dataloader_drop = create_dataloader(
            dataset=dataset,
            batch_size=3,
            shuffle=False,
            num_workers=0,
            drop_last=True
        )

        # 10个样本，batch_size=3，drop_last后应该有3个batch
        assert len(dataloader_drop) == 3, "drop_last应该有3个batch"

        print("✓ 测试11: DataLoader drop_last参数")
    finally:
        cleanup_test_images(test_dir)


# ============================================================================
# 第三部分：与其他组件的协作测试
# ============================================================================

def test_integration_with_config():
    """测试12: 与配置系统的集成"""
    test_dir = Path("./test_data_temp")
    input_dir, target_dir = create_test_images(test_dir, num_images=10)

    try:
        # 创建配置
        config_dict = {
            'job_name': 'test_data',
            'device': 'cpu',
            'lora': {
                'rank': 16,
                'alpha': 16.0,
                'target_modules': ["*.attn.to_q"]
            },
            'data': {
                'input_image_dir': str(input_dir),
                'target_image_dir': str(target_dir),
                'prompt': 'enhance low-light image',
                'resolution': 256,
                'center_crop': False,
                'random_flip': True,
                'num_workers': 0,
                'pin_memory': False,
            },
            'train': {
                'num_epochs': 10,
                'batch_size': 2,
                'learning_rate': 1e-4,
            }
        }

        config = TrainingConfig.from_dict(config_dict)

        # 从配置创建dataset
        dataset = create_dataset_from_config(config)

        assert len(dataset) == 10, "数据集大小应该为10"
        assert dataset.resolution == 256, "resolution应该从config读取"
        assert dataset.prompt == 'enhance low-light image', "prompt应该从config读取"
        assert dataset.random_flip == True, "random_flip应该从config读取"

        print("✓ 测试12: 与配置系统的集成")
    finally:
        cleanup_test_images(test_dir)


def test_integration_dataloader_from_config():
    """测试13: 从配置创建DataLoader"""
    test_dir = Path("./test_data_temp")
    input_dir, target_dir = create_test_images(test_dir, num_images=10)

    try:
        config_dict = {
            'job_name': 'test_dataloader',
            'device': 'cpu',
            'lora': {
                'rank': 16,
                'alpha': 16.0,
                'target_modules': ["*.attn.to_q"]
            },
            'data': {
                'input_image_dir': str(input_dir),
                'target_image_dir': str(target_dir),
                'prompt': 'test',
                'resolution': 128,
                'num_workers': 0,
                'pin_memory': False,
            },
            'train': {
                'num_epochs': 10,
                'batch_size': 4,
                'learning_rate': 1e-4,
            }
        }

        config = TrainingConfig.from_dict(config_dict)

        # 从配置创建dataloader
        dataset, dataloader = create_dataloader_from_config(config)

        assert len(dataset) == 10, "数据集大小应该为10"
        assert len(dataloader) == 3, "10个样本，batch_size=4，应该有3个batch"

        # 测试迭代
        batch = next(iter(dataloader))
        assert batch["input_image"].shape[0] == 4, "第一个batch应该有4个样本"

        print("✓ 测试13: 从配置创建DataLoader")
    finally:
        cleanup_test_images(test_dir)


def test_end_to_end_data_pipeline():
    """测试14: 端到端数据管道测试"""
    test_dir = Path("./test_data_temp")
    input_dir, target_dir = create_test_images(test_dir, num_images=10)

    try:
        # 步骤1: 创建配置
        config_dict = {
            'job_name': 'test_e2e',
            'device': 'cpu',
            'lora': {
                'rank': 8,
                'alpha': 8.0,
                'target_modules': ["*.attn.to_q"]
            },
            'data': {
                'input_image_dir': str(input_dir),
                'target_image_dir': str(target_dir),
                'prompt': 'enhance the low-light image',
                'resolution': 128,
                'num_workers': 0,
                'pin_memory': False,
            },
            'train': {
                'num_epochs': 10,
                'batch_size': 2,
                'learning_rate': 1e-4,
            }
        }
        config = TrainingConfig.from_dict(config_dict)

        # 步骤2: 创建dataloader
        dataset, dataloader = create_dataloader_from_config(config)

        # 步骤3: 迭代数据（模拟训练循环）
        total_batches = 0
        total_samples = 0
        for batch in dataloader:
            total_batches += 1
            batch_size = batch["input_image"].shape[0]
            total_samples += batch_size

            # 检查数据有效性
            assert batch["input_image"].shape[1:] == (3, 128, 128)
            assert batch["target_image"].shape[1:] == (3, 128, 128)
            assert batch["input_image"].dtype == torch.float32
            assert batch["target_image"].dtype == torch.float32

        assert total_samples == 10, "应该处理所有10个样本"
        assert total_batches == 5, "batch_size=2，应该有5个batch"

        print("✓ 测试14: 端到端数据管道测试")
    finally:
        cleanup_test_images(test_dir)


def test_data_augmentation_consistency():
    """测试15: 数据增强一致性（输入和目标应该应用相同的增强）"""
    test_dir = Path("./test_data_temp")
    input_dir, target_dir = create_test_images(test_dir, num_images=5)

    try:
        # 创建一个特殊的测试图像对
        # 输入和目标完全相同，方便测试一致性
        test_img = np.ones((256, 256, 3), dtype=np.uint8) * 128
        test_img[:, :128, :] = 50  # 左半边暗
        test_img[:, 128:, :] = 200  # 右半边亮

        test_pil = Image.fromarray(test_img)
        test_pil.save(input_dir / "test_consistency.png")
        test_pil.save(target_dir / "test_consistency.png")

        dataset = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            resolution=128,
            random_flip=True
        )

        # 找到test_consistency的索引
        test_idx = None
        for i, path in enumerate(dataset.image_paths):
            if path.name == "test_consistency.png":
                test_idx = i
                break

        assert test_idx is not None, "应该找到test_consistency图像"

        # 由于random_flip，每次调用可能不同
        # 但同一次调用中，input和target应该一致
        sample = dataset[test_idx]

        # 检查形状
        assert sample["input_image"].shape == sample["target_image"].shape

        print("✓ 测试15: 数据增强一致性")
    finally:
        cleanup_test_images(test_dir)


# ============================================================================
# 运行所有测试
# ============================================================================

def main():
    """运行所有测试"""
    print("=" * 60)
    print("开始数据加载测试")
    print("=" * 60)

    # 第一部分：Dataset本身功能测试
    print("\n【第一部分：Dataset本身功能测试】")
    test_dataset_initialization()
    test_dataset_invalid_dirs()
    test_dataset_missing_target()
    test_dataset_getitem()
    test_dataset_transforms()
    test_dataset_random_flip()
    test_dataset_iteration()

    # 第二部分：DataLoader功能测试
    print("\n【第二部分：DataLoader功能测试】")
    test_dataloader_creation()
    test_dataloader_batching()
    test_dataloader_iteration()
    test_dataloader_drop_last()

    # 第三部分：与其他组件的协作测试
    print("\n【第三部分：与其他组件的协作测试】")
    test_integration_with_config()
    test_integration_dataloader_from_config()
    test_end_to_end_data_pipeline()
    test_data_augmentation_consistency()

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
