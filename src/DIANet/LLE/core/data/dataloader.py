"""
数据加载器实现

MVP版本：
- 基础DataLoader封装
- 从配置创建DataLoader
- 简单的collate函数
"""

from torch.utils.data import DataLoader
from typing import Tuple
from .dataset import ImagePairDataset, create_dataset_from_config


def create_dataloader(
    dataset: ImagePairDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """
    创建DataLoader

    Args:
        dataset: ImagePairDataset实例
        batch_size: 批大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作线程数
        pin_memory: 是否将数据固定在内存中（加速GPU传输）
        drop_last: 是否丢弃最后不完整的batch

    Returns:
        DataLoader实例
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader


def create_dataloader_from_config(config) -> Tuple[ImagePairDataset, DataLoader]:
    """
    从TrainingConfig创建数据集和DataLoader

    Args:
        config: TrainingConfig对象

    Returns:
        (dataset, dataloader)元组
    """
    # 创建数据集
    dataset = create_dataset_from_config(config)

    # 创建DataLoader
    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False,
    )

    print(f"[DataLoader] 批大小: {config.train.batch_size}")
    print(f"[DataLoader] 工作线程数: {config.data.num_workers}")
    print(f"[DataLoader] Pin memory: {config.data.pin_memory}")
    print(f"[DataLoader] 总批次数: {len(dataloader)}")

    return dataset, dataloader
