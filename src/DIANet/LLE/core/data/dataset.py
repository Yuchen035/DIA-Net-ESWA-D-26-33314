"""
数据集类实现

MVP版本：
- 简单的图像对数据集（输入图像 + 目标图像）
- 基础的图像变换（resize + normalize）
- 不包含latent缓存
- 不包含复杂数据增强

双路由MoE-LoRA支持：
- 自动曝光度分类（用于Style-Specific路由）
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from typing import Dict, Optional, List
import torchvision.transforms as transforms

# 导入曝光度分类工具
from core.utils.exposure import classify_exposure


class ImagePairDataset(Dataset):
    """
    图像对数据集 - 用于图像到图像的训练（如暗光增强）

    MVP版本：简单实现，不包含缓存和复杂增强
    """

    def __init__(
        self,
        input_image_dir: str,
        target_image_dir: str,
        prompt: str = "",
        resolution: int = 512,
        center_crop: bool = False,
        random_flip: bool = False,
        classify_exposure_category: bool = False,
        exposure_method: str = "mean_brightness",
        exposure_thresholds: Optional[List[float]] = None,
    ):
        """
        Args:
            input_image_dir: 输入图像目录（如暗光图像）
            target_image_dir: 目标图像目录（如正常曝光图像）
            prompt: 文本提示词
            resolution: 图像分辨率
            center_crop: 是否中心裁剪
            random_flip: 是否随机翻转
            classify_exposure_category: 是否启用自动曝光度分类（用于双路由MoE-LoRA）
            exposure_method: 曝光度分类方法 (mean_brightness/histogram/percentile)
            exposure_thresholds: 曝光度分类阈值，默认 [0.1, 0.2, 0.3, 0.5]
        """
        self.input_dir = Path(input_image_dir)
        self.target_dir = Path(target_image_dir)
        self.prompt = prompt
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip

        # 曝光度分类配置
        self.classify_exposure_category = classify_exposure_category
        self.exposure_method = exposure_method
        self.exposure_thresholds = exposure_thresholds or [0.1, 0.2, 0.3, 0.5]

        # 检查目录是否存在
        if not self.input_dir.exists():
            raise ValueError(f"输入图像目录不存在: {self.input_dir}")
        if not self.target_dir.exists():
            raise ValueError(f"目标图像目录不存在: {self.target_dir}")

        # 获取图像列表（支持png和jpg）
        self.image_paths = self._get_image_paths()

        if len(self.image_paths) == 0:
            raise ValueError(f"在 {self.input_dir} 中没有找到图像")

        # 验证所有输入图像都有对应的目标图像
        self._validate_image_pairs()

        print(f"[数据集] 加载了 {len(self.image_paths)} 对训练图像")
        print(f"[数据集] 输入目录: {self.input_dir}")
        print(f"[数据集] 目标目录: {self.target_dir}")
        print(f"[数据集] 分辨率: {self.resolution}x{self.resolution}")

        if self.classify_exposure_category:
            print(f"[数据集] 曝光度分类: 启用")
            print(f"[数据集]   - 方法: {self.exposure_method}")
            print(f"[数据集]   - 阈值: {self.exposure_thresholds}")
            print(f"[数据集]   - 类别数: {len(self.exposure_thresholds) + 1}")

        # 创建图像变换
        self.transform = self._create_transforms()

    def _get_image_paths(self) -> List[Path]:
        """获取所有图像路径"""
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_paths.extend(list(self.input_dir.glob(ext)))
        return sorted(image_paths)

    def _validate_image_pairs(self):
        """验证所有输入图像都有对应的目标图像"""
        missing_targets = []
        for input_path in self.image_paths:
            target_path = self.target_dir / input_path.name
            if not target_path.exists():
                missing_targets.append(input_path.name)

        if missing_targets:
            raise ValueError(
                f"以下输入图像没有对应的目标图像: {', '.join(missing_targets[:5])}"
                + (f" (还有 {len(missing_targets) - 5} 个...)" if len(missing_targets) > 5 else "")
            )

    def _create_transforms(self) -> transforms.Compose:
        """创建图像变换"""
        transform_list = []

        # Resize
        if self.center_crop:
            transform_list.append(transforms.Resize(self.resolution))
            transform_list.append(transforms.CenterCrop(self.resolution))
        else:
            transform_list.append(transforms.Resize((self.resolution, self.resolution)))

        # 随机翻转
        if self.random_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        # 转换为Tensor并归一化到[-1, 1]
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

        return transforms.Compose(transform_list)

    def __len__(self) -> int:
        """数据集大小"""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一对训练数据

        Returns:
            字典包含:
            - input_image: 输入图像张量 [3, H, W]
            - target_image: 目标图像张量 [3, H, W]
            - prompt: 文本提示词
            - exposure_category: 曝光度类别 (如果启用分类) [int]
        """
        # 加载输入图像
        input_path = self.image_paths[idx]
        input_image = Image.open(input_path).convert("RGB")

        # 加载目标图像
        target_path = self.target_dir / input_path.name
        target_image = Image.open(target_path).convert("RGB")

        # 曝光度分类（在应用变换之前，使用原始图像）
        exposure_category = None
        if self.classify_exposure_category:
            exposure_category = classify_exposure(
                input_image,
                thresholds=self.exposure_thresholds,
                method=self.exposure_method
            )

        # 应用变换
        # 注意：如果启用random_flip，需要对输入和目标应用相同的翻转
        if self.random_flip:
            # 使用相同的随机种子确保一致性
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            input_tensor = self.transform(input_image)
            torch.manual_seed(seed)
            target_tensor = self.transform(target_image)
        else:
            input_tensor = self.transform(input_image)
            target_tensor = self.transform(target_image)

        result = {
            "input_image": input_tensor,
            "target_image": target_tensor,
            "prompt": self.prompt,
        }

        # 添加曝光度类别（如果启用）
        if exposure_category is not None:
            result["exposure_category"] = exposure_category

        return result


def create_dataset_from_config(config) -> ImagePairDataset:
    """
    从TrainingConfig创建数据集

    Args:
        config: TrainingConfig对象

    Returns:
        ImagePairDataset实例
    """
    # 基础参数
    dataset_kwargs = {
        "input_image_dir": config.data.input_image_dir,
        "target_image_dir": config.data.target_image_dir,
        "prompt": config.data.prompt,
        "resolution": config.data.resolution,
        "center_crop": config.data.center_crop,
        "random_flip": config.data.random_flip,
    }

    # 曝光度分类参数（如果配置中有的话）
    if hasattr(config.data, 'classify_exposure_category'):
        dataset_kwargs["classify_exposure_category"] = config.data.classify_exposure_category

    if hasattr(config.data, 'exposure_method'):
        dataset_kwargs["exposure_method"] = config.data.exposure_method

    if hasattr(config.data, 'exposure_thresholds'):
        dataset_kwargs["exposure_thresholds"] = config.data.exposure_thresholds

    return ImagePairDataset(**dataset_kwargs)
