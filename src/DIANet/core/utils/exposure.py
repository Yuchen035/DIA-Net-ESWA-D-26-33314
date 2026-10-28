"""
曝光度分类工具

用于自动分类图像的曝光程度，支持双路由MoE-LoRA中的Style-Specific路由
"""

import torch
import numpy as np
from typing import Union, List, Tuple
from PIL import Image


def classify_exposure(
    image: Union[np.ndarray, torch.Tensor, Image.Image],
    thresholds: List[float] = [0.1, 0.2, 0.3, 0.5],
    method: str = "mean_brightness"
) -> int:
    """
    根据图像亮度自动分类曝光程度

    Args:
        image: 输入图像，支持多种格式：
            - np.ndarray: [H, W, 3], 范围[0, 1]或[0, 255]
            - torch.Tensor: [C, H, W]或[H, W, C], 范围[0, 1]
            - PIL.Image: RGB图像
        thresholds: 亮度阈值列表，如 [0.1, 0.2, 0.3, 0.5]
            默认分为5类：[0-0.1), [0.1-0.2), [0.2-0.3), [0.3-0.5), [0.5-1.0]
        method: 分类方法
            - "mean_brightness": 平均亮度（简单快速）
            - "histogram": 直方图分析（权重偏向暗部）
            - "percentile": 百分位数法（鲁棒性好）

    Returns:
        category: 曝光类别
            0: extremely_dark (极暗，亮度 < 0.1)
            1: very_dark (很暗，0.1 ~ 0.2)
            2: dark (暗，0.2 ~ 0.3)
            3: moderate (中等，0.3 ~ 0.5)
            4: normal (正常，>= 0.5)

    Examples:
        >>> img = Image.open("dark_image.jpg")
        >>> category = classify_exposure(img)
        >>> print(f"曝光类别: {category}")  # 输出: 曝光类别: 0
    """
    # 统一转换为numpy数组，范围[0, 1]
    if isinstance(image, Image.Image):
        image = np.array(image).astype(np.float32) / 255.0
    elif isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        # 如果是[C, H, W]格式，转换为[H, W, C]
        if image.shape[0] == 3 or image.shape[0] == 1:
            image = np.transpose(image, (1, 2, 0))
        # 确保范围在[0, 1]
        if image.max() > 1.0:
            image = image / 255.0
    elif isinstance(image, np.ndarray):
        image = image.astype(np.float32)
        # 如果范围是[0, 255]，归一化到[0, 1]
        if image.max() > 1.0:
            image = image / 255.0
    else:
        raise TypeError(f"不支持的图像类型: {type(image)}")

    # 计算亮度值
    if method == "mean_brightness":
        # 方法1: 简单平均亮度
        mean_brightness = float(image.mean())

    elif method == "histogram":
        # 方法2: 直方图分析（权重偏向暗部，对暗光图像更敏感）
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 1))
        # 暗部权重更大（指数衰减）
        weights = np.exp(-np.linspace(0, 1, 256) * 2)
        bin_centers = np.linspace(0, 1, 256)
        mean_brightness = float(np.average(bin_centers, weights=hist * weights))

    elif method == "percentile":
        # 方法3: 百分位数法（鲁棒性好，不受极值影响）
        # 使用25th百分位数（较暗部分的代表）
        p25 = float(np.percentile(image.flatten(), 25))
        # 使用中位数
        p50 = float(np.percentile(image.flatten(), 50))
        # 加权平均（更关注暗部）
        mean_brightness = 0.6 * p25 + 0.4 * p50

    else:
        raise ValueError(f"未知的分类方法: {method}. 支持: mean_brightness, histogram, percentile")

    # 根据阈值分类
    for i, threshold in enumerate(thresholds):
        if mean_brightness < threshold:
            return i

    return len(thresholds)  # 最亮类别


def batch_classify_exposure(
    images: Union[torch.Tensor, List[np.ndarray]],
    thresholds: List[float] = [0.1, 0.2, 0.3, 0.5],
    method: str = "mean_brightness"
) -> Union[torch.Tensor, List[int]]:
    """
    批量分类图像曝光程度

    Args:
        images: 批量图像
            - torch.Tensor: [B, C, H, W], 范围[0, 1]
            - List[np.ndarray]: 列表，每个元素为[H, W, 3]
        thresholds: 亮度阈值列表
        method: 分类方法

    Returns:
        categories: 曝光类别
            - torch.Tensor: [B], dtype=torch.long (如果输入是tensor)
            - List[int]: 列表 (如果输入是list)

    Examples:
        >>> imgs = torch.randn(4, 3, 256, 256)  # 4张图像
        >>> categories = batch_classify_exposure(imgs)
        >>> print(categories)  # tensor([0, 1, 2, 1])
    """
    if isinstance(images, torch.Tensor):
        # Tensor输入
        batch_size = images.size(0)
        categories = []

        for i in range(batch_size):
            category = classify_exposure(images[i], thresholds, method)
            categories.append(category)

        return torch.tensor(categories, dtype=torch.long, device=images.device)

    elif isinstance(images, list):
        # List输入
        categories = []
        for img in images:
            category = classify_exposure(img, thresholds, method)
            categories.append(category)
        return categories

    else:
        raise TypeError(f"不支持的输入类型: {type(images)}")


def get_exposure_stats(
    image: Union[np.ndarray, torch.Tensor, Image.Image]
) -> dict:
    """
    获取图像的详细曝光统计信息

    Args:
        image: 输入图像

    Returns:
        stats: 统计信息字典，包含：
            - mean: 平均亮度
            - std: 标准差
            - min: 最小值
            - max: 最大值
            - percentiles: 各百分位数 [10, 25, 50, 75, 90]
            - histogram: 亮度直方图 (256 bins)

    Examples:
        >>> img = Image.open("image.jpg")
        >>> stats = get_exposure_stats(img)
        >>> print(f"平均亮度: {stats['mean']:.3f}")
        >>> print(f"中位数: {stats['percentiles'][2]:.3f}")
    """
    # 统一转换为numpy数组
    if isinstance(image, Image.Image):
        image = np.array(image).astype(np.float32) / 255.0
    elif isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.shape[0] == 3 or image.shape[0] == 1:
            image = np.transpose(image, (1, 2, 0))
        if image.max() > 1.0:
            image = image / 255.0
    elif isinstance(image, np.ndarray):
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

    # 展平为1D
    pixels = image.flatten()

    # 计算统计量
    stats = {
        'mean': float(pixels.mean()),
        'std': float(pixels.std()),
        'min': float(pixels.min()),
        'max': float(pixels.max()),
        'percentiles': [float(np.percentile(pixels, p)) for p in [10, 25, 50, 75, 90]],
        'histogram': np.histogram(pixels, bins=256, range=(0, 1))[0].tolist()
    }

    return stats


# 预定义的曝光类别名称映射
EXPOSURE_CATEGORIES = {
    0: "extremely_dark",  # 极暗
    1: "very_dark",       # 很暗
    2: "dark",            # 暗
    3: "moderate",        # 中等
    4: "normal"           # 正常
}


def get_category_name(category: int) -> str:
    """
    获取曝光类别的名称

    Args:
        category: 类别索引

    Returns:
        name: 类别名称

    Examples:
        >>> name = get_category_name(0)
        >>> print(name)  # "extremely_dark"
    """
    return EXPOSURE_CATEGORIES.get(category, "unknown")


def analyze_dataset_exposure(
    image_paths: List[str],
    thresholds: List[float] = [0.1, 0.2, 0.3, 0.5],
    method: str = "mean_brightness",
    verbose: bool = True
) -> dict:
    """
    分析整个数据集的曝光分布

    Args:
        image_paths: 图像路径列表
        thresholds: 亮度阈值
        method: 分类方法
        verbose: 是否打印详细信息

    Returns:
        analysis: 分析结果，包含：
            - categories: 每张图像的类别列表
            - distribution: 各类别的数量统计
            - percentages: 各类别的百分比

    Examples:
        >>> paths = ["img1.jpg", "img2.jpg", ...]
        >>> analysis = analyze_dataset_exposure(paths)
        >>> print(analysis['distribution'])  # {0: 10, 1: 20, 2: 15, ...}
    """
    from PIL import Image

    categories = []
    category_counts = {i: 0 for i in range(len(thresholds) + 1)}

    if verbose:
        from tqdm import tqdm
        iterator = tqdm(image_paths, desc="分析曝光度")
    else:
        iterator = image_paths

    for img_path in iterator:
        try:
            img = Image.open(img_path).convert('RGB')
            category = classify_exposure(img, thresholds, method)
            categories.append(category)
            category_counts[category] += 1
        except Exception as e:
            if verbose:
                print(f"警告: 无法处理图像 {img_path}: {e}")
            continue

    total = len(categories)
    percentages = {cat: (count / total * 100) if total > 0 else 0
                   for cat, count in category_counts.items()}

    if verbose:
        print("\n" + "=" * 60)
        print("数据集曝光度分布统计")
        print("=" * 60)
        for category in sorted(category_counts.keys()):
            name = get_category_name(category)
            count = category_counts[category]
            pct = percentages[category]
            print(f"  {name:20s} (类别{category}): {count:5d} 张 ({pct:5.1f}%)")
        print("=" * 60)
        print(f"  总计: {total} 张图像\n")

    return {
        'categories': categories,
        'distribution': category_counts,
        'percentages': percentages,
        'total': total
    }
