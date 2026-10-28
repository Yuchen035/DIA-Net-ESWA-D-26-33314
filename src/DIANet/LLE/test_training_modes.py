#!/usr/bin/env python3
"""
测试脚本 - 验证三种训练模式
使用随机噪声数据集测试：
1. 普通LoRA模式（rank=1）
2. 标准MoE LoRA模式（rank=1）
3. 双路由MoE LoRA模式（rank=1）

使用方法:
    python test_training_modes.py --mode all        # 测试所有模式
    python test_training_modes.py --mode lora       # 只测试普通LoRA
    python test_training_modes.py --mode moe_single # 只测试标准MoE
    python test_training_modes.py --mode moe_dual   # 只测试双路由MoE
"""

import os
import sys
import argparse
import shutil
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import torch

# 添加项目根目录到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def create_random_noise_dataset(output_dir: str, num_samples: int = 5):
    """
    创建随机噪声数据集用于测试

    Args:
        output_dir: 输出目录
        num_samples: 生成样本数量
    """
    print(f"\n{'='*70}")
    print("创建随机噪声测试数据集")
    print(f"{'='*70}")

    input_dir = Path(output_dir) / "input"
    target_dir = Path(output_dir) / "target"

    # 创建目录
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 生成随机图像
    for i in range(num_samples):
        # 输入图像：暗光（亮度较低的随机噪声）
        input_img = np.random.rand(512, 512, 3) * 0.3  # 0-0.3范围的暗光图像
        input_img = (input_img * 255).astype(np.uint8)
        Image.fromarray(input_img).save(input_dir / f"input_{i:03d}.png")

        # 目标图像：增强后（亮度较高的随机噪声）
        target_img = np.random.rand(512, 512, 3) * 0.7 + 0.3  # 0.3-1.0范围的明亮图像
        target_img = (target_img * 255).astype(np.uint8)
        Image.fromarray(target_img).save(target_dir / f"target_{i:03d}.png")

    print(f"✓ 已生成 {num_samples} 对随机噪声图像")
    print(f"  - 输入目录: {input_dir}")
    print(f"  - 目标目录: {target_dir}")
    print(f"{'='*70}\n")


def run_training(config_path: str, mode_name: str):
    """
    运行训练脚本

    Args:
        config_path: 配置文件路径
        mode_name: 模式名称（用于显示）

    Returns:
        bool: 训练是否成功
    """
    print(f"\n{'='*70}")
    print(f"测试模式: {mode_name}")
    print(f"配置文件: {config_path}")
    print(f"{'='*70}\n")

    # 构建命令
    cmd = [
        "python", "train_lora.py",
        "--config", config_path,
        "--verbose"
    ]

    # 运行训练
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            check=True,
            capture_output=False,
            text=True
        )

        print(f"\n{'='*70}")
        print(f"✅ {mode_name} 测试通过！")
        print(f"{'='*70}\n")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n{'='*70}")
        print(f"❌ {mode_name} 测试失败！")
        print(f"错误信息: {e}")
        print(f"{'='*70}\n")
        return False

    except KeyboardInterrupt:
        print(f"\n\n⚠️  {mode_name} 测试被用户中断")
        return False


def cleanup_test_data(data_dir: str):
    """清理测试数据"""
    if os.path.exists(data_dir):
        print(f"清理测试数据: {data_dir}")
        shutil.rmtree(data_dir)


def main():
    parser = argparse.ArgumentParser(description="测试三种训练模式")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "lora", "moe_single", "moe_dual"],
        default="all",
        help="测试模式"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="生成的测试样本数量"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="测试完成后清理测试数据"
    )

    args = parser.parse_args()

    # 定义测试配置
    test_configs = {
        "lora": {
            "name": "普通LoRA (rank=1)",
            "config": "configs/test_lora_rank1.yaml"
        },
        "moe_single": {
            "name": "标准MoE LoRA (rank=1)",
            "config": "configs/test_moe_single_rank1.yaml"
        },
        "moe_dual": {
            "name": "双路由MoE LoRA (rank=1)",
            "config": "configs/test_moe_dual_rank1.yaml"
        }
    }

    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法运行测试")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("MoE-LoRA训练模式测试")
    print(f"{'='*70}")
    print(f"测试模式: {args.mode}")
    print(f"测试样本数: {args.num_samples}")
    print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}\n")

    # 创建测试数据集
    data_dir = Path(__file__).parent / "data" / "test_random"
    create_random_noise_dataset(str(data_dir), args.num_samples)

    # 确定要测试的模式
    if args.mode == "all":
        modes_to_test = ["lora", "moe_single", "moe_dual"]
    else:
        modes_to_test = [args.mode]

    # 运行测试
    results = {}
    for mode in modes_to_test:
        config_info = test_configs[mode]
        config_path = Path(__file__).parent / config_info["config"]

        if not config_path.exists():
            print(f"❌ 配置文件不存在: {config_path}")
            results[mode] = False
            continue

        success = run_training(str(config_path), config_info["name"])
        results[mode] = success

        # 如果测试失败，询问是否继续
        if not success and len(modes_to_test) > 1:
            response = input("\n是否继续测试下一个模式? (y/n): ")
            if response.lower() != 'y':
                break

    # 打印测试总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    for mode, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_configs[mode]['name']}: {status}")
    print("="*70)

    # 清理测试数据
    if args.cleanup:
        print("\n清理测试数据...")
        cleanup_test_data(str(data_dir))
        for mode in modes_to_test:
            output_dir = Path(__file__).parent / "outputs" / f"test_{mode}_rank1"
            if output_dir.exists():
                cleanup_test_data(str(output_dir))
        print("✓ 清理完成")

    # 返回退出码
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
