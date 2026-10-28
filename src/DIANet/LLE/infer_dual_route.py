"""
简单的双路由MoE-LoRA推理脚本

使用训练好的模型增强低光图像
"""

import torch
import argparse
from pathlib import Path
from PIL import Image
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.mock_flux import MockFluxModelLoader
from core.inference import create_inference_pipeline_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="双路由MoE-LoRA图像增强推理")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="训练好的检查点文件路径"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入图像路径或目录"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="输出目录（默认: ./output）"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Enhance the low-light image",
        help="文本提示词"
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="图像分辨率（默认: 256）"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="批处理大小（默认: 4）"
    )

    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="推理步数（默认: 50）"
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Guidance scale（默认: 3.5）"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备（默认: cuda if available）"
    )

    parser.add_argument(
        "--num-specific",
        type=int,
        default=5,
        help="Specific专家数量（默认: 5）"
    )

    parser.add_argument(
        "--num-shared",
        type=int,
        default=4,
        help="Shared专家数量（默认: 4）"
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="LoRA rank（默认: 8）"
    )

    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="使用Mock FLUX模型（用于测试）"
    )

    return parser.parse_args()


def get_image_paths(input_path: Path):
    """获取所有输入图像路径"""
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_paths.extend(list(input_path.glob(ext)))
        return sorted(image_paths)
    else:
        raise ValueError(f"输入路径不存在: {input_path}")


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("双路由MoE-LoRA图像增强推理")
    print("=" * 70)

    # 检查输入
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 错误: 输入路径不存在: {input_path}")
        return 1

    # 检查检查点
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ 错误: 检查点文件不存在: {checkpoint_path}")
        return 1

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图像路径
    image_paths = get_image_paths(input_path)
    if len(image_paths) == 0:
        print(f"❌ 错误: 在 {input_path} 中没有找到图像")
        return 1

    print(f"\n配置:")
    print(f"  输入: {input_path}")
    print(f"  检查点: {checkpoint_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  图像数量: {len(image_paths)}")
    print(f"  分辨率: {args.resolution}x{args.resolution}")
    print(f"  设备: {args.device}")
    print(f"  使用Mock FLUX: {args.use_mock}")

    # 1. 加载模型
    print("\n" + "=" * 70)
    print("加载模型")
    print("=" * 70)

    if args.use_mock:
        print("使用Mock FLUX模型...")
        loader = MockFluxModelLoader(device=args.device)
        (transformer, vae, text_encoder_1, text_encoder_2,
         tokenizer_1, tokenizer_2, scheduler) = loader.load_all()
    else:
        raise RuntimeError(
            "This archived dual-routing entry point is not connected to real FLUX. "
            "Use --use-mock only for the explicitly labelled prototype. "
            "No model is substituted automatically."
        )

    # 2. 创建推理管道
    print("\n" + "=" * 70)
    print("创建推理管道")
    print("=" * 70)

    pipeline = create_inference_pipeline_from_checkpoint(
        checkpoint_path=checkpoint_path,
        transformer=transformer,
        vae=vae,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        scheduler=scheduler,
        device=args.device,
        dtype=torch.bfloat16,
        num_specific=args.num_specific,
        num_shared=args.num_shared,
        rank=args.rank,
        verbose=True
    )

    # 3. 运行推理
    print("\n" + "=" * 70)
    print("运行推理")
    print("=" * 70)

    if len(image_paths) == 1:
        # 单张图像
        print(f"\n处理单张图像: {image_paths[0].name}")
        enhanced = pipeline.enhance(
            image=image_paths[0],
            prompt=args.prompt,
            resolution=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            verbose=True
        )

        # 保存结果
        output_path = output_dir / f"enhanced_{image_paths[0].name}"
        enhanced.save(output_path)
        print(f"\n✓ 保存到: {output_path}")

    else:
        # 批量处理
        print(f"\n批量处理 {len(image_paths)} 张图像...")
        enhanced_images = pipeline.enhance_batch(
            images=image_paths,
            prompt=args.prompt,
            resolution=args.resolution,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            verbose=True
        )

        # 保存结果
        print(f"\n保存结果到 {output_dir}...")
        for img_path, enhanced in zip(image_paths, enhanced_images):
            output_path = output_dir / f"enhanced_{img_path.name}"
            enhanced.save(output_path)

        print(f"✓ 所有图像已保存")

    # 4. 完成
    print("\n" + "=" * 70)
    print("✅ 推理完成！")
    print("=" * 70)
    print(f"\n增强后的图像保存在: {output_dir}")

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠ 推理被用户中断")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
