# -*- coding: utf-8 -*-
"""
FLUX Kontext 图生图推理代码
=============================

这是一个简单、清晰的 FLUX Kontext 推理实现，适合深度学习初学者学习。
代码结构分为以下几个模块：
1. 图像处理模块：负责加载和保存图像
2. 模型加载模块：负责加载 FLUX Kontext 模型
3. 推理执行模块：执行图生图推理
4. 主函数：整合所有功能
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union
import torch
from PIL import Image

# ============================================================================
# 第一部分：导入必要的库
# ============================================================================

# 添加本地 diffusers 库到系统路径
# 这样可以使用本地下载的 diffusers 库，而不是系统安装的版本
CURRENT_DIR = Path(__file__).parent
DIFFUSERS_PATH = "diffusers/src"
sys.path.insert(0, str(DIFFUSERS_PATH))

# 从 diffusers 库导入 FLUX Kontext 推理管道
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image


# ============================================================================
# 第二部分：图像处理模块
# ============================================================================

def load_input_image(image_path: str) -> Image.Image:
    """
    加载输入图像

    功能说明：
        从指定路径加载图像文件，并转换为 RGB 格式。
        FLUX Kontext 需要 RGB 格式的图像作为输入。

    参数：
        image_path (str): 输入图像的文件路径

    返回：
        Image.Image: PIL 图像对象（RGB 格式）

    异常：
        FileNotFoundError: 如果图像文件不存在
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到输入图像：{image_path}")

    print(f"[图像加载] 正在加载图像：{image_path}")

    # 使用 diffusers 提供的工具函数加载图像
    # 这个函数可以处理本地文件路径或 URL
    image = load_image(image_path)

    # 确保图像是 RGB 格式（有些图像可能是 RGBA 或灰度图）
    image = image.convert("RGB")

    print(f"[图像加载] 图像加载成功，尺寸：{image.size[0]}x{image.size[1]}")

    return image


def save_output_image(image: Image.Image, output_path: str) -> None:
    """
    保存输出图像

    功能说明：
        将生成的图像保存到指定路径。
        自动创建输出目录（如果不存在）。

    参数：
        image (Image.Image): 要保存的 PIL 图像对象
        output_path (str): 输出图像的保存路径
    """
    # 获取输出目录
    output_dir = os.path.dirname(output_path)

    # 如果输出目录不为空且不存在，则创建它
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[图像保存] 创建输出目录：{output_dir}")

    # 保存图像
    image.save(output_path)
    print(f"[图像保存] 图像已保存到：{output_path}")


# ============================================================================
# 第三部分：模型加载模块
# ============================================================================

def load_flux_kontext_pipeline(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    gpu_id: int = 0,
    use_group_offload: bool = True,
    num_blocks_per_group: int = 1
) -> FluxKontextPipeline:
    """
    加载 FLUX Kontext 推理管道

    功能说明：
        从本地路径加载 FLUX Kontext 模型的所有组件，包括：
        - Transformer 模型（主要的扩散模型）
        - VAE（变分自编码器，用于图像编码和解码）
        - 文本编码器（CLIP 和 T5，用于理解文本提示词）
        - 调度器（控制去噪过程）

        重要：使用混合 Offload 策略来优化显存使用（2025年最新方法）
        - Transformer: 使用 block_level group offloading（最大组件，约 20GB）
        - 其他组件: 使用传统 CPU offload（Text Encoders, VAE）
        - 真正实现显存优化：从 24GB 降到约 8-10GB

        混合策略优势：
        1. Transformer 使用 block_level 可以高效地分块加载/卸载
        2. 避免了 group offloading 在 VAE 等组件上的已知设备不匹配 bug
        3. 使用 CUDA stream 异步加载，提升推理速度

    参数：
        model_path (str): 模型权重文件的本地路径
        device (str): 运行设备，"cuda" 表示 GPU，"cpu" 表示 CPU
        dtype (torch.dtype): 模型精度，torch.bfloat16 可以节省显存
        gpu_id (int): GPU 设备 ID，默认为 0（如果有多张显卡可以指定）
        use_group_offload (bool): 是否使用混合 Offload 策略，默认 True（强烈推荐）
        num_blocks_per_group (int): Transformer 的块数配置，默认 1（建议 1-2）

    返回：
        FluxKontextPipeline: 加载完成的推理管道
    """
    print(f"[模型加载] 开始加载 FLUX Kontext 模型")
    print(f"[模型加载] 模型路径：{model_path}")
    print(f"[模型加载] 运行设备：{device}")
    print(f"[模型加载] 数据类型：{dtype}")
    print(f"[模型加载] GPU ID：{gpu_id}")
    print(f"[模型加载] Group Offload：{'启用' if use_group_offload else '禁用'}")
    if use_group_offload:
        print(f"[模型加载] Offload 类型：{offload_type}")

    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型路径：{model_path}")

    # 从本地路径加载预训练模型
    # from_pretrained 会自动加载 model_index.json 中定义的所有组件
    # 注意：这里不指定 device，模型会先在 CPU 上加载
    print(f"[模型加载] 正在从磁盘加载模型组件...")
    pipeline = FluxKontextPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,  # 使用 bfloat16 精度以节省显存
    )

    # 使用混合 Offload 策略来优化显存使用
    if device == "cuda" and use_group_offload:
        print(f"\n[模型加载] ========== 启用混合 Offload 策略 ==========")
        print(f"[模型加载] 策略说明：")
        print(f"[模型加载]   - Transformer: block_level（最大组件，约 20GB）")
        print(f"[模型加载]   - Text Encoders & VAE: leaf_level（避免设备不匹配）")
        print(f"[模型加载] ")

        from diffusers.hooks import apply_group_offloading

        onload_device = torch.device(f"cuda:{gpu_id}")
        offload_device = torch.device("cpu")

        # 步骤 1：对 Transformer 应用 block_level group offloading
        # Transformer 是最大的组件（约 20+GB），使用 block_level 可以显著节省显存
        print(f"[模型加载] 正在对 Transformer 应用 block_level group offloading...")
        apply_group_offloading(
            pipeline.transformer,
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type="block_level",
            num_blocks_per_group=num_blocks_per_group,
            use_stream=True,
            record_stream=False,
        )
        print(f"[模型加载] ✓ Transformer offloading 完成 (block_level, {num_blocks_per_group} blocks/group)")

        # 步骤 2：对其他组件应用 leaf_level group offloading
        # 关键：所有组件都必须使用 group offloading 以避免设备不匹配
        # 使用 leaf_level 避免 VAE 在 block_level 下的已知 bug

        print(f"[模型加载] 正在对 Text Encoder (CLIP) 应用 leaf_level offloading...")
        apply_group_offloading(
            pipeline.text_encoder,
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type="leaf_level",
            use_stream=True,
        )
        print(f"[模型加载] ✓ Text Encoder (CLIP) offloading 完成 (leaf_level)")

        print(f"[模型加载] 正在对 Text Encoder 2 (T5) 应用 leaf_level offloading...")
        apply_group_offloading(
            pipeline.text_encoder_2,
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type="leaf_level",
            use_stream=True,
        )
        print(f"[模型加载] ✓ Text Encoder 2 (T5) offloading 完成 (leaf_level)")

        print(f"[模型加载] 正在对 VAE 应用 leaf_level offloading...")
        apply_group_offloading(
            pipeline.vae,
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type="leaf_level",
            use_stream=True,
        )
        print(f"[模型加载] ✓ VAE offloading 完成 (leaf_level)")

        print(f"\n[模型加载] 混合 Offload 策略已启用，将使用 GPU {gpu_id}")
        print(f"[模型加载] 预计显存占用：8-10GB（混合 group offloading）")
        print(f"[模型加载] - Transformer: block_level (num_blocks={num_blocks_per_group})")
        print(f"[模型加载] - Text Encoders & VAE: leaf_level")
        print(f"[模型加载] ========================================================\n")

    elif device == "cuda":
        # 如果不使用 Group Offload，则将整个模型移动到 GPU
        # 警告：这需要约 24GB 显存！
        print(f"[模型加载] 警告：未启用 Group Offload，将尝试完整加载模型到 GPU")
        print(f"[模型加载] 这需要约 24GB 显存，可能会导致 OOM 错误！")
        pipeline = pipeline.to(device)
    else:
        # CPU 模式（速度会非常慢）
        print(f"[模型加载] 使用 CPU 模式（警告：速度会非常慢）")
        pipeline = pipeline.to(device)

    print(f"[模型加载] 模型加载完成！")

    return pipeline


# ============================================================================
# 第四部分：推理执行模块
# ============================================================================

def run_image_to_image(
    pipeline: FluxKontextPipeline,
    input_image: Image.Image,
    prompt: str,
    num_inference_steps: int = 28,
    guidance_scale: float = 2.5,
    seed: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Image.Image:
    """
    执行图生图推理

    功能说明：
        使用 FLUX Kontext 模型根据输入图像和文本提示词生成新图像。

        核心概念：
        1. 输入图像：作为生成的起点和参考
        2. 文本提示词：描述你想要的修改或风格
        3. 推理步数：去噪迭代的次数，越多质量越好但速度越慢
        4. 引导尺度：控制生成结果与提示词的贴合程度
        5. 随机种子：控制随机性，相同种子产生相同结果

    参数：
        pipeline (FluxKontextPipeline): 已加载的推理管道
        input_image (Image.Image): 输入的参考图像
        prompt (str): 文本提示词，描述想要的效果
        num_inference_steps (int): 推理步数，默认 28（推荐范围 20-50）
        guidance_scale (float): 引导尺度，默认 2.5（推荐范围 1.5-3.5）
        seed (int, optional): 随机种子，用于结果复现
        height (int, optional): 输出图像高度，None 表示自动调整
        width (int, optional): 输出图像宽度，None 表示自动调整

    返回：
        Image.Image: 生成的图像
    """
    print(f"\n{'='*60}")
    print(f"[推理执行] 开始图生图推理")
    print(f"{'='*60}")
    print(f"[推理参数] 文本提示词：{prompt}")
    print(f"[推理参数] 推理步数：{num_inference_steps}")
    print(f"[推理参数] 引导尺度：{guidance_scale}")
    print(f"[推理参数] 随机种子：{seed if seed is not None else '随机'}")

    # 如果指定了随机种子，创建随机数生成器
    # 这样可以确保每次运行产生相同的结果（结果可复现）
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        print(f"[推理执行] 已设置随机种子：{seed}")

    # 执行推理
    # 这是整个代码的核心部分，调用 FLUX Kontext 模型进行图像生成
    print(f"[推理执行] 正在生成图像，请稍候...")

    # 计算max_area（如果指定了height/width）
    # 这样可以确保输出分辨率与输入保持一致，避免被自动调整到1024x1024
    max_area_param = {}
    if height is not None and width is not None:
        max_area_param['max_area'] = height * width
        max_area_param['_auto_resize'] = False  # 禁用控制图像自动resize

    result = pipeline(
        image=input_image,              # 输入图像
        prompt=prompt,                   # 文本提示词
        num_inference_steps=num_inference_steps,  # 推理步数
        guidance_scale=guidance_scale,   # 引导尺度
        generator=generator,             # 随机数生成器
        height=height,                   # 输出高度
        width=width,                     # 输出宽度
        **max_area_param,               # 分辨率控制参数
    )

    # 从结果中提取生成的图像
    # result.images 是一个列表，包含所有生成的图像
    # 我们只取第一张（因为 num_images_per_prompt 默认为 1）
    output_image = result.images[0]

    print(f"[推理执行] 图像生成完成！")
    print(f"{'='*60}\n")

    return output_image


# ============================================================================
# 第五部分：主函数
# ============================================================================

def main():
    """
    主函数：整合所有功能，执行完整的推理流程

    流程说明：
        1. 配置参数（模型路径、输入图像、提示词等）
        2. 加载 FLUX Kontext 模型
        3. 加载输入图像
        4. 执行图生图推理
        5. 保存生成的图像
    """

    print("\n" + "="*60)
    print("FLUX Kontext 图生图推理程序")
    print("="*60 + "\n")

    # ========================================================================
    # 配置部分：在这里修改你的参数
    # ========================================================================

    # 模型路径（本地下载的 FLUX Kontext 模型）
    MODEL_PATH = "FLUX.1-Kontext-dev"

    # 输入图像路径（你想要编辑的图像）
    INPUT_IMAGE_PATH = "__LOCAL_PROJECT_ROOT__/test/input/0.png"

    # 输出图像路径（生成的图像保存位置）
    OUTPUT_IMAGE_PATH = "__LOCAL_PROJECT_ROOT__/test/output/0.png"

    # 文本提示词（描述你想要的效果）
    # 例如：
    # - "让这个人微笑"
    # - "把背景改成海滩"
    # - "改成油画风格"
    # - "添加一只猫在旁边"
    PROMPT = "Please convert the image to a perfectly exposed photograph with bright, natural lighting and clear details."

    # 推理步数（20-50 之间，越大质量越好但速度越慢）
    NUM_INFERENCE_STEPS = 28

    # 引导尺度（1.5-3.5 之间，越大越贴合提示词）
    GUIDANCE_SCALE = 2.5

    # 随机种子（固定种子可以复现结果，None 表示随机）
    SEED = 42

    # 运行设备（"cuda" 表示 GPU，"cpu" 表示 CPU）
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # GPU 设备 ID（如果有多张显卡，可以指定使用哪一张）
    # 提示：运行 nvidia-smi 查看哪些 GPU 空闲
    # GPU 0, 1, 3 都是空闲的，可以选择其中一张
    GPU_ID = 0  # 使用第一张 GPU

    # 是否启用混合 Offload 策略（强烈推荐！2025年最新方法）
    # 启用后显存占用从 24GB 降到 8-10GB
    USE_GROUP_OFFLOAD = True

    # ========================================================================
    # 混合 Offload 策略（自动应用，无需手动配置）
    # ========================================================================
    # 代码会自动对不同组件应用最优的 offload 策略：
    # - Transformer: block_level group offload（最大组件）
    # - Text Encoders & VAE: CPU offload（稳定可靠）
    # ========================================================================

    # Transformer 的块数配置
    # - 1: 每次加载1个块（最节省显存，推荐）✅
    # - 2: 每次加载2个块（速度稍快，显存稍多）
    # 注意：使用 use_stream=True 时，会自动设置为 1
    NUM_BLOCKS_PER_GROUP = 1

    # 输出图像尺寸（None 表示自动调整到合适尺寸）
    OUTPUT_HEIGHT = None
    OUTPUT_WIDTH = None

    # ========================================================================
    # 执行推理流程
    # ========================================================================

    try:
        # 步骤 1：加载模型
        # 注意：现在使用混合 offload 策略，自动对不同组件应用最优配置
        pipeline = load_flux_kontext_pipeline(
            model_path=MODEL_PATH,
            device=DEVICE,
            dtype=torch.bfloat16,
            gpu_id=GPU_ID,
            use_group_offload=USE_GROUP_OFFLOAD,
            num_blocks_per_group=NUM_BLOCKS_PER_GROUP  # Transformer 的 block_level 块数
        )

        # 步骤 2：加载输入图像
        input_image = load_input_image(INPUT_IMAGE_PATH)

        # 步骤 3：执行推理
        output_image = run_image_to_image(
            pipeline=pipeline,
            input_image=input_image,
            prompt=PROMPT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            seed=SEED,
            height=OUTPUT_HEIGHT,
            width=OUTPUT_WIDTH,
        )

        # 步骤 4：保存结果
        save_output_image(output_image, OUTPUT_IMAGE_PATH)

        # 打印成功信息
        print("\n" + "="*60)
        print("推理完成！")
        print(f"输出图像已保存到：{OUTPUT_IMAGE_PATH}")
        print("="*60 + "\n")

    except FileNotFoundError as e:
        print(f"\n[错误] {e}")
        print("请检查文件路径是否正确。\n")
    except Exception as e:
        print(f"\n[错误] 推理过程中发生错误：{e}")
        print("请检查模型文件是否完整，或尝试调整参数。\n")
        raise


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    """
    当直接运行此脚本时执行主函数

    使用方法：
        python inference.py
    """
    main()
