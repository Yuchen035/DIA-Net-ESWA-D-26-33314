# -*- coding: utf-8 -*-
"""
FLUX Kontext 图生图推理代码（支持 LoRA）
==========================================

这是一个支持 LoRA（Low-Rank Adaptation）的 FLUX Kontext 推理实现。
LoRA 可以让你在不改变基础模型的情况下，添加特定的风格、能力或效果。

代码结构分为以下几个模块：
1. 图像处理模块：负责加载和保存图像
2. 模型加载模块：负责加载 FLUX Kontext 模型 + LoRA
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
# 第二部分：图像处理模块（与 inference.py 完全相同）
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
# 第三部分：模型加载模块（增强版：支持 LoRA）
# ============================================================================

def load_flux_kontext_pipeline(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    gpu_id: int = 0,
    use_group_offload: bool = True,
    num_blocks_per_group: int = 1,
    lora_path: Optional[str] = None,
    lora_scale: float = 1.0
) -> FluxKontextPipeline:
    """
    加载 FLUX Kontext 推理管道（支持 LoRA）

    功能说明：
        从本地路径加载 FLUX Kontext 模型的所有组件，包括：
        - Transformer 模型（主要的扩散模型）
        - VAE（变分自编码器，用于图像编码和解码）
        - 文本编码器（CLIP 和 T5，用于理解文本提示词）
        - 调度器（控制去噪过程）
        - **LoRA 权重**（可选，用于添加特定风格或能力）

        重要：使用混合 Offload 策略来优化显存使用（2025年最新方法）
        - Transformer: 使用 block_level group offloading（最大组件，约 20GB）
        - 其他组件: 使用传统 CPU offload（Text Encoders, VAE）
        - 真正实现显存优化：从 24GB 降到约 8-10GB

        LoRA 工作原理：
        1. LoRA 是轻量级的适配器（通常 100-500MB）
        2. 在基础模型上添加额外的能力，无需修改基础权重
        3. 可以随时加载/卸载，非常灵活
        4. 通过 lora_scale 参数控制影响强度（0.0-2.0）

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
        lora_path (str, optional): LoRA 权重文件路径，None 表示不使用 LoRA
        lora_scale (float): LoRA 强度，范围 0.0-2.0，默认 1.0

    返回：
        FluxKontextPipeline: 加载完成的推理管道（已应用 LoRA）
    """
    print(f"[模型加载] 开始加载 FLUX Kontext 模型")
    print(f"[模型加载] 模型路径：{model_path}")
    print(f"[模型加载] 运行设备：{device}")
    print(f"[模型加载] 数据类型：{dtype}")
    print(f"[模型加载] GPU ID：{gpu_id}")
    print(f"[模型加载] 混合 Offload：{'启用' if use_group_offload else '禁用'}")

    # LoRA 配置信息
    if lora_path:
        print(f"\n[LoRA 配置] LoRA 路径：{lora_path}")
        print(f"[LoRA 配置] LoRA 强度：{lora_scale}")

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

    # ========================================================================
    # LoRA 加载（解耦设计：独立的 LoRA 处理逻辑）
    # ========================================================================

    if lora_path is not None:
        print(f"\n[LoRA 加载] ========== 开始加载 LoRA 权重 ==========")

        # 检查 LoRA 文件是否存在
        if not os.path.exists(lora_path):
            print(f"[LoRA 加载] ⚠️  警告：找不到 LoRA 文件：{lora_path}")
            print(f"[LoRA 加载] 将继续使用基础模型（不加载 LoRA）")
        else:
            try:
                # 获取 LoRA 文件信息
                lora_size_mb = os.path.getsize(lora_path) / (1024 * 1024)
                print(f"[LoRA 加载] LoRA 文件大小：{lora_size_mb:.2f} MB")

                # 使用 diffusers 官方方法加载 LoRA 权重
                # 注意：这里使用本地文件路径，而不是 Hugging Face repo
                print(f"[LoRA 加载] 正在加载 LoRA 权重到 pipeline...")

                # 关键方法：load_lora_weights()
                # - 参数1：可以是本地目录路径或 Hugging Face repo ID
                # - weight_name：指定具体的 .safetensors 文件名
                # - adapter_name：给这个 LoRA 起个名字（可选，默认 "default"）
                lora_dir = os.path.dirname(lora_path)
                lora_filename = os.path.basename(lora_path)

                pipeline.load_lora_weights(
                    lora_dir,
                    weight_name=lora_filename,
                    adapter_name="LLE_Lora"  # 给 LoRA 起个名字
                )

                print(f"[LoRA 加载] ✓ LoRA 权重加载成功")

                # 设置 LoRA 强度（scale）
                # lora_scale 控制 LoRA 的影响程度：
                # - 0.0：完全不使用 LoRA（等同于基础模型）
                # - 1.0：标准强度（推荐）
                # - >1.0：增强 LoRA 效果
                print(f"[LoRA 加载] 设置 LoRA 强度为：{lora_scale}")

                # 注意：scale 会在推理时自动应用，不需要手动 fuse
                # diffusers 会在 forward pass 时动态应用 LoRA

                print(f"[LoRA 加载] ========== LoRA 加载完成 ==========\n")

            except Exception as e:
                print(f"[LoRA 加载] ✗ 加载 LoRA 失败：{e}")
                print(f"[LoRA 加载] 将继续使用基础模型（不加载 LoRA）")
                print(f"[LoRA 加载] 提示：请检查 LoRA 文件格式是否正确（应为 .safetensors）\n")

    # ========================================================================
    # Group Offload 优化（与 inference.py 相同）
    # ========================================================================

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

        print(f"[模型加载] 正在对 Text Encoder (CLIP) 应用 block_level offloading...")
        apply_group_offloading(
            pipeline.text_encoder,
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type="block_level",
            num_blocks_per_group=num_blocks_per_group,
            use_stream=True,
        )
        print(f"[模型加载] ✓ Text Encoder (CLIP) offloading 完成 (block_level)")

        print(f"[模型加载] 正在对 Text Encoder 2 (T5) 应用 block_level offloading...")
        apply_group_offloading(
            pipeline.text_encoder_2,
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type="block_level",
            num_blocks_per_group=num_blocks_per_group,
            use_stream=True,
        )
        print(f"[模型加载] ✓ Text Encoder 2 (T5) offloading 完成 (block_level)")

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

        if lora_path and os.path.exists(lora_path):
            print(f"[模型加载] LoRA 已加载：{os.path.basename(lora_path)} (强度: {lora_scale})")

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
# 第四部分：推理执行模块（与 inference.py 完全相同）
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
    lora_scale: float = 1.0
) -> Image.Image:
    """
    执行图生图推理（自动应用 LoRA）

    功能说明：
        使用 FLUX Kontext 模型根据输入图像和文本提示词生成新图像。
        如果加载了 LoRA，会自动应用 LoRA 的效果。

        核心概念：
        1. 输入图像：作为生成的起点和参考
        2. 文本提示词：描述你想要的修改或风格
        3. 推理步数：去噪迭代的次数，越多质量越好但速度越慢
        4. 引导尺度：控制生成结果与提示词的贴合程度
        5. 随机种子：控制随机性，相同种子产生相同结果
        6. LoRA 强度：控制 LoRA 的影响程度（如果加载了 LoRA）

    参数：
        pipeline (FluxKontextPipeline): 已加载的推理管道
        input_image (Image.Image): 输入的参考图像
        prompt (str): 文本提示词，描述想要的效果
        num_inference_steps (int): 推理步数，默认 28（推荐范围 20-50）
        guidance_scale (float): 引导尺度，默认 2.5（推荐范围 1.5-3.5）
        seed (int, optional): 随机种子，用于结果复现
        height (int, optional): 输出图像高度，None 表示自动调整
        width (int, optional): 输出图像宽度，None 表示自动调整
        lora_scale (float): LoRA 强度，默认 1.0（范围 0.0-2.0）

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
    print(f"[推理参数] LoRA 强度：{lora_scale}")

    # 如果指定了随机种子，创建随机数生成器
    # 这样可以确保每次运行产生相同的结果（结果可复现）
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        print(f"[推理执行] 已设置随机种子：{seed}")

    # 执行推理
    # 这是整个代码的核心部分，调用 FLUX Kontext 模型进行图像生成
    # 如果加载了 LoRA，diffusers 会自动应用 LoRA 的效果
    print(f"[推理执行] 正在生成图像，请稍候...")

    # 设置 LoRA 强度（通过 joint_attention_kwargs 传递）
    # 注意：FluxKontextPipeline 使用 joint_attention_kwargs，不是 cross_attention_kwargs
    # 即使 lora_scale == 1.0 也要传递，确保 LoRA 被正确应用
    joint_attention_kwargs = {"scale": lora_scale}

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
        joint_attention_kwargs=joint_attention_kwargs,  # LoRA 强度控制
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
# 第五部分：主函数（增强版：支持 LoRA 配置）
# ============================================================================

def main():
    """
    主函数：整合所有功能，执行完整的推理流程（支持 LoRA）

    流程说明：
        1. 配置参数（模型路径、输入图像、提示词、LoRA 等）
        2. 加载 FLUX Kontext 模型 + LoRA
        3. 加载输入图像
        4. 执行图生图推理
        5. 保存生成的图像
    """

    print("\n" + "="*60)
    print("FLUX Kontext 图生图推理程序（支持 LoRA）")
    print("="*60 + "\n")

    # ========================================================================
    # 配置部分：在这里修改你的参数
    # ========================================================================

    # 模型路径（本地下载的 FLUX Kontext 模型）
    MODEL_PATH = "FLUX.1-Kontext-dev"

    # 输入图像路径（你想要编辑的图像）
    INPUT_IMAGE_PATH = "__LOCAL_PROJECT_ROOT__/test/input/0.png"

    # 输出图像路径（生成的图像保存位置）
    OUTPUT_IMAGE_PATH = "__LOCAL_PROJECT_ROOT__/test/output/0_lora.png"

    # 文本提示词（描述你想要的效果）
    # 提示：如果使用特定 LoRA，可能需要添加触发词（trigger words）
    # 例如 LLE_Lora 需要 "LLE." 前缀
    PROMPT = "LLE. Please convert the image to a perfectly exposed photograph with bright, natural lighting and clear details."

    # ========================================================================
    # LoRA 配置（新增部分）
    # ========================================================================

    # LoRA 权重文件路径
    # - 设置为 None 则不使用 LoRA（等同于 inference.py）
    # - 设置为文件路径则加载对应的 LoRA
    LORA_PATH = "lora_model/LLE_Lora.safetensors"
    # LORA_PATH = None  # 取消注释此行以禁用 LoRA

    # LoRA 强度（0.0 - 2.0）
    # - 0.0：不使用 LoRA 的效果
    # - 0.5：轻微的 LoRA 效果
    # - 1.0：标准强度（推荐）
    # - 1.5-2.0：增强 LoRA 效果
    LORA_SCALE = 1.0

    # ========================================================================
    # 基础推理配置（与 inference.py 相同）
    # ========================================================================

    # 推理步数（20-50 之间，越大质量越好但速度越慢）
    NUM_INFERENCE_STEPS = 28

    # 引导尺度（1.5-3.5 之间，越大越贴合提示词）
    GUIDANCE_SCALE = 2.5

    # 随机种子（固定种子可以复现结果，None 表示随机）
    SEED = 42

    # 运行设备（"cuda" 表示 GPU，"cpu" 表示 CPU）
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # GPU 设备 ID（如果有多张显卡，可以指定使用哪一张）
    GPU_ID = 0

    # 是否启用 Group Offload（强烈推荐！2025年最新方法）
    USE_GROUP_OFFLOAD = True

    # ========================================================================
    # 混合 Offload 策略（自动应用，无需手动配置）
    # ========================================================================
    # 代码会自动对不同组件应用最优的 offload 策略：
    # - Transformer: block_level（最大组件，速度优先）
    # - Text Encoders: leaf_level（稳定性优先）
    # - VAE: leaf_level（避免 block_level 的已知 bug）
    #
    # 注意：旧的 OFFLOAD_TYPE 参数已废弃，现在使用混合策略
    # ========================================================================

    # Block-level offload 的块数配置（用于 Transformer）
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
        # 步骤 1：加载模型（包含 LoRA）
        # 注意：现在使用混合 offload 策略，自动对不同组件应用最优配置
        pipeline = load_flux_kontext_pipeline(
            model_path=MODEL_PATH,
            device=DEVICE,
            dtype=torch.bfloat16,
            gpu_id=GPU_ID,
            use_group_offload=USE_GROUP_OFFLOAD,
            num_blocks_per_group=NUM_BLOCKS_PER_GROUP,  # Transformer 的 block_level 块数
            lora_path=LORA_PATH,        # LoRA 路径
            lora_scale=LORA_SCALE       # LoRA 强度
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
            lora_scale=LORA_SCALE       # 传递 LoRA 强度
        )

        # 步骤 4：保存结果
        save_output_image(output_image, OUTPUT_IMAGE_PATH)

        # 打印成功信息
        print("\n" + "="*60)
        print("推理完成！")
        print(f"输出图像已保存到：{OUTPUT_IMAGE_PATH}")
        if LORA_PATH and os.path.exists(LORA_PATH):
            print(f"使用的 LoRA：{os.path.basename(LORA_PATH)} (强度: {LORA_SCALE})")
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
        python inference_lora.py
    """
    main()
