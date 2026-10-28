# -*- coding: utf-8 -*-
"""
FLUX Kontext 图生图推理代码（部分 LoRA - 仅单流块）
===========================================================

这是一个特殊版本的 FLUX Kontext 推理实现，只对单流 DIT 块加载 LoRA。

FLUX 模型架构说明：
- 双流 DIT 块（Double-Stream DiT Blocks）: transformer_blocks (19层)
  这些块分别处理图像latents和文本，只在注意力操作时合并

- 单流 DIT 块（Single-Stream DiT Blocks）: single_transformer_blocks (38层)
  这些块对合并后的图像和文本流进行统一处理

本代码的特点：
✅ LoRA 只加载到 single_transformer_blocks（单流块）
✅ transformer_blocks（双流块）、VAE、Text Encoders 不受 LoRA 影响
✅ 通过过滤 state_dict 实现选择性加载
✅ 保持与原始模型相同的推理流程

适用场景：
- 当你的 LoRA 主要影响最终的图像合成阶段（单流块）
- 想要保持文本-图像对齐阶段（双流块）不变
- 需要更精细的 LoRA 控制

代码结构：
1. 图像处理模块：负责加载和保存图像
2. 模型加载模块：负责加载 FLUX Kontext 模型 + 部分 LoRA
3. 推理执行模块：执行图生图推理
4. 主函数：整合所有功能
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict
import torch
from PIL import Image

# ============================================================================
# 第一部分：导入必要的库
# ============================================================================

# 添加本地 diffusers 库到系统路径
CURRENT_DIR = Path(__file__).parent
DIFFUSERS_PATH = "diffusers/src"
sys.path.insert(0, str(DIFFUSERS_PATH))

# 从 diffusers 库导入 FLUX Kontext 推理管道
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

# 导入 safetensors 用于直接加载 LoRA 权重
try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    print("[警告] safetensors 未安装，将无法加载 LoRA 权重")
    print("[提示] 请运行: pip install safetensors")
    SAFETENSORS_AVAILABLE = False


# ============================================================================
# 第二部分：图像处理模块（与 inference_lora.py 完全相同）
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
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到输入图像：{image_path}")

    print(f"[图像加载] 正在加载图像：{image_path}")
    image = load_image(image_path)
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
    output_dir = os.path.dirname(output_path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[图像保存] 创建输出目录：{output_dir}")

    image.save(output_path)
    print(f"[图像保存] 图像已保存到：{output_path}")


# ============================================================================
# 第三部分：LoRA 权重过滤模块（核心新增功能）
# ============================================================================

def filter_lora_state_dict_for_single_stream(
    state_dict: Dict[str, torch.Tensor],
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """
    过滤 LoRA state_dict，只保留单流 DIT 块相关的权重

    功能说明：
        FLUX Transformer 包含两类块：
        1. transformer_blocks（双流块，19层）- 将被过滤掉
        2. single_transformer_blocks（单流块，38层）- 保留

        本函数会遍历 state_dict，只保留键名中包含 "single_transformer_blocks" 的权重。

        LoRA 权重的键名格式示例：
        - transformer.transformer_blocks.0.attn.to_q.lora_A.weight
        - transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight

        根据 Hugging Face diffusers 文档，single_transformer_blocks 是负责
        处理合并后的图像+文本流的并行块，对最终图像生成有重要影响。

    参数：
        state_dict (Dict[str, torch.Tensor]): 完整的 LoRA 权重字典
        verbose (bool): 是否打印详细信息

    返回：
        Dict[str, torch.Tensor]: 过滤后的 state_dict（只包含单流块权重）
    """
    if verbose:
        print(f"\n[LoRA 过滤] ========== 开始过滤 LoRA 权重 ==========")
        print(f"[LoRA 过滤] 原始权重数量：{len(state_dict)} 个张量")

    # 关键词列表：用于识别单流块
    # 根据 FluxTransformer2DModel 的源码，单流块的属性名为 "single_transformer_blocks"
    single_stream_keywords = ["single_transformer_blocks"]

    # 过滤逻辑：保留包含关键词的权重
    filtered_dict = {}
    excluded_dict = {}

    for key, value in state_dict.items():
        # 检查键名是否包含单流块关键词
        is_single_stream = any(keyword in key for keyword in single_stream_keywords)

        if is_single_stream:
            filtered_dict[key] = value
        else:
            excluded_dict[key] = value

    if verbose:
        print(f"[LoRA 过滤] 保留权重数量：{len(filtered_dict)} 个张量（单流块）")
        print(f"[LoRA 过滤] 排除权重数量：{len(excluded_dict)} 个张量（双流块、文本编码器等）")

        # 打印一些示例
        if filtered_dict:
            print(f"\n[LoRA 过滤] 保留的权重示例（前3个）：")
            for i, key in enumerate(list(filtered_dict.keys())[:3]):
                print(f"  ✓ {key}")

        if excluded_dict:
            print(f"\n[LoRA 过滤] 排除的权重示例（前3个）：")
            for i, key in enumerate(list(excluded_dict.keys())[:3]):
                print(f"  ✗ {key}")

        print(f"\n[LoRA 过滤] ========== LoRA 权重过滤完成 ==========\n")

    return filtered_dict


# ============================================================================
# 第四部分：模型加载模块（增强版：支持部分 LoRA）
# ============================================================================

def load_flux_kontext_pipeline(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    gpu_id: int = 0,
    use_group_offload: bool = True,
    num_blocks_per_group: int = 1,
    lora_path: Optional[str] = None,
    lora_scale: float = 1.0,
    lora_target_modules: str = "single_stream_only"
) -> FluxKontextPipeline:
    """
    加载 FLUX Kontext 推理管道（支持部分 LoRA）

    功能说明：
        加载 FLUX Kontext 模型，并支持选择性地只对单流 DIT 块加载 LoRA。

        LoRA 加载策略（新增）：
        1. 从 safetensors 文件加载完整的 LoRA 权重
        2. 过滤 state_dict，只保留 single_transformer_blocks 相关的权重
        3. 使用过滤后的权重加载 LoRA
        4. 结果：双流块、VAE、文本编码器保持原始权重不变

        为什么只加载到单流块：
        - 单流块（single_transformer_blocks）处理合并后的图像+文本信息
        - 双流块（transformer_blocks）负责早期的文本-图像对齐
        - 如果你的 LoRA 主要影响图像生成风格，单流块是主要作用点
        - 保持双流块不变可以维持更好的文本理解和对齐

    参数：
        model_path (str): 模型权重文件的本地路径
        device (str): 运行设备，"cuda" 表示 GPU
        dtype (torch.dtype): 模型精度，torch.bfloat16 可以节省显存
        gpu_id (int): GPU 设备 ID
        use_group_offload (bool): 是否使用混合 Offload 策略
        num_blocks_per_group (int): Transformer 的块数配置
        lora_path (str, optional): LoRA 权重文件路径
        lora_scale (float): LoRA 强度，范围 0.0-2.0
        lora_target_modules (str): LoRA 目标模块，"single_stream_only" 表示只加载到单流块

    返回：
        FluxKontextPipeline: 加载完成的推理管道（已应用部分 LoRA）
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
        print(f"[LoRA 配置] LoRA 目标：{lora_target_modules}")
        print(f"[LoRA 配置] 策略说明：只加载到单流 DIT 块（single_transformer_blocks）")

    # 检查模型路径
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型路径：{model_path}")

    # 从本地路径加载预训练模型
    print(f"[模型加载] 正在从磁盘加载模型组件...")
    pipeline = FluxKontextPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )

    # ========================================================================
    # 部分 LoRA 加载（核心新增功能）
    # ========================================================================

    if lora_path is not None and lora_target_modules == "single_stream_only":
        print(f"\n[LoRA 加载] ========== 开始加载部分 LoRA 权重 ==========")

        # 检查 LoRA 文件
        if not os.path.exists(lora_path):
            print(f"[LoRA 加载] ⚠️  警告：找不到 LoRA 文件：{lora_path}")
            print(f"[LoRA 加载] 将继续使用基础模型（不加载 LoRA）")
        elif not SAFETENSORS_AVAILABLE:
            print(f"[LoRA 加载] ⚠️  警告：safetensors 库未安装")
            print(f"[LoRA 加载] 将继续使用基础模型（不加载 LoRA）")
        else:
            try:
                # 步骤 1：加载完整的 LoRA 权重
                lora_size_mb = os.path.getsize(lora_path) / (1024 * 1024)
                print(f"[LoRA 加载] LoRA 文件大小：{lora_size_mb:.2f} MB")
                print(f"[LoRA 加载] 正在从磁盘加载 LoRA 权重...")

                full_state_dict = load_file(lora_path)
                print(f"[LoRA 加载] ✓ LoRA 权重加载成功（完整）")

                # 步骤 2：过滤权重，只保留单流块相关的权重
                filtered_state_dict = filter_lora_state_dict_for_single_stream(
                    full_state_dict,
                    verbose=True
                )

                # 步骤 3：检查是否有有效的权重
                if not filtered_state_dict:
                    print(f"[LoRA 加载] ⚠️  警告：没有找到单流块相关的 LoRA 权重")
                    print(f"[LoRA 加载] 提示：请确认 LoRA 文件是否包含 'single_transformer_blocks' 权重")
                    print(f"[LoRA 加载] 将继续使用基础模型（不加载 LoRA）")
                else:
                    # 步骤 4：创建临时文件保存过滤后的权重
                    # 因为 load_lora_weights 需要文件路径，我们需要保存过滤后的权重
                    from safetensors.torch import save_file

                    temp_lora_dir = CURRENT_DIR / "temp_lora"
                    temp_lora_dir.mkdir(exist_ok=True)
                    temp_lora_path = temp_lora_dir / "filtered_single_stream_lora.safetensors"

                    print(f"[LoRA 加载] 正在保存过滤后的权重到临时文件...")
                    save_file(filtered_state_dict, str(temp_lora_path))
                    print(f"[LoRA 加载] ✓ 临时文件已创建：{temp_lora_path}")

                    # 步骤 5：使用 diffusers 的方法加载过滤后的 LoRA
                    print(f"[LoRA 加载] 正在加载 LoRA 权重到 pipeline...")

                    pipeline.load_lora_weights(
                        str(temp_lora_dir),
                        weight_name="filtered_single_stream_lora.safetensors",
                        adapter_name="LLE_Lora_SingleStream"
                    )

                    print(f"[LoRA 加载] ✓ LoRA 权重加载成功（仅单流块）")
                    print(f"[LoRA 加载] 设置 LoRA 强度为：{lora_scale}")
                    print(f"[LoRA 加载] ========== 部分 LoRA 加载完成 ==========\n")

            except Exception as e:
                print(f"[LoRA 加载] ✗ 加载 LoRA 失败：{e}")
                print(f"[LoRA 加载] 将继续使用基础模型（不加载 LoRA）")
                import traceback
                traceback.print_exc()

    elif lora_path is not None:
        # 如果指定了其他 target_modules，使用标准方法
        print(f"\n[LoRA 加载] 使用标准 LoRA 加载方法（非部分加载）")
        try:
            lora_dir = os.path.dirname(lora_path)
            lora_filename = os.path.basename(lora_path)

            pipeline.load_lora_weights(
                lora_dir,
                weight_name=lora_filename,
                adapter_name="LLE_Lora"
            )
            print(f"[LoRA 加载] ✓ LoRA 权重加载成功（完整）")
        except Exception as e:
            print(f"[LoRA 加载] ✗ 加载 LoRA 失败：{e}")

    # ========================================================================
    # Group Offload 优化（与 inference_lora.py 相同）
    # ========================================================================

    if device == "cuda" and use_group_offload:
        print(f"\n[模型加载] ========== 启用混合 Offload 策略 ==========")
        print(f"[模型加载] 策略说明：")
        print(f"[模型加载]   - Transformer: block_level（最大组件，约 20GB）")
        print(f"[模型加载]   - Text Encoders: block_level（稳定性优先）")
        print(f"[模型加载]   - VAE: leaf_level（避免设备不匹配）")
        print(f"[模型加载] ")

        from diffusers.hooks import apply_group_offloading

        onload_device = torch.device(f"cuda:{gpu_id}")
        offload_device = torch.device("cpu")

        # Transformer
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

        # Text Encoder (CLIP)
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

        # Text Encoder 2 (T5)
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

        # VAE
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
        print(f"[模型加载] - Text Encoders: block_level")
        print(f"[模型加载] - VAE: leaf_level")

        if lora_path and os.path.exists(lora_path):
            print(f"[模型加载] LoRA 已加载：{os.path.basename(lora_path)} (强度: {lora_scale})")
            if lora_target_modules == "single_stream_only":
                print(f"[模型加载] LoRA 范围：仅单流 DIT 块（single_transformer_blocks）")

        print(f"[模型加载] ========================================================\n")

    elif device == "cuda":
        print(f"[模型加载] 警告：未启用 Group Offload，将尝试完整加载模型到 GPU")
        print(f"[模型加载] 这需要约 24GB 显存，可能会导致 OOM 错误！")
        pipeline = pipeline.to(device)
    else:
        print(f"[模型加载] 使用 CPU 模式（警告：速度会非常慢）")
        pipeline = pipeline.to(device)

    print(f"[模型加载] 模型加载完成！")

    return pipeline


# ============================================================================
# 第五部分：推理执行模块（与 inference_lora.py 完全相同）
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

    参数：
        pipeline (FluxKontextPipeline): 已加载的推理管道
        input_image (Image.Image): 输入的参考图像
        prompt (str): 文本提示词
        num_inference_steps (int): 推理步数
        guidance_scale (float): 引导尺度
        seed (int, optional): 随机种子
        height (int, optional): 输出图像高度
        width (int, optional): 输出图像宽度
        lora_scale (float): LoRA 强度

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

    # 设置随机种子
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        print(f"[推理执行] 已设置随机种子：{seed}")

    print(f"[推理执行] 正在生成图像，请稍候...")

    # 设置 LoRA 强度
    joint_attention_kwargs = {"scale": lora_scale}

    result = pipeline(
        image=input_image,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        height=height,
        width=width,
        joint_attention_kwargs=joint_attention_kwargs,
    )

    output_image = result.images[0]

    print(f"[推理执行] 图像生成完成！")
    print(f"{'='*60}\n")

    return output_image


# ============================================================================
# 第六部分：主函数
# ============================================================================

def main():
    """
    主函数：执行完整的推理流程（使用部分 LoRA）
    """

    print("\n" + "="*60)
    print("FLUX Kontext 图生图推理程序（部分 LoRA - 仅单流块）")
    print("="*60 + "\n")

    # ========================================================================
    # 配置部分
    # ========================================================================

    # 模型路径
    MODEL_PATH = "FLUX.1-Kontext-dev"

    # 输入/输出图像路径
    INPUT_IMAGE_PATH = "__LOCAL_PROJECT_ROOT__/test/input/0.png"
    OUTPUT_IMAGE_PATH = "__LOCAL_PROJECT_ROOT__/test/output/0_lora_single_stream.png"

    # 文本提示词
    PROMPT = "LLE. Please convert the image to a perfectly exposed photograph with bright, natural lighting and clear details."

    # ========================================================================
    # LoRA 配置（部分加载）
    # ========================================================================

    # LoRA 权重文件路径
    LORA_PATH = "lora_model/LLE_Lora.safetensors"
    # LORA_PATH = None  # 取消注释以禁用 LoRA

    # LoRA 强度
    LORA_SCALE = 1.0

    # LoRA 目标模块（重要！）
    # - "single_stream_only": 只加载到单流块（推荐）
    # - "all": 加载到所有模块（标准方法）
    LORA_TARGET_MODULES = "single_stream_only"

    # ========================================================================
    # 推理配置
    # ========================================================================

    NUM_INFERENCE_STEPS = 28
    GUIDANCE_SCALE = 2.5
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_ID = 0
    USE_GROUP_OFFLOAD = True
    NUM_BLOCKS_PER_GROUP = 1
    OUTPUT_HEIGHT = None
    OUTPUT_WIDTH = None

    # ========================================================================
    # 执行推理流程
    # ========================================================================

    try:
        # 步骤 1：加载模型（包含部分 LoRA）
        pipeline = load_flux_kontext_pipeline(
            model_path=MODEL_PATH,
            device=DEVICE,
            dtype=torch.bfloat16,
            gpu_id=GPU_ID,
            use_group_offload=USE_GROUP_OFFLOAD,
            num_blocks_per_group=NUM_BLOCKS_PER_GROUP,
            lora_path=LORA_PATH,
            lora_scale=LORA_SCALE,
            lora_target_modules=LORA_TARGET_MODULES
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
            lora_scale=LORA_SCALE
        )

        # 步骤 4：保存结果
        save_output_image(output_image, OUTPUT_IMAGE_PATH)

        # 打印成功信息
        print("\n" + "="*60)
        print("推理完成！")
        print(f"输出图像已保存到：{OUTPUT_IMAGE_PATH}")
        if LORA_PATH and os.path.exists(LORA_PATH):
            print(f"使用的 LoRA：{os.path.basename(LORA_PATH)} (强度: {LORA_SCALE})")
            print(f"LoRA 范围：仅单流 DIT 块（single_transformer_blocks）")
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
    使用方法：
        python inference_lora_part_module.py

    与 inference_lora.py 的区别：
        本脚本只对单流 DIT 块（single_transformer_blocks）加载 LoRA，
        双流块（transformer_blocks）、VAE、文本编码器保持原始权重。

    适用场景：
        - 你的 LoRA 主要影响图像生成风格
        - 想保持文本理解和对齐不变
        - 需要更精细的 LoRA 控制
    """
    main()
