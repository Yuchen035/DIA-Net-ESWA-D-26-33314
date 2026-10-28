# -*- coding: utf-8 -*-
"""
FLUX Kontext 推理（MoE-LoRA 版，带混合 Offload，2025 策略）
用法示例：
    python inference/inference_moe_lora.py \\
        --model-path ../FLUX.1-Kontext-dev \\
        --lora-path ./lora_output/pytorch_lora_weights.safetensors \\
        --image ./test/input/0.png \\
        --prompt "make it bright" \\
        --output ./outputs/out.png \\
        --dtype float16 \\
        --offload sequential
"""

import argparse
import os
import sys
from pathlib import Path
CURRENT_DIR = Path(__file__).parent
SOURCE_ROOT = CURRENT_DIR.parent
sys.path[:0] = [str(SOURCE_ROOT / "peft" / "src"),
               str(SOURCE_ROOT / "diffusers" / "src"), str(SOURCE_ROOT)]

from adapter_checkpoint import read_checkpoint_metadata, load_adapter_checkpoint

# Model libraries are imported only after CLI/resource preflight in main().


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="基础模型路径")
    adapter = parser.add_mutually_exclusive_group(required=True)
    adapter.add_argument("--lora-path", help="v02 .safetensors 权重（必须有同名 .safetensors.json）")
    adapter.add_argument("--base-model-only", action="store_true", help="显式使用未适配基础模型；不是 DIA-Net 推理")
    parser.add_argument("--prompt", required=True, help="提示词")
    parser.add_argument("--image", required=True, help="控制图路径")
    parser.add_argument("--output", default="./outputs/sample.png", help="输出路径")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--offload", choices=["sequential", "model", "none"], default="sequential")
    return parser.parse_args()


def str_to_dtype(name: str):
    import torch
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return torch.float16


def main():
    args = parse_args()
    if args.lora_path:
        # Fail before loading the base model, including for legacy/absent files.
        read_checkpoint_metadata(args.lora_path)
    if not Path(args.model_path).is_dir():
        raise ValueError("--model-path must be an existing local directory.")
    if not Path(args.image).is_file():
        raise ValueError("--image must be an existing local file.")
    if Path(args.output).exists():
        raise ValueError("Refusing to overwrite an existing output image.")
    import torch
    from diffusers import FluxKontextPipeline
    from diffusers.utils import load_image
    dtype = str_to_dtype(args.dtype)

    print("[加载] 模型与管线")
    pipe = FluxKontextPipeline.from_pretrained(args.model_path, torch_dtype=dtype,
                                               local_files_only=True)

    if args.lora_path:
        print(f"[加载] 应用 LoRA: {args.lora_path}")
        restored = load_adapter_checkpoint(pipe.transformer, args.lora_path)
        print(f"[加载] 已恢复 {restored['loaded_tensor_count']} 个适配器张量")
    else:
        print("[模型] 显式使用未适配基础模型；不是 DIA-Net 推理")

    # Offload failures stop execution; do not continue with an unknown device state.
    if args.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
        print("[Offload] 已启用 sequential cpu offload")
    elif args.offload == "model":
        pipe.enable_model_cpu_offload()
        print("[Offload] 已启用 model cpu offload")

    image = load_image(args.image).convert("RGB").resize((args.width, args.height))

    print("[推理] 开始生成")
    with torch.inference_mode():
        out = pipe(
            prompt=args.prompt,
            image=image,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance,
            output_type="pil",
        ).images[0]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.save(args.output)
    print(f"[完成] 已保存到 {args.output}")


if __name__ == "__main__":
    main()
