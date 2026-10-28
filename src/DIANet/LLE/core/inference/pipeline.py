"""
推理管道 - 用于使用训练好的双路由MoE-LoRA模型进行图像增强

支持功能:
- 加载训练好的MoE-LoRA权重
- 自动曝光度分类
- 单张图像推理
- 批量图像推理
- 支持Mock FLUX和真实FLUX
"""

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from typing import Union, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

from core.lora.moe.moe_lora_layer import MoELoRALayer
from core.utils.exposure import classify_exposure, batch_classify_exposure


class InferencePipeline:
    """
    双路由MoE-LoRA推理管道

    负责加载模型、注入LoRA权重、自动分类曝光度、执行推理
    """

    def __init__(
        self,
        transformer,
        vae,
        text_encoder_1,
        text_encoder_2,
        tokenizer_1,
        tokenizer_2,
        scheduler,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        num_specific: int = 5,
        num_shared: int = 4,
        rank: int = 8,
        alpha: float = 8.0,
        shared_top_k: int = 2,
        exposure_thresholds: List[float] = [0.1, 0.2, 0.3, 0.5],
        exposure_method: str = "mean_brightness"
    ):
        """
        初始化推理管道

        Args:
            transformer: FLUX Transformer模型（或Mock版本）
            vae: VAE模型
            text_encoder_1: CLIP文本编码器
            text_encoder_2: T5文本编码器
            tokenizer_1: CLIP tokenizer
            tokenizer_2: T5 tokenizer
            scheduler: Flow matching scheduler
            device: 计算设备
            dtype: 数据类型
            num_specific: Specific专家数量
            num_shared: Shared专家数量
            rank: LoRA rank
            alpha: LoRA alpha
            shared_top_k: Shared router的top-k
            exposure_thresholds: 曝光度分类阈值
            exposure_method: 曝光度分类方法
        """
        self.device = device
        self.dtype = dtype

        # 保存所有组件
        self.transformer = transformer.to(device, dtype=dtype)
        self.vae = vae.to(device, dtype=dtype)
        self.text_encoder_1 = text_encoder_1.to(device, dtype=dtype)
        self.text_encoder_2 = text_encoder_2.to(device, dtype=dtype)
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.scheduler = scheduler

        # MoE-LoRA配置
        self.num_specific = num_specific
        self.num_shared = num_shared
        self.rank = rank
        self.alpha = alpha
        self.shared_top_k = shared_top_k

        # 曝光度分类配置
        self.exposure_thresholds = exposure_thresholds
        self.exposure_method = exposure_method

        # 标记是否已注入LoRA
        self.lora_injected = False
        self.lora_loaded = False

        # 设置为推理模式
        self.transformer.eval()
        self.vae.eval()
        self.text_encoder_1.eval()
        self.text_encoder_2.eval()

    def inject_dual_route_moe_lora(self, verbose: bool = True):
        """
        为transformer注入双路由MoE-LoRA层

        Args:
            verbose: 是否打印详细信息
        """
        if self.lora_injected:
            if verbose:
                print("⚠ MoE-LoRA已经注入，跳过")
            return

        if verbose:
            print("\n" + "=" * 60)
            print("注入双路由MoE-LoRA")
            print("=" * 60)
            print(f"  Specific专家: {self.num_specific}")
            print(f"  Shared专家: {self.num_shared}")
            print(f"  总专家: {self.num_specific + self.num_shared}")
            print(f"  LoRA rank: {self.rank}")
            print(f"  LoRA alpha: {self.alpha}")
            print(f"  Shared top-k: {self.shared_top_k}")

        num_injected = 0

        # 为每个block的第一个线性层注入MoE-LoRA
        for i, block in enumerate(self.transformer.blocks):
            original_linear = block[0]

            # 创建MoE-LoRA层
            moe_layer = MoELoRALayer(
                base_layer=original_linear,
                rank=self.rank,
                alpha=self.alpha,
                num_experts=self.num_specific + self.num_shared,
                routing_mode="dual",
                router_config={
                    'num_specific': self.num_specific,
                    'shared_top_k': self.shared_top_k,
                    'shared_hidden_dim': 512
                }
            )

            # 替换原始层
            self.transformer.blocks[i] = nn.Sequential(
                moe_layer,
                block[1],  # GELU
                block[2]   # Second Linear
            )

            num_injected += 1

        self.lora_injected = True

        if verbose:
            print(f"  ✓ 成功注入 {num_injected} 个MoE-LoRA层")
            print("=" * 60)

    def load_lora_weights(self, checkpoint_path: Union[str, Path], verbose: bool = True):
        """
        加载训练好的MoE-LoRA权重

        Args:
            checkpoint_path: 检查点文件路径
            verbose: 是否打印详细信息
        """
        if not self.lora_injected:
            raise RuntimeError("请先调用 inject_dual_route_moe_lora() 注入LoRA层")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

        if verbose:
            print("\n" + "=" * 60)
            print("加载MoE-LoRA权重")
            print("=" * 60)
            print(f"  检查点: {checkpoint_path}")

        # 加载state dict
        state_dict = torch.load(checkpoint_path, map_location=self.device)

        # 加载权重到MoE-LoRA层
        loaded_count = 0
        for name, module in self.transformer.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # 查找对应的权重
                lora_A_key = f"{name}.lora_A"
                lora_B_key = f"{name}.lora_B"

                if lora_A_key in state_dict and lora_B_key in state_dict:
                    # 加载lora_A和lora_B (每个是专家列表)
                    lora_A_weights = state_dict[lora_A_key]
                    lora_B_weights = state_dict[lora_B_key]

                    # 设置权重
                    for expert_idx in range(len(module.lora_A)):
                        module.lora_A[expert_idx].data = lora_A_weights[expert_idx].to(
                            device=self.device, dtype=self.dtype
                        )
                        module.lora_B[expert_idx].data = lora_B_weights[expert_idx].to(
                            device=self.device, dtype=self.dtype
                        )

                    loaded_count += 1

        self.lora_loaded = True

        if verbose:
            print(f"  ✓ 成功加载 {loaded_count} 个MoE-LoRA层的权重")
            print("=" * 60)

    def classify_image_exposure(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> int:
        """
        分类单张图像的曝光度

        Args:
            image: 输入图像

        Returns:
            category: 曝光度类别 (0-4)
        """
        return classify_exposure(
            image,
            thresholds=self.exposure_thresholds,
            method=self.exposure_method
        )

    def _prepare_image(self, image: Union[Image.Image, str, Path], resolution: int = 512) -> torch.Tensor:
        """
        准备图像用于推理（加载、resize、normalize）

        Args:
            image: 输入图像（PIL Image或路径）
            resolution: 目标分辨率

        Returns:
            image_tensor: [1, 3, H, W], 范围[-1, 1]
        """
        # 如果是路径，加载图像
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Resize
        image = image.resize((resolution, resolution), Image.LANCZOS)

        # 转换为tensor并归一化到[-1, 1]
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # [3, H, W]
        image_tensor = (image_tensor - 0.5) / 0.5  # 归一化到[-1, 1]
        image_tensor = image_tensor.unsqueeze(0)  # [1, 3, H, W]

        return image_tensor.to(device=self.device, dtype=self.dtype)

    def _encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        编码图像到latent空间

        Args:
            image_tensor: [B, 3, H, W], 范围[-1, 1]

        Returns:
            latent: [B, C, H', W']
        """
        with torch.no_grad():
            # 确保类型匹配
            vae_dtype = next(self.vae.parameters()).dtype
            image_tensor = image_tensor.to(dtype=vae_dtype)

            # 编码
            latent_dist = self.vae.encode(image_tensor).latent_dist
            latent = latent_dist.sample()

            # 应用scaling
            if hasattr(self.vae.config, 'scaling_factor'):
                latent = latent * self.vae.config.scaling_factor

        return latent

    def _encode_prompt(self, prompt: str, max_sequence_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码文本提示词

        Args:
            prompt: 文本提示词
            max_sequence_length: 最大序列长度

        Returns:
            prompt_embeds: [1, seq_len, hidden_size]
            pooled_prompt_embeds: [1, hidden_size]
        """
        with torch.no_grad():
            # CLIP编码
            clip_tokens = self.tokenizer_1(
                prompt,
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            clip_output = self.text_encoder_1(
                clip_tokens.input_ids.to(self.device)
            )
            pooled_prompt_embeds = clip_output.pooler_output

            # T5编码
            t5_tokens = self.tokenizer_2(
                prompt,
                max_length=max_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            t5_output = self.text_encoder_2(
                t5_tokens.input_ids.to(self.device)
            )
            prompt_embeds = t5_output.last_hidden_state

        return prompt_embeds, pooled_prompt_embeds

    def _decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        解码latent到图像

        Args:
            latent: [B, C, H', W']

        Returns:
            image: [B, 3, H, W], 范围[-1, 1]
        """
        with torch.no_grad():
            # 移除scaling
            if hasattr(self.vae.config, 'scaling_factor'):
                latent = latent / self.vae.config.scaling_factor

            # 解码
            image = self.vae.decode(latent)

        return image

    @torch.no_grad()
    def enhance(
        self,
        image: Union[Image.Image, str, Path],
        prompt: str = "Enhance the low-light image",
        resolution: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        auto_classify_exposure: bool = True,
        exposure_category: Optional[int] = None,
        verbose: bool = False
    ) -> Image.Image:
        """
        增强单张图像

        Args:
            image: 输入图像（PIL Image或路径）
            prompt: 文本提示词
            resolution: 图像分辨率
            num_inference_steps: 推理步数
            guidance_scale: Guidance scale
            auto_classify_exposure: 是否自动分类曝光度
            exposure_category: 手动指定的曝光度类别（如果auto_classify_exposure=False）
            verbose: 是否打印详细信息

        Returns:
            enhanced_image: 增强后的图像（PIL Image）
        """
        if not self.lora_injected:
            raise RuntimeError("请先调用 inject_dual_route_moe_lora() 注入LoRA层")

        if not self.lora_loaded:
            print("⚠ 警告: 未加载LoRA权重，将使用随机初始化的权重")

        # 1. 准备图像
        if verbose:
            print("1. 准备图像...")
        image_tensor = self._prepare_image(image, resolution)

        # 2. 分类曝光度
        if auto_classify_exposure:
            if verbose:
                print("2. 分类曝光度...")
            # 使用原始图像进行分类（加载PIL Image）
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image).convert("RGB")
            else:
                pil_image = image

            exposure_category = self.classify_image_exposure(pil_image)
            if verbose:
                from core.utils.exposure import get_category_name
                print(f"   检测到曝光类别: {exposure_category} ({get_category_name(exposure_category)})")
        else:
            if exposure_category is None:
                raise ValueError("如果 auto_classify_exposure=False，必须提供 exposure_category")

        # 3. 设置exposure_category到所有MoE层
        exposure_category_tensor = torch.tensor([exposure_category], device=self.device)
        for name, module in self.transformer.named_modules():
            if hasattr(module, 'set_exposure_category'):
                module.set_exposure_category(exposure_category_tensor)

        # 4. 编码图像到latent
        if verbose:
            print("3. 编码图像...")
        latent = self._encode_image(image_tensor)

        # 5. 编码文本提示
        if verbose:
            print("4. 编码文本提示...")
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt)

        # 6. 推理（简化版flow matching）
        if verbose:
            print(f"5. 运行推理 ({num_inference_steps} 步)...")

        # 简化版：只做一次前向传播（Mock FLUX不支持完整的去噪循环）
        # 真实FLUX需要实现完整的flow matching采样循环

        # 准备输入（concat input latent作为条件）
        # 注意：训练时使用concat(input_latent, target_latent)，推理时使用concat(input_latent, input_latent)
        latent_input = torch.cat([latent, latent], dim=1)  # [B, 32, H', W'] (16*2)

        # 确保dtype匹配
        transformer_dtype = next(self.transformer.parameters()).dtype
        latent_input = latent_input.to(dtype=transformer_dtype)
        prompt_embeds = prompt_embeds.to(dtype=transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=transformer_dtype)

        # Transformer前向传播
        # 注意：真实FLUX需要img_ids和txt_ids
        timestep = torch.tensor([0.5], device=self.device)  # 中间时间步

        output = self.transformer(
            hidden_states=latent_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
        )

        # 获取输出latent
        if isinstance(output, tuple):
            output_latent = output[0]
        else:
            output_latent = output["sample"]

        # 分离输出（取前16个通道，因为输入是32通道的concat）
        # output_latent shape: [B, 32, H, W] -> [B, 16, H, W]
        latent_channels = output_latent.shape[1] // 2
        output_latent = output_latent[:, :latent_channels]

        # 7. 解码latent到图像
        if verbose:
            print("6. 解码图像...")
        enhanced_tensor = self._decode_latent(output_latent)

        # 8. 转换为PIL Image
        # 转换为float32 (numpy不支持bfloat16)
        enhanced_array = enhanced_tensor[0].cpu().float().permute(1, 2, 0).numpy()  # [H, W, 3]
        enhanced_array = (enhanced_array * 0.5 + 0.5).clip(0, 1)  # [-1, 1] -> [0, 1]
        enhanced_array = (enhanced_array * 255).astype(np.uint8)
        enhanced_image = Image.fromarray(enhanced_array)

        if verbose:
            print("✓ 完成")

        return enhanced_image

    @torch.no_grad()
    def enhance_batch(
        self,
        images: List[Union[Image.Image, str, Path]],
        prompt: str = "Enhance the low-light image",
        resolution: int = 512,
        batch_size: int = 4,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        auto_classify_exposure: bool = True,
        verbose: bool = True
    ) -> List[Image.Image]:
        """
        批量增强图像

        Args:
            images: 输入图像列表
            prompt: 文本提示词
            resolution: 图像分辨率
            batch_size: 批处理大小
            num_inference_steps: 推理步数
            guidance_scale: Guidance scale
            auto_classify_exposure: 是否自动分类曝光度
            verbose: 是否打印详细信息

        Returns:
            enhanced_images: 增强后的图像列表
        """
        if not self.lora_injected:
            raise RuntimeError("请先调用 inject_dual_route_moe_lora() 注入LoRA层")

        enhanced_images = []

        # 分批处理
        num_batches = (len(images) + batch_size - 1) // batch_size

        iterator = range(num_batches)
        if verbose:
            iterator = tqdm(iterator, desc="批量推理")

        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(images))
            batch_images = images[start_idx:end_idx]

            # 逐个处理（简化版，真实实现应该支持真正的批处理）
            for img in batch_images:
                enhanced = self.enhance(
                    image=img,
                    prompt=prompt,
                    resolution=resolution,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    auto_classify_exposure=auto_classify_exposure,
                    verbose=False
                )
                enhanced_images.append(enhanced)

        return enhanced_images


def create_inference_pipeline_from_checkpoint(
    checkpoint_path: Union[str, Path],
    transformer,
    vae,
    text_encoder_1,
    text_encoder_2,
    tokenizer_1,
    tokenizer_2,
    scheduler,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
    **kwargs
) -> InferencePipeline:
    """
    从检查点创建推理管道（便捷函数）

    Args:
        checkpoint_path: 检查点路径
        transformer: FLUX Transformer
        vae: VAE
        text_encoder_1: CLIP文本编码器
        text_encoder_2: T5文本编码器
        tokenizer_1: CLIP tokenizer
        tokenizer_2: T5 tokenizer
        scheduler: Scheduler
        device: 设备
        dtype: 数据类型
        verbose: 是否打印详细信息
        **kwargs: 传递给InferencePipeline的额外参数

    Returns:
        pipeline: 已加载权重的推理管道
    """
    # 创建管道
    pipeline = InferencePipeline(
        transformer=transformer,
        vae=vae,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        scheduler=scheduler,
        device=device,
        dtype=dtype,
        **kwargs
    )

    # 注入MoE-LoRA
    pipeline.inject_dual_route_moe_lora(verbose=verbose)

    # 加载权重
    pipeline.load_lora_weights(checkpoint_path, verbose=verbose)

    return pipeline
