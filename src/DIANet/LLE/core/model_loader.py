"""
模型加载器 - 负责加载FLUX模型的所有组件

MVP版本：
- 加载Flux基础模型（transformer, vae, text encoders, scheduler）
- 不包含PhotoDoodle集成（留给阶段2）
- 不包含量化和offload（留给阶段6）
- 支持torch_dtype和device配置

FLUX模型组件：
- FluxTransformer2DModel: 条件Transformer用于去噪
- AutoencoderKL: VAE用于编码/解码图像
- CLIPTextModel: CLIP文本编码器
- T5EncoderModel: T5文本编码器
- FlowMatchEulerDiscreteScheduler: 欧拉离散调度器
"""

import torch
from typing import Tuple, Optional
from pathlib import Path


class ModelLoader:
    """
    模型加载器 - 负责加载FLUX模型的所有组件

    MVP版本：简单实现，不包含量化和offload
    """

    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        verbose: bool = True,
    ):
        """
        Args:
            model_path: 模型路径（本地路径或HuggingFace模型ID）
                例如: "black-forest-labs/FLUX.1-dev"
            torch_dtype: 模型精度（bfloat16, float16, float32）
            device: 目标设备（"cuda", "cpu"）
            verbose: 是否打印详细信息
        """
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.device = device
        self.verbose = verbose

        # 检查设备可用性
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available but device='cuda' was specified")

    def load_all(self) -> Tuple:
        """
        加载所有FLUX模型组件

        Returns:
            (transformer, vae, text_encoder_1, text_encoder_2,
             tokenizer_1, tokenizer_2, scheduler)
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("开始加载FLUX模型")
            print("=" * 60)
            print(f"模型路径: {self.model_path}")
            print(f"精度: {self.torch_dtype}")
            print(f"设备: {self.device}")
            print("=" * 60 + "\n")

        # 延迟导入（避免在模块导入时就需要diffusers）
        from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
        from transformers import CLIPTokenizer, CLIPTextModel, T5TokenizerFast, T5EncoderModel

        # 1. 加载Transformer
        if self.verbose:
            print("[1/7] 加载 FluxTransformer2DModel...")
        transformer = FluxTransformer2DModel.from_pretrained(
            self.model_path,
            subfolder="transformer",
            torch_dtype=self.torch_dtype,
        )
        transformer = transformer.to(self.device)
        if self.verbose:
            print(f"      ✓ Transformer加载完成 ({self._count_parameters(transformer):,} 参数)")

        # 2. 加载VAE
        if self.verbose:
            print("[2/7] 加载 AutoencoderKL (VAE)...")
        vae = AutoencoderKL.from_pretrained(
            self.model_path,
            subfolder="vae",
            torch_dtype=self.torch_dtype,
        )
        vae = vae.to(self.device)
        if self.verbose:
            print(f"      ✓ VAE加载完成 ({self._count_parameters(vae):,} 参数)")

        # 3. 加载CLIP Text Encoder
        if self.verbose:
            print("[3/7] 加载 CLIPTextModel (Text Encoder 1)...")
        text_encoder_1 = CLIPTextModel.from_pretrained(
            self.model_path,
            subfolder="text_encoder",
            torch_dtype=self.torch_dtype,
        )
        text_encoder_1 = text_encoder_1.to(self.device)
        if self.verbose:
            print(f"      ✓ CLIP Text Encoder加载完成 ({self._count_parameters(text_encoder_1):,} 参数)")

        # 4. 加载CLIP Tokenizer
        if self.verbose:
            print("[4/7] 加载 CLIPTokenizer (Tokenizer 1)...")
        tokenizer_1 = CLIPTokenizer.from_pretrained(
            self.model_path,
            subfolder="tokenizer",
        )
        if self.verbose:
            print("      ✓ CLIP Tokenizer加载完成")

        # 5. 加载T5 Text Encoder
        if self.verbose:
            print("[5/7] 加载 T5EncoderModel (Text Encoder 2)...")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            self.model_path,
            subfolder="text_encoder_2",
            torch_dtype=self.torch_dtype,
        )
        text_encoder_2 = text_encoder_2.to(self.device)
        if self.verbose:
            print(f"      ✓ T5 Text Encoder加载完成 ({self._count_parameters(text_encoder_2):,} 参数)")

        # 6. 加载T5 Tokenizer
        if self.verbose:
            print("[6/7] 加载 T5TokenizerFast (Tokenizer 2)...")
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            self.model_path,
            subfolder="tokenizer_2",
        )
        if self.verbose:
            print("      ✓ T5 Tokenizer加载完成")

        # 7. 加载Scheduler
        if self.verbose:
            print("[7/7] 加载 FlowMatchEulerDiscreteScheduler...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_path,
            subfolder="scheduler",
        )
        if self.verbose:
            print("      ✓ Scheduler加载完成")

        if self.verbose:
            print("\n" + "=" * 60)
            print("✅ 所有组件加载完成!")
            total_params = (
                self._count_parameters(transformer) +
                self._count_parameters(vae) +
                self._count_parameters(text_encoder_1) +
                self._count_parameters(text_encoder_2)
            )
            print(f"总参数量: {total_params:,}")
            print("=" * 60 + "\n")

        return transformer, vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, scheduler

    def load_transformer_only(self):
        """
        仅加载Transformer（用于测试或特殊场景）

        Returns:
            transformer: FluxTransformer2DModel
        """
        from diffusers import FluxTransformer2DModel

        if self.verbose:
            print(f"加载Transformer: {self.model_path}")

        transformer = FluxTransformer2DModel.from_pretrained(
            self.model_path,
            subfolder="transformer",
            torch_dtype=self.torch_dtype,
        )
        transformer = transformer.to(self.device)

        if self.verbose:
            print(f"✓ Transformer加载完成 ({self._count_parameters(transformer):,} 参数)")

        return transformer

    def _count_parameters(self, model) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_path={self.model_path}, "
            f"torch_dtype={self.torch_dtype}, "
            f"device={self.device})"
        )


def create_model_loader_from_config(config) -> ModelLoader:
    """
    从TrainingConfig创建ModelLoader

    Args:
        config: TrainingConfig对象

    Returns:
        ModelLoader实例
    """
    # 将dtype字符串转换为torch.dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    torch_dtype = dtype_map.get(config.model.dtype, torch.bfloat16)

    return ModelLoader(
        model_path=config.model.base_model,
        torch_dtype=torch_dtype,
        device=config.device,
        verbose=True,
    )


# ========================================
# RealFluxModelLoader - 支持PhotoDoodle
# ========================================

from typing import Dict, Any

try:
    from diffusers import FluxPipeline
    from diffusers.models import FluxTransformer2DModel, AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from loguru import logger
except ImportError as e:
    print(f"Warning: Cannot import diffusers/loguru: {e}")
    logger = None


class RealFluxModelLoader:
    """真实FLUX模型加载器 - 基于diffusers FluxPipeline，支持PhotoDoodle"""

    def __init__(self, config):
        """
        初始化模型加载器

        参数：
            config: 配置对象，包含以下字段：
                - model_path: FLUX模型路径（本地路径或HuggingFace model ID）
                - device: 设备
                - dtype: 数据类型（通常为bfloat16）
                - photodoodle: PhotoDoodle配置
        """
        self.config = config
        # 优先使用model_path，如果没有则使用base_model
        if hasattr(config, 'model_path') and config.model_path:
            self.model_path = config.model_path
        elif hasattr(config, 'base_model') and config.base_model:
            self.model_path = config.base_model
        else:
            self.model_path = 'black-forest-labs/FLUX.1-dev'

        self.device = getattr(config, 'device', 'cuda')
        self.dtype = self._get_dtype()

        # PhotoDoodle配置
        self.photodoodle_config = getattr(config, 'photodoodle', None)

        # Pipeline
        self.pipeline: Optional[FluxPipeline] = None

    def _get_dtype(self) -> torch.dtype:
        """获取数据类型"""
        dtype_str = getattr(self.config, 'dtype', 'bfloat16')
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.bfloat16)

    def load_pipeline(self, enable_cpu_offload: bool = False):
        """加载FLUX Pipeline"""
        if logger:
            logger.info(f"[FLUX加载] 开始加载FLUX.1-dev模型")
            logger.info(f"[FLUX加载] 模型路径: {self.model_path}")

        self.pipeline = FluxPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
        )

        if enable_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
            if logger:
                logger.info(f"[FLUX加载] ✓ 已启用Model CPU Offload")

        return self.pipeline

    def load_photodoodle_lora(self, lora_path: Optional[str] = None):
        """加载PhotoDoodle预训练LoRA权重"""
        if self.pipeline is None:
            raise ValueError("请先调用load_pipeline()加载基础模型")

        if lora_path is None:
            if self.photodoodle_config is None:
                raise ValueError("未提供PhotoDoodle配置")
            lora_path = getattr(self.photodoodle_config, 'lora_path', None)

        if lora_path is None:
            raise ValueError("未指定LoRA权重路径")

        if logger:
            logger.info(f"[PhotoDoodle] 开始加载预训练LoRA权重")
            logger.info(f"[PhotoDoodle] 权重路径: {lora_path}")

        self.pipeline.load_lora_weights(lora_path)
        return self.pipeline

    def fuse_lora(self, lora_scale: float = 1.0, save_fused: bool = False):
        """融合LoRA权重到基础模型"""
        if self.pipeline is None:
            raise ValueError("请先加载模型和LoRA权重")

        if logger:
            logger.info(f"[PhotoDoodle] 开始融合LoRA权重到基础模型")
            logger.info(f"[PhotoDoodle] LoRA缩放因子: {lora_scale}")

        self.pipeline.fuse_lora(lora_scale=lora_scale)
        self.pipeline.unload_lora_weights()

        if logger:
            logger.info(f"[PhotoDoodle] ✓ LoRA权重已融合并卸载")

        return self.pipeline

    def extract_components(self) -> Tuple:
        """提取Pipeline中的各个组件（用于训练）"""
        if self.pipeline is None:
            raise ValueError("Pipeline未初始化，请先调用load_pipeline()")

        if logger:
            logger.info(f"[组件提取] 正在提取Pipeline组件...")

        return (
            self.pipeline.transformer,
            self.pipeline.vae,
            self.pipeline.text_encoder,
            self.pipeline.text_encoder_2,
            self.pipeline.tokenizer,
            self.pipeline.tokenizer_2,
            self.pipeline.scheduler
        )

    def load_all(
        self,
        load_photodoodle: bool = True,
        fuse_photodoodle: bool = True,
        enable_cpu_offload: bool = True
    ) -> Tuple:
        """一键加载所有模型组件（包括PhotoDoodle预训练）"""
        if logger:
            logger.info(f"\n{'='*80}")
            logger.info(f"开始加载FLUX.1-dev模型（含PhotoDoodle预训练）")
            logger.info(f"{'='*80}\n")

        # 1. 加载基础Pipeline
        self.load_pipeline(enable_cpu_offload=enable_cpu_offload)

        # 2. 加载PhotoDoodle LoRA（如果启用）
        if load_photodoodle and self.photodoodle_config:
            if getattr(self.photodoodle_config, 'enabled', False):
                self.load_photodoodle_lora()

                # 3. 融合LoRA（如果启用）
                if fuse_photodoodle and getattr(self.photodoodle_config, 'fuse_before_training', False):
                    lora_scale = getattr(self.photodoodle_config, 'lora_scale', 1.0)
                    save_fused = getattr(self.photodoodle_config, 'save_fused_model', False)
                    self.fuse_lora(lora_scale=lora_scale, save_fused=save_fused)

        # 4. 提取组件
        components = self.extract_components()

        if logger:
            logger.info(f"\n{'='*80}")
            logger.info(f"✓ FLUX.1-dev模型加载完成")
            logger.info(f"{'='*80}\n")

        return components


def create_real_flux_loader(config) -> RealFluxModelLoader:
    """
    工厂函数：创建真实FLUX模型加载器

    参数：
        config: 配置对象

    返回：
        RealFluxModelLoader实例
    """
    return RealFluxModelLoader(config)
