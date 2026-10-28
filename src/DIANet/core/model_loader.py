# -*- coding: utf-8 -*-
"""
真实FLUX模型加载器
========================================

功能：
1. 加载FLUX.1-dev模型（使用diffusers FluxPipeline）
2. 加载PhotoDoodle预训练LoRA权重
3. 支持权重融合(fuse)到基础模型
4. 集成显存优化策略
5. 支持保存融合后的模型

基于diffusers库的FluxPipeline实现
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from loguru import logger

try:
    from diffusers import FluxPipeline
    from diffusers.models import FluxTransformer2DModel, AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
except ImportError as e:
    logger.error(f"Failed to import diffusers: {e}")
    logger.error("Please install: pip install diffusers transformers accelerate")
    raise


class RealFluxModelLoader:
    """真实FLUX模型加载器 - 基于diffusers FluxPipeline"""

    def __init__(self, config):
        """
        初始化模型加载器

        参数：
            config: 配置对象，包含以下字段：
                - model_path: FLUX模型路径（本地路径或HuggingFace model ID）
                - device: 设备
                - dtype: 数据类型（通常为bfloat16）
                - photodoodle: PhotoDoodle配置
                    - enabled: 是否启用
                    - lora_path: LoRA权重路径
                    - fuse_before_training: 是否在训练前融合
                    - save_fused_model: 是否保存融合后的模型
                    - fused_model_path: 融合模型保存路径
        """
        self.config = config
        self.model_path = getattr(config, 'model_path', 'black-forest-labs/FLUX.1-dev')
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

    def load_pipeline(self, enable_cpu_offload: bool = False) -> FluxPipeline:
        """
        加载FLUX Pipeline

        参数：
            enable_cpu_offload: 是否启用CPU offload

        返回：
            FluxPipeline实例
        """
        logger.info(f"[FLUX加载] 开始加载FLUX.1-dev模型")
        logger.info(f"[FLUX加载] 模型路径: {self.model_path}")
        logger.info(f"[FLUX加载] 数据类型: {self.dtype}")

        # 加载Pipeline
        try:
            self.pipeline = FluxPipeline.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
            )

            logger.info(f"[FLUX加载] ✓ Pipeline加载成功")

            # CPU Offload（节省显存）
            if enable_cpu_offload:
                self.pipeline.enable_model_cpu_offload()
                logger.info(f"[FLUX加载] ✓ 已启用Model CPU Offload")

            return self.pipeline

        except Exception as e:
            logger.error(f"[FLUX加载] 加载失败: {e}")
            raise

    def load_photodoodle_lora(self, lora_path: Optional[str] = None) -> FluxPipeline:
        """
        加载PhotoDoodle预训练LoRA权重

        参数：
            lora_path: LoRA权重路径（可选，覆盖配置）

        返回：
            加载了LoRA的Pipeline
        """
        if self.pipeline is None:
            raise ValueError("请先调用load_pipeline()加载基础模型")

        # 确定LoRA路径
        if lora_path is None:
            if self.photodoodle_config is None:
                raise ValueError("未提供PhotoDoodle配置")
            lora_path = getattr(self.photodoodle_config, 'lora_path', None)

        if lora_path is None:
            raise ValueError("未指定LoRA权重路径")

        logger.info(f"[PhotoDoodle] 开始加载预训练LoRA权重")
        logger.info(f"[PhotoDoodle] 权重路径: {lora_path}")

        try:
            # 加载LoRA权重
            self.pipeline.load_lora_weights(lora_path)
            logger.info(f"[PhotoDoodle] ✓ LoRA权重加载成功")

            return self.pipeline

        except Exception as e:
            logger.error(f"[PhotoDoodle] LoRA加载失败: {e}")
            raise

    def fuse_lora(self, lora_scale: float = 1.0, save_fused: bool = False) -> FluxPipeline:
        """
        融合LoRA权重到基础模型

        参数：
            lora_scale: LoRA权重缩放因子（默认1.0）
            save_fused: 是否保存融合后的模型

        返回：
            融合后的Pipeline
        """
        if self.pipeline is None:
            raise ValueError("请先加载模型和LoRA权重")

        logger.info(f"[PhotoDoodle] 开始融合LoRA权重到基础模型")
        logger.info(f"[PhotoDoodle] LoRA缩放因子: {lora_scale}")

        try:
            # 融合LoRA
            self.pipeline.fuse_lora(lora_scale=lora_scale)
            logger.info(f"[PhotoDoodle] ✓ LoRA权重已融合")

            # 卸载LoRA适配器（已融合到权重中）
            self.pipeline.unload_lora_weights()
            logger.info(f"[PhotoDoodle] ✓ LoRA适配器已卸载")

            # 保存融合后的模型
            if save_fused and self.photodoodle_config:
                fused_path = getattr(self.photodoodle_config, 'fused_model_path', './fused_flux_photodoodle')
                self.save_fused_model(fused_path)

            return self.pipeline

        except Exception as e:
            logger.error(f"[PhotoDoodle] LoRA融合失败: {e}")
            raise

    def save_fused_model(self, output_path: str):
        """
        保存融合后的模型

        参数：
            output_path: 输出路径
        """
        if self.pipeline is None:
            raise ValueError("Pipeline未初始化")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"[保存模型] 正在保存融合后的模型...")
        logger.info(f"[保存模型] 输出路径: {output_path}")

        try:
            # 保存整个Pipeline
            self.pipeline.save_pretrained(output_path)
            logger.info(f"[保存模型] ✓ 融合模型已保存")

        except Exception as e:
            logger.error(f"[保存模型] 保存失败: {e}")
            raise

    def extract_components(self) -> Tuple:
        """
        提取Pipeline中的各个组件（用于训练）

        返回：
            (transformer, vae, text_encoder_1, text_encoder_2,
             tokenizer_1, tokenizer_2, scheduler)
        """
        if self.pipeline is None:
            raise ValueError("Pipeline未初始化，请先调用load_pipeline()")

        logger.info(f"[组件提取] 正在提取Pipeline组件...")

        transformer = self.pipeline.transformer
        vae = self.pipeline.vae
        text_encoder_1 = self.pipeline.text_encoder
        text_encoder_2 = self.pipeline.text_encoder_2
        tokenizer_1 = self.pipeline.tokenizer
        tokenizer_2 = self.pipeline.tokenizer_2
        scheduler = self.pipeline.scheduler

        logger.info(f"[组件提取] ✓ 所有组件提取成功")

        return (
            transformer,
            vae,
            text_encoder_1,
            text_encoder_2,
            tokenizer_1,
            tokenizer_2,
            scheduler
        )

    def load_all(
        self,
        load_photodoodle: bool = True,
        fuse_photodoodle: bool = True,
        enable_cpu_offload: bool = True
    ) -> Tuple:
        """
        一键加载所有模型组件（包括PhotoDoodle预训练）

        参数：
            load_photodoodle: 是否加载PhotoDoodle LoRA
            fuse_photodoodle: 是否融合PhotoDoodle LoRA
            enable_cpu_offload: 是否启用CPU offload

        返回：
            (transformer, vae, text_encoder_1, text_encoder_2,
             tokenizer_1, tokenizer_2, scheduler)
        """
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

        logger.info(f"\n{'='*80}")
        logger.info(f"✓ FLUX.1-dev模型加载完成")
        logger.info(f"{'='*80}\n")

        return components

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        返回：
            模型信息字典
        """
        if self.pipeline is None:
            return {"loaded": False}

        info = {
            "loaded": True,
            "model_path": self.model_path,
            "device": self.device,
            "dtype": str(self.dtype),
        }

        # Transformer信息
        if hasattr(self.pipeline, 'transformer'):
            transformer = self.pipeline.transformer
            num_params = sum(p.numel() for p in transformer.parameters())
            info["transformer"] = {
                "num_parameters": num_params,
                "num_parameters_m": num_params / 1e6,
            }

        # PhotoDoodle信息
        if self.photodoodle_config:
            info["photodoodle"] = {
                "enabled": getattr(self.photodoodle_config, 'enabled', False),
                "lora_path": getattr(self.photodoodle_config, 'lora_path', None),
                "fused": getattr(self.photodoodle_config, 'fuse_before_training', False),
            }

        return info

    def print_model_info(self):
        """打印模型信息"""
        info = self.get_model_info()

        print("\n" + "=" * 80)
        print("FLUX模型信息")
        print("=" * 80)

        if not info.get("loaded"):
            print("模型未加载")
            return

        print(f"模型路径: {info['model_path']}")
        print(f"设备: {info['device']}")
        print(f"数据类型: {info['dtype']}")

        if "transformer" in info:
            print(f"\n[Transformer]")
            print(f"  参数量: {info['transformer']['num_parameters_m']:.2f}M")

        if "photodoodle" in info:
            print(f"\n[PhotoDoodle]")
            print(f"  启用: {'✓' if info['photodoodle']['enabled'] else '✗'}")
            if info['photodoodle']['enabled']:
                print(f"  LoRA路径: {info['photodoodle']['lora_path']}")
                print(f"  已融合: {'✓' if info['photodoodle']['fused'] else '✗'}")

        print("=" * 80 + "\n")


def create_real_flux_loader(config) -> RealFluxModelLoader:
    """
    工厂函数：创建真实FLUX模型加载器

    参数：
        config: 配置对象

    返回：
        RealFluxModelLoader实例
    """
    return RealFluxModelLoader(config)
