# -*- coding: utf-8 -*-
"""
配置加载模块
"""
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class TrainingConfig:
    """训练配置类 - 从 YAML 加载所有参数"""
    
    # 从 YAML 加载
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """从 YAML 文件加载配置"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    def __init__(self, config_dict: Dict[str, Any]):
        """初始化配置"""
        self.config_dict = config_dict
        self._load_config()
    
    def _load_config(self):
        """加载配置到属性"""
        cfg = self.config_dict
        
        # 基础配置
        self.job_name = cfg.get("job_name", "flux_lora_training")
        self.device = cfg.get("device", "cuda")
        self.gpu_id = cfg.get("gpu_id", 0)
        self.seed = cfg.get("seed", 42)
        
        # 模型配置
        model_cfg = cfg.get("model", {})
        self.model_path = model_cfg.get("name_or_path")
        self.model_dtype = model_cfg.get("dtype", "bfloat16")
        self.quantize = model_cfg.get("quantize", False)
        self.quantize_te = model_cfg.get("quantize_te", False)
        self.qtype = model_cfg.get("qtype", "int8")
        
        # LoRA 配置
        lora_cfg = cfg.get("lora", {})
        self.lora_rank = lora_cfg.get("rank", 16)
        self.lora_alpha = lora_cfg.get("alpha", 16)
        self.lora_dropout = lora_cfg.get("dropout", 0.0)
        self.lora_target_modules = lora_cfg.get("target_modules", None)
        self.lora_variant = lora_cfg.get("variant", "auto")  # auto: 按 module_modes 推断
        self.lora_variant_config = lora_cfg.get("variant_config", {}) or {}
        self.lora_module_modes = lora_cfg.get("module_modes", {}) or {}
        
        # 数据集配置
        dataset_cfg = cfg.get("dataset", {})
        self.input_image_dir = dataset_cfg.get("input_image_dir")
        self.target_image_dir = dataset_cfg.get("target_image_dir")
        self.prompt = dataset_cfg.get("prompt", "")
        self.use_prompt_file = dataset_cfg.get("use_prompt_file", False)
        self.prompt_ext = dataset_cfg.get("prompt_ext", "txt")
        self.resolution = dataset_cfg.get("resolution", 1024)
        self.center_crop = dataset_cfg.get("center_crop", False)
        self.random_flip = dataset_cfg.get("random_flip", False)
        self.num_workers = dataset_cfg.get("num_workers", 0)
        self.pin_memory = dataset_cfg.get("pin_memory", True)
        
        # 训练配置
        train_cfg = cfg.get("train", {})
        self.num_epochs = train_cfg.get("num_epochs", 10)
        self.batch_size = train_cfg.get("batch_size", 1)
        self.gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)
        self.learning_rate = train_cfg.get("learning_rate", 1e-4)
        self.lr_scheduler = train_cfg.get("lr_scheduler", "constant")
        self.lr_warmup_steps = train_cfg.get("lr_warmup_steps", 0)
        self.optimizer_type = train_cfg.get("optimizer", "adamw")
        self.adam_beta1 = train_cfg.get("adam_beta1", 0.9)
        self.adam_beta2 = train_cfg.get("adam_beta2", 0.999)
        self.adam_epsilon = train_cfg.get("adam_epsilon", 1e-8)
        self.adam_weight_decay = train_cfg.get("adam_weight_decay", 0.01)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.gradient_checkpointing = train_cfg.get("gradient_checkpointing", False)
        self.mixed_precision = train_cfg.get("mixed_precision", "no")
        self.max_sequence_length = train_cfg.get("max_sequence_length", 512)
        self.cfg_scale = train_cfg.get("cfg_scale", 1.0)  # guidance embedding scale，用于 flux kontext
        
        # 显存优化配置
        mem_opt_cfg = cfg.get("memory_optimization", {})
        self.vae_offload = mem_opt_cfg.get("vae_offload", False)
        self.cache_latents = mem_opt_cfg.get("cache_latents", False)
        self.cache_to_disk = mem_opt_cfg.get("cache_to_disk", False)
        self.cache_dir = mem_opt_cfg.get("cache_dir", "./cache")
        self.te_offload = mem_opt_cfg.get("te_offload", False)
        
        # 保存配置
        save_cfg = cfg.get("save", {})
        self.output_dir = save_cfg.get("output_dir", "./lora_output")
        self.save_every_n_epochs = save_cfg.get("save_every_n_epochs", 5)
        self.save_total_limit = save_cfg.get("save_total_limit", 3)
        self.save_dtype = save_cfg.get("save_dtype", "float16")
        
        # 日志配置
        log_cfg = cfg.get("logging", {})
        self.log_level = log_cfg.get("log_level", "INFO")
        self.log_every_n_steps = log_cfg.get("log_every_n_steps", 10)
        self.use_tensorboard = log_cfg.get("use_tensorboard", False)
        self.tensorboard_dir = log_cfg.get("tensorboard_dir", "./logs")
        
        # 验证配置
        val_cfg = cfg.get("validation", {})
        self.validation_enabled = val_cfg.get("enabled", False)
        self.validation_every_n_epochs = val_cfg.get("every_n_epochs", 5)
        self.validation_save_dir = val_cfg.get("save_dir", "./lora_output/samples")
        self.validation_seed = val_cfg.get("seed", None)
        self.validation_width = val_cfg.get("width", None)
        self.validation_height = val_cfg.get("height", None)
        self.validation_prompts = val_cfg.get("validation_prompts", [])
        self.validation_image = val_cfg.get("validation_image", None)
        self.num_inference_steps = val_cfg.get("num_inference_steps", 28)
        self.guidance_scale = val_cfg.get("guidance_scale", 2.5)
        self.validation_external_infer = val_cfg.get("external_infer", False)
        self.validation_offload_mode = val_cfg.get("offload_mode", "sequential")  # sequential/model/none
        self.validation_infer_dtype = val_cfg.get("infer_dtype", "float16")
    
    def get_dtype(self):
        """获取 PyTorch dtype"""
        import torch
        if self.model_dtype == "bfloat16":
            return torch.bfloat16
        elif self.model_dtype == "float16":
            return torch.float16
        else:
            return torch.float32
