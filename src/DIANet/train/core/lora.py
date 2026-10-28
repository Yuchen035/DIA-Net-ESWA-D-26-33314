# -*- coding: utf-8 -*-
"""LoRA 配置模块 - 支持变体扩展"""
import torch
from peft import LoraConfig

class LoRAConfigurator:
    """LoRA 配置器 - 支持标准 LoRA 和变体（如 MoE-LoRA）"""
    
    def __init__(self, config):
        self.config = config
        self.logger = None
    
    def set_logger(self, logger):
        self.logger = logger
    
    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _prepare_targets_and_modes(self, default_targets):
        """
        根据配置生成 target_modules 和 module_modes
        module_modes: {pattern: "moe"|"lora"|"freeze"}
        """
        # 归一化模式
        raw_modes = self.config.lora_module_modes or {}
        module_modes = {}
        valid_modes = {"moe", "lora", "freeze"}
        for k, v in raw_modes.items():
            if v is None:
                continue
            mode = str(v).lower()
            if mode not in valid_modes:
                raise ValueError(f"module_modes[{k}] 非法模式：{v}，仅支持 {valid_modes}")
            module_modes[k] = mode

        # 如果指定了 module_modes，则 target_modules 取其中非 freeze 的键
        if module_modes:
            moe_targets = [k for k, mode in module_modes.items() if mode == "moe"]
            lora_targets = [k for k, mode in module_modes.items() if mode == "lora"]
            freeze_targets = [k for k, mode in module_modes.items() if mode == "freeze"]
            # 实际传给 PEFT 的 target_modules 仍使用默认/显式列表，避免通配符导致“不存在模块”错误
            # module_modes 里的模式只用于前向匹配，不再直接用作 target_modules
            target_modules = self.config.lora_target_modules or default_targets

            self._log(f"[LoRA 配置] 使用 module_modes 精准控制，冻结模块：{', '.join(freeze_targets) if freeze_targets else '无'}")
        else:
            target_modules = self.config.lora_target_modules or default_targets
            module_modes = {}
            freeze_targets = []

        return target_modules, module_modes, freeze_targets
    
    def apply_lora(self, transformer):
        """为 Transformer 应用 LoRA"""
        self._log(f"\n[LoRA 配置] ========== 开始配置 LoRA ==========")
        # 自动推断变体：若包含 moe 模式，则选择 moe，否则 standard
        module_modes = self.config.lora_module_modes or {}
        auto_variant = "moe" if any(mode == "moe" for mode in module_modes.values()) else "standard"
        variant = self.config.lora_variant or "auto"
        if variant == "auto":
            variant = auto_variant

        self._log(f"[LoRA 配置] 变体：{variant}")
        self._log(f"[LoRA 配置] LoRA 秩：{self.config.lora_rank}")
        self._log(f"[LoRA 配置] LoRA alpha：{self.config.lora_alpha}")
        
        # 根据变体类型选择配置方法
        if variant == "standard":
            return self._apply_standard_lora(transformer)
        elif variant == "moe":
            return self._apply_moe_lora(transformer)
        else:
            raise ValueError(f"不支持的 LoRA 变体：{variant}")
    
    def _apply_standard_lora(self, transformer):
        """应用标准 LoRA"""
        # 目标模块
        target_modules, module_modes, _ = self._prepare_targets_and_modes(
            default_targets=[
                "attn.to_k",
                "attn.to_q",
                "attn.to_v",
                "attn.to_out.0",
            ]
        )

        self._log(f"[LoRA 配置] 目标模块：{', '.join(target_modules)}")
        if module_modes:
            detail = ", ".join([f"{k}:{v}" for k, v in module_modes.items()])
            self._log(f"[LoRA 配置] module_modes：{detail}")

        variant_cfg = self.config.lora_variant_config or {}
        num_experts = variant_cfg.get("num_experts", 6)
        expert_rank = variant_cfg.get("expert_rank", 32)
        expert_alpha = variant_cfg.get("expert_alpha", 32)
        top_k = variant_cfg.get("top_k", 1)
        blc_alpha = variant_cfg.get("blc_alpha", 0.0)
        blc_weight = variant_cfg.get("blc_weight", 0.0)
        
        # 创建 LoRA 配置
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
            module_modes=module_modes,
            num_experts=num_experts,
            expert_rank=expert_rank,
            expert_alpha=expert_alpha,
            top_k=top_k,
            blc_alpha=blc_alpha,
            blc_weight=blc_weight,
        )
        
        # 添加 LoRA 适配器
        transformer.add_adapter(lora_config)
        
        # LoRA 权重强制使用 float32（对齐 ai-toolkit：模型可低精度，LoRA 保持 fp32 以保证梯度稳定）
        for name, param in transformer.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(dtype=torch.float32, device=param.device)
                param.requires_grad = True
        self._log("[LoRA 配置] ✓ LoRA 权重已切换为 float32 参与训练")
        
        self._log(f"[LoRA 配置] ✓ 标准 LoRA 配置完成")
        self._log(f"[LoRA 配置] =============================================\n")
        
        return transformer
    
    def _apply_moe_lora(self, transformer):
        """应用 MoE-LoRA"""
        self._log(f"[LoRA 配置] MoE-LoRA 变体配置...")

        variant_cfg = self.config.lora_variant_config or {}
        # 默认同时覆盖注意力和前馈/输出层，MoE 只在 ff.net/proj_out 生效（可被 module_modes 覆盖）
        target_modules, module_modes, freeze_targets = self._prepare_targets_and_modes(
            default_targets=[
                "attn.to_k",
                "attn.to_q",
                "attn.to_v",
                "attn.to_out.0",
                "ff.net.0.proj",
                "ff.net.2",
                "proj_out",
            ]
        )

        num_experts = variant_cfg.get("num_experts", 6)
        expert_rank = variant_cfg.get("expert_rank", 32)
        expert_alpha = variant_cfg.get("expert_alpha", 32)
        top_k = variant_cfg.get("top_k", 1)
        blc_alpha = variant_cfg.get("blc_alpha", 0.0)
        blc_weight = variant_cfg.get("blc_weight", 0.0)

        self._log(f"[LoRA 配置] 目标模块：{', '.join(target_modules)}")
        if module_modes:
            detail = ", ".join([f"{k}:{v}" for k, v in module_modes.items()])
            self._log(f"[LoRA 配置] module_modes：{detail}")
        self._log(
            f"[LoRA 配置] MoE 参数：experts={num_experts}, expert_rank={expert_rank}, "
            f"expert_alpha={expert_alpha}, top_k={top_k}, blc_alpha={blc_alpha}, blc_weight={blc_weight}"
        )

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
            num_experts=num_experts,
            expert_rank=expert_rank,
            expert_alpha=expert_alpha,
            top_k=top_k,
            blc_alpha=blc_alpha,
            blc_weight=blc_weight,
            module_modes=module_modes,
        )

        transformer.add_adapter(lora_config)

        # LoRA 权重保持 float32 训练稳定性
        for name, param in transformer.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(dtype=torch.float32, device=param.device)
                param.requires_grad = True

        self._log(f"[LoRA 配置] ✓ MoE-LoRA 配置完成")
        self._log(f"[LoRA 配置] =============================================\n")

        return transformer
