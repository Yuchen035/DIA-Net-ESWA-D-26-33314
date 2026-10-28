"""
LoRA配置器 - 负责将LoRA注入到模型中

功能：
- 遍历模型的所有模块，根据配置注入LoRA/MoE-LoRA
- 支持fnmatch通配符模式匹配
- 支持细粒度层级控制（lora/moe/freeze）
- 支持混合配置（某些层用LoRA，某些层用MoE-LoRA）
- 完全解耦，不依赖peft库
"""

import torch
import torch.nn as nn
import fnmatch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class LoRAInjectionStats:
    """LoRA注入统计信息"""
    total_modules: int = 0
    lora_injected: int = 0
    moe_injected: int = 0
    frozen: int = 0
    injected_module_names: List[str] = field(default_factory=list)
    moe_module_names: List[str] = field(default_factory=list)


class LoRAConfigurator:
    """
    LoRA配置器 - 负责根据配置向模型注入LoRA/MoE-LoRA

    支持：
    - 标准LoRA注入
    - MoE-LoRA注入（单路由/双路由）
    - 混合配置（部分层用LoRA，部分层用MoE-LoRA）
    - 模块级别mode控制
    """

    def __init__(
        self,
        target_modules: Optional[List[str]] = None,
        module_modes: Optional[Dict[str, str]] = None,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        init_method: str = "kaiming",
        # MoE-LoRA参数
        moe_config: Optional[Dict] = None,
        verbose: bool = True
    ):
        """
        Args:
            target_modules: 目标模块列表（兼容旧API），支持fnmatch通配符
                例如：["single_transformer_blocks.*.attn.to_q"]
                如果提供module_modes，此参数将被忽略
            module_modes: 模块级别的mode配置，支持fnmatch通配符
                例如：{
                    "single_transformer_blocks.*.attn.to_q": "moe",
                    "single_transformer_blocks.*.attn.to_v": "lora",
                    "single_transformer_blocks.*.proj_out": "freeze"
                }
            rank: LoRA的秩
            alpha: LoRA的缩放因子
            dropout: LoRA的dropout率
            init_method: LoRA权重初始化方法
            moe_config: MoE-LoRA配置，例如：
                {
                    "num_experts": 4,
                    "routing_mode": "single",  # "single" or "dual"
                    "router_config": {
                        "top_k": 2,
                        "use_noisy_gating": False
                    }
                }
            verbose: 是否打印详细信息
        """
        # 如果提供了module_modes，使用它；否则使用target_modules（兼容旧API）
        if module_modes is not None:
            self.module_modes = module_modes
            self.target_modules = None
        elif target_modules is not None:
            # 兼容旧API：将target_modules转换为module_modes（全部为lora）
            self.module_modes = {pattern: "lora" for pattern in target_modules}
            self.target_modules = target_modules
        else:
            raise ValueError("必须提供target_modules或module_modes之一")

        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.init_method = init_method
        self.verbose = verbose

        # MoE-LoRA配置
        self.moe_config = moe_config or {
            "num_experts": 4,
            "routing_mode": "single",
            "router_config": {"top_k": 2}
        }

        # 统计信息
        self.stats = LoRAInjectionStats()

    def inject_lora(self, model: nn.Module) -> Tuple[nn.Module, LoRAInjectionStats]:
        """
        将LoRA/MoE-LoRA注入到模型中

        Args:
            model: 要注入LoRA的模型（如Transformer）

        Returns:
            (injected_model, stats): 注入LoRA后的模型和统计信息
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("开始注入LoRA/MoE-LoRA")
            print("=" * 70)
            print(f"模块配置: {len(self.module_modes)} 个模式")
            for pattern, mode in list(self.module_modes.items())[:5]:  # 只显示前5个
                print(f"  {pattern}: {mode}")
            if len(self.module_modes) > 5:
                print(f"  ... 还有 {len(self.module_modes) - 5} 个")
            print(f"\nLoRA参数: rank={self.rank}, alpha={self.alpha}, dropout={self.dropout}")
            if any(mode == "moe" for mode in self.module_modes.values()):
                print(f"MoE配置: num_experts={self.moe_config['num_experts']}, "
                      f"mode={self.moe_config['routing_mode']}")
            print(f"初始化方法: {self.init_method}")
            print("=" * 70 + "\n")

        # 重置统计信息
        self.stats = LoRAInjectionStats()

        # 递归注入LoRA/MoE-LoRA
        self._inject_lora_recursive(model, parent_name="")

        if self.verbose:
            print("\n" + "=" * 70)
            print("LoRA/MoE-LoRA注入完成")
            print("=" * 70)
            print(f"总模块数: {self.stats.total_modules}")
            print(f"注入标准LoRA: {self.stats.lora_injected}")
            print(f"注入MoE-LoRA: {self.stats.moe_injected}")
            print(f"冻结模块: {self.stats.frozen}")
            if self.stats.injected_module_names:
                print(f"\n注入标准LoRA的模块 ({len(self.stats.injected_module_names)}):")
                for name in self.stats.injected_module_names[:10]:  # 只显示前10个
                    print(f"  - {name}")
                if len(self.stats.injected_module_names) > 10:
                    print(f"  ... 还有 {len(self.stats.injected_module_names) - 10} 个")
            if self.stats.moe_module_names:
                print(f"\n注入MoE-LoRA的模块 ({len(self.stats.moe_module_names)}):")
                for name in self.stats.moe_module_names[:10]:
                    print(f"  - {name}")
                if len(self.stats.moe_module_names) > 10:
                    print(f"  ... 还有 {len(self.stats.moe_module_names) - 10} 个")
            print("=" * 70 + "\n")

        return model, self.stats

    def _inject_lora_recursive(self, module: nn.Module, parent_name: str = ""):
        """
        递归遍历模块，注入LoRA/MoE-LoRA

        Args:
            module: 当前模块
            parent_name: 父模块的名称
        """
        # 导入LoRA层（延迟导入避免循环依赖）
        from .standard.lora_layer import StandardLoRALayer
        from .moe.moe_lora_layer import MoELoRALayer

        # 获取所有子模块
        children = list(module.named_children())

        if not children:
            # 叶子节点，检查是否需要注入LoRA
            self.stats.total_modules += 1

            # 获取此模块的mode
            mode = self._get_module_mode(parent_name)

            if mode == "freeze":
                # 冻结
                self._freeze_module(module)
                self.stats.frozen += 1
            elif mode in ("lora", "moe"):
                # 需要注入LoRA/MoE-LoRA
                if isinstance(module, nn.Linear):
                    # 注入的逻辑在父模块中进行替换
                    pass
                else:
                    # 非Linear层，冻结
                    self._freeze_module(module)
                    self.stats.frozen += 1
            else:
                # 默认冻结
                self._freeze_module(module)
                self.stats.frozen += 1
        else:
            # 非叶子节点，递归处理子模块
            for child_name, child_module in children:
                # 构建完整的模块名
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name

                # 获取此模块的mode
                mode = self._get_module_mode(full_name)

                if mode == "lora":
                    # 注入标准LoRA
                    if isinstance(child_module, nn.Linear):
                        lora_layer = StandardLoRALayer(
                            base_layer=child_module,
                            rank=self.rank,
                            alpha=self.alpha,
                            dropout=self.dropout,
                            init_method=self.init_method
                        )
                        setattr(module, child_name, lora_layer)
                        self.stats.lora_injected += 1
                        self.stats.injected_module_names.append(full_name)
                        self.stats.total_modules += 1

                        if self.verbose:
                            print(f"✓ 注入标准LoRA: {full_name}")
                    else:
                        # 非Linear层，继续递归
                        self._inject_lora_recursive(child_module, full_name)

                elif mode == "moe":
                    # 注入MoE-LoRA
                    if isinstance(child_module, nn.Linear):
                        moe_lora_layer = MoELoRALayer(
                            base_layer=child_module,
                            num_experts=self.moe_config["num_experts"],
                            rank=self.rank,
                            alpha=self.alpha,
                            dropout=self.dropout,
                            routing_mode=self.moe_config["routing_mode"],
                            router_config=self.moe_config.get("router_config", {}),
                            init_method=self.init_method
                        )
                        setattr(module, child_name, moe_lora_layer)
                        self.stats.moe_injected += 1
                        self.stats.moe_module_names.append(full_name)
                        self.stats.total_modules += 1

                        if self.verbose:
                            print(f"✓ 注入MoE-LoRA: {full_name} "
                                  f"({self.moe_config['num_experts']} experts, "
                                  f"{self.moe_config['routing_mode']} mode)")
                    else:
                        # 非Linear层，继续递归
                        self._inject_lora_recursive(child_module, full_name)

                else:
                    # freeze或其他，递归处理（可能子模块需要注入）
                    self._inject_lora_recursive(child_module, full_name)

    def _get_module_mode(self, module_name: str) -> str:
        """
        获取模块的mode（lora/moe/freeze）

        Args:
            module_name: 模块的完整名称

        Returns:
            模块的mode：'lora', 'moe', 或 'freeze'
        """
        # 遍历所有模式，找到第一个匹配的
        for pattern, mode in self.module_modes.items():
            if fnmatch.fnmatch(module_name, pattern):
                return mode

        # 默认freeze
        return "freeze"

    def _should_inject_lora(self, module_name: str) -> bool:
        """
        检查模块名是否匹配target_modules中的任何模式（兼容旧API）

        Args:
            module_name: 模块的完整名称

        Returns:
            是否应该注入LoRA
        """
        mode = self._get_module_mode(module_name)
        return mode in ("lora", "moe")

    def _freeze_module(self, module: nn.Module):
        """
        冻结模块的所有参数

        Args:
            module: 要冻结的模块
        """
        for param in module.parameters():
            param.requires_grad = False

    def get_trainable_parameters(self, model: nn.Module) -> List[nn.Parameter]:
        """
        获取所有可训练的参数（仅LoRA参数）

        Args:
            model: 模型

        Returns:
            可训练参数列表
        """
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def print_trainable_parameters(self, model: nn.Module):
        """
        打印模型的可训练参数统计

        Args:
            model: 模型
        """
        trainable_params = 0
        all_param = 0

        for name, param in model.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        print(f"\n可训练参数统计:")
        print(f"  总参数: {all_param:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  可训练比例: {100 * trainable_params / all_param:.2f}%\n")


def create_lora_configurator_from_config(config) -> LoRAConfigurator:
    """
    从TrainingConfig创建LoRAConfigurator

    Args:
        config: TrainingConfig对象

    Returns:
        LoRAConfigurator实例
    """
    return LoRAConfigurator(
        target_modules=config.lora.target_modules,
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        verbose=True
    )
