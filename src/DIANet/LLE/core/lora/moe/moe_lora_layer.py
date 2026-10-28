"""
MoE-LoRA层实现

MoE-LoRA层包含多个LoRA专家，使用Router进行专家选择和加权聚合

支持两种路由模式：
1. 单路由模式（single）: 使用StandardRouter
2. 双路由模式（dual）: 使用SpecificRouter + SharedRouter
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .router import StandardRouter, SpecificRouter, SharedRouter
from ..base.base_lora import BaseLoRALayer


class MoELoRALayer(BaseLoRALayer):
    """
    MoE-LoRA层

    每个专家是一个独立的LoRA (A_i, B_i)对
    使用Router选择Top-K个专家并加权聚合它们的输出

    Args:
        base_layer: 基础层（将被冻结）
        num_experts: 专家数量
        rank: LoRA rank（每个专家的rank）
        alpha: LoRA alpha（scaling factor）
        dropout: Dropout概率
        routing_mode: 路由模式 "single" 或 "dual"
        router_config: 路由器配置
        init_method: 初始化方法
    """

    def __init__(
        self,
        base_layer: nn.Module,
        num_experts: int = 4,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        routing_mode: str = "single",
        router_config: Optional[dict] = None,
        init_method: str = "kaiming"
    ):
        super().__init__(base_layer=base_layer, rank=rank, alpha=alpha, dropout=dropout)

        # 基本参数
        self.num_experts = num_experts
        self.routing_mode = routing_mode
        self.init_method = init_method

        # 当前批次的曝光度类别（用于训练时的自动传递）
        self._current_exposure_category = None

        # 获取基础层的维度
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # 创建多个LoRA专家
        # 每个专家是一对 (A, B) 矩阵
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.zeros(in_features, rank))
            for _ in range(num_experts)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(rank, out_features))
            for _ in range(num_experts)
        ])

        # Dropout层（如果需要）
        # self.dropout已经从BaseLoRALayer继承
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        else:
            self.dropout_layer = None

        # 路由器配置
        if router_config is None:
            router_config = {}

        # 创建路由器
        if routing_mode == "single":
            # 单路由模式：使用StandardRouter
            top_k = router_config.get('top_k', 2)
            use_noisy_gating = router_config.get('use_noisy_gating', False)
            noise_epsilon = router_config.get('noise_epsilon', 1e-2)

            self.router = StandardRouter(
                input_dim=in_features,
                num_experts=num_experts,
                top_k=top_k,
                use_noisy_gating=use_noisy_gating,
                noise_epsilon=noise_epsilon
            )

        elif routing_mode == "dual":
            # 双路由模式：SpecificRouter + SharedRouter
            # 前面的专家是Specific专家，后面的是Shared专家
            num_specific = router_config.get('num_specific', 3)
            num_shared = num_experts - num_specific

            if num_shared <= 0:
                raise ValueError(
                    f"num_experts ({num_experts}) must be > num_specific ({num_specific})"
                )

            self.num_specific = num_specific

            # Specific路由器（硬编码）
            self.specific_router = SpecificRouter(num_experts=num_specific)

            # Shared路由器（可学习）
            shared_top_k = router_config.get('shared_top_k', 2)
            shared_hidden_dim = router_config.get('shared_hidden_dim', 512)

            self.shared_router = SharedRouter(
                input_dim=in_features,
                num_experts=num_shared,
                top_k=shared_top_k,
                hidden_dim=shared_hidden_dim
            )
        else:
            raise ValueError(f"Unknown routing_mode: {routing_mode}")

        # 初始化LoRA参数
        self._init_lora_parameters()

        # 移动LoRA参数到与base_layer相同的device和dtype
        if base_layer.weight is not None:
            device = base_layer.weight.device
            dtype = base_layer.weight.dtype
            for i in range(num_experts):
                self.lora_A[i].data = self.lora_A[i].data.to(device=device, dtype=dtype)
                self.lora_B[i].data = self.lora_B[i].data.to(device=device, dtype=dtype)

            # 移动router到相同设备和dtype
            if routing_mode == "single":
                self.router = self.router.to(device=device, dtype=dtype)
            elif routing_mode == "dual":
                self.specific_router = self.specific_router.to(device=device, dtype=dtype)
                self.shared_router = self.shared_router.to(device=device, dtype=dtype)

    def _init_lora_parameters(self):
        """初始化LoRA参数"""
        for i in range(self.num_experts):
            if self.init_method == "kaiming":
                # Kaiming初始化（He initialization）
                nn.init.kaiming_uniform_(self.lora_A[i], a=5**0.5)
                nn.init.zeros_(self.lora_B[i])
            elif self.init_method == "normal":
                # 正态分布初始化
                nn.init.normal_(self.lora_A[i], std=0.02)
                nn.init.zeros_(self.lora_B[i])
            elif self.init_method == "zero":
                # 全零初始化
                nn.init.zeros_(self.lora_A[i])
                nn.init.zeros_(self.lora_B[i])
            else:
                raise ValueError(f"Unknown init_method: {self.init_method}")

    def forward(
        self,
        x: torch.Tensor,
        exposure_category: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入 [batch_size, ..., in_features]
            exposure_category: 曝光度类别 [batch_size]，仅在dual模式下需要
                             如果为None，会尝试使用 self._current_exposure_category

        Returns:
            输出 [batch_size, ..., out_features]
        """
        # 1. 基础层输出
        base_out = self.base_layer(x)

        # 2. 计算LoRA输出
        if self.routing_mode == "single":
            lora_out = self._forward_single_route(x)
        elif self.routing_mode == "dual":
            # 尝试从参数或实例变量获取 exposure_category
            if exposure_category is None:
                exposure_category = self._current_exposure_category

            if exposure_category is None:
                raise ValueError(
                    "exposure_category is required for dual routing mode. "
                    "Either pass it as an argument or set it via set_exposure_category()"
                )
            lora_out = self._forward_dual_route(x, exposure_category)
        else:
            raise ValueError(f"Unknown routing_mode: {self.routing_mode}")

        # 3. 合并输出
        return base_out + self.scaling * lora_out

    def set_exposure_category(self, exposure_category: Optional[torch.Tensor]):
        """
        设置当前批次的曝光度类别（用于训练时自动传递）

        Args:
            exposure_category: 曝光度类别 [batch_size]
        """
        self._current_exposure_category = exposure_category

    def _forward_single_route(self, x: torch.Tensor) -> torch.Tensor:
        """
        单路由模式前向传播

        Args:
            x: 输入 [batch_size, seq_len, in_features] 或 [batch_size, in_features]

        Returns:
            LoRA输出 [batch_size, ..., out_features]
        """
        original_shape = x.shape
        batch_size = x.size(0)

        # 如果是2D输入，需要添加seq_len维度用于Router
        if x.ndim == 2:
            x_for_router = x.unsqueeze(1)  # [batch_size, 1, in_features]
        else:
            x_for_router = x

        # 路由选择专家
        # weights: [batch_size, top_k]
        # indices: [batch_size, top_k]
        weights, indices = self.router(x_for_router)

        # 初始化输出
        lora_out = torch.zeros(
            *original_shape[:-1],
            self.base_layer.out_features,
            device=x.device,
            dtype=x.dtype
        )

        # 对每个batch独立处理
        for batch_idx in range(batch_size):
            # 获取这个batch选中的专家
            batch_weights = weights[batch_idx]  # [top_k]
            batch_indices = indices[batch_idx]  # [top_k]

            # 获取这个batch的输入
            if x.ndim == 2:
                x_batch = x[batch_idx]  # [in_features]
            else:
                x_batch = x[batch_idx]  # [seq_len, in_features]

            # 聚合所有选中专家的输出
            for k_idx in range(len(batch_indices)):
                expert_idx = batch_indices[k_idx].item()
                weight = batch_weights[k_idx]

                # LoRA计算: x @ A @ B
                expert_out = x_batch @ self.lora_A[expert_idx] @ self.lora_B[expert_idx]

                # 应用Dropout（如果有）
                if self.dropout_layer is not None and self.training:
                    expert_out = self.dropout_layer(expert_out)

                # 加权累加
                lora_out[batch_idx] += weight * expert_out

        return lora_out

    def _forward_dual_route(
        self,
        x: torch.Tensor,
        exposure_category: torch.Tensor
    ) -> torch.Tensor:
        """
        双路由模式前向传播

        Args:
            x: 输入 [batch_size, seq_len, in_features] 或 [batch_size, in_features]
            exposure_category: 曝光度类别 [batch_size]

        Returns:
            LoRA输出 [batch_size, ..., out_features]
        """
        original_shape = x.shape
        batch_size = x.size(0)

        # 如果是2D输入，需要添加seq_len维度
        if x.ndim == 2:
            x_for_router = x.unsqueeze(1)  # [batch_size, 1, in_features]
        else:
            x_for_router = x

        # 1. Style-Specific路由（硬编码）
        specific_weights, specific_indices = self.specific_router(
            x_for_router,
            exposure_category=exposure_category
        )

        # 2. Style-Shared路由（可学习）
        shared_weights, shared_indices = self.shared_router(x_for_router)

        # 初始化输出
        lora_out = torch.zeros(
            *original_shape[:-1],
            self.base_layer.out_features,
            device=x.device,
            dtype=x.dtype
        )

        # 对每个batch独立处理
        for batch_idx in range(batch_size):
            if x.ndim == 2:
                x_batch = x[batch_idx]  # [in_features]
            else:
                x_batch = x[batch_idx]  # [seq_len, in_features]

            # Specific专家输出
            specific_idx = specific_indices[batch_idx, 0].item()
            specific_out = x_batch @ self.lora_A[specific_idx] @ self.lora_B[specific_idx]

            if self.dropout_layer is not None and self.training:
                specific_out = self.dropout_layer(specific_out)

            lora_out[batch_idx] += specific_out

            # Shared专家输出
            batch_shared_weights = shared_weights[batch_idx]  # [top_k]
            batch_shared_indices = shared_indices[batch_idx]  # [top_k]

            for k_idx in range(len(batch_shared_indices)):
                # Shared专家索引从num_specific开始
                shared_local_idx = batch_shared_indices[k_idx].item()
                expert_idx = self.num_specific + shared_local_idx
                weight = batch_shared_weights[k_idx]

                expert_out = x_batch @ self.lora_A[expert_idx] @ self.lora_B[expert_idx]

                if self.dropout_layer is not None and self.training:
                    expert_out = self.dropout_layer(expert_out)

                lora_out[batch_idx] += weight * expert_out

        return lora_out

    def get_lora_parameters(self):
        """获取LoRA参数（用于保存）"""
        params = []
        for i in range(self.num_experts):
            params.extend([self.lora_A[i], self.lora_B[i]])
        return params

    def get_routing_stats(self):
        """获取路由统计信息"""
        if self.routing_mode == "single":
            return self.router.get_routing_stats()
        elif self.routing_mode == "dual":
            return {
                'specific': self.specific_router.get_routing_stats(),
                'shared': self.shared_router.get_routing_stats()
            }

    def reset_routing_stats(self):
        """重置路由统计"""
        if self.routing_mode == "single":
            self.router.reset_stats()
        elif self.routing_mode == "dual":
            self.specific_router.reset_stats()
            self.shared_router.reset_stats()

    def merge_to_base_layer(self):
        """
        将LoRA参数合并到基础层（不支持MoE-LoRA）

        MoE-LoRA有多个专家，无法简单合并，抛出错误
        """
        raise NotImplementedError(
            "MoE-LoRA cannot be merged to base layer. "
            "Use save_lora() to save expert weights separately."
        )

    def get_trainable_parameters(self):
        """
        获取可训练参数

        Returns:
            可训练参数的迭代器
        """
        # LoRA专家参数
        for param in self.get_lora_parameters():
            yield param

        # Router参数
        if self.routing_mode == "single":
            for param in self.router.parameters():
                yield param
        elif self.routing_mode == "dual":
            # Specific router没有可训练参数（硬编码）
            # Shared router有可训练参数
            for param in self.shared_router.parameters():
                yield param
