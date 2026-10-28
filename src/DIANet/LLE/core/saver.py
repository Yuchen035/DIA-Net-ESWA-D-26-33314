"""
模型保存器 - 负责保存LoRA权重

MVP版本：
- 保存LoRA权重到safetensors格式
- 支持epoch保存和最终保存
- 支持不同的保存精度（float16/bfloat16/float32）
- 简单的文件管理（保存限制）
"""

import os
import torch
from safetensors.torch import save_file
from typing import Optional


class ModelSaver:
    """
    模型保存器 - MVP版本

    职责：
    - 保存LoRA权重到safetensors
    - 管理检查点文件
    - 支持不同的保存精度
    """

    def __init__(
        self,
        output_dir: str,
        save_dtype: str = "float16",
        save_total_limit: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Args:
            output_dir: 输出目录
            save_dtype: 保存精度 ("float16", "bfloat16", "float32")
            save_total_limit: 最多保留的检查点数量（None表示不限制）
            verbose: 是否打印详细信息
        """
        self.output_dir = output_dir
        self.save_dtype = save_dtype
        self.save_total_limit = save_total_limit
        self.verbose = verbose

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 验证save_dtype
        self._validate_save_dtype()

    def _validate_save_dtype(self):
        """验证保存精度是否合法"""
        valid_dtypes = ["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"]
        if self.save_dtype.lower() not in valid_dtypes:
            raise ValueError(
                f"不支持的保存精度: {self.save_dtype}. "
                f"支持的精度: {', '.join(valid_dtypes)}"
            )

    def _resolve_save_dtype(self) -> torch.dtype:
        """将配置中的保存精度字符串转换为torch.dtype"""
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return dtype_map[self.save_dtype.lower()]

    def _log(self, message: str):
        """打印日志"""
        if self.verbose:
            print(message)

    def save_lora(self, transformer, epoch: Optional[int] = None):
        """
        保存LoRA权重到safetensors

        Args:
            transformer: 包含LoRA的Transformer模型
            epoch: Epoch编号（None表示最终保存）
        """
        # 确定保存路径
        if epoch is not None:
            save_path = os.path.join(self.output_dir, f"lora_epoch_{epoch}.safetensors")
            self._log(f"\n[保存] 正在保存 Epoch {epoch} 的 LoRA 权重...")
        else:
            save_path = os.path.join(self.output_dir, "lora_final.safetensors")
            self._log(f"\n[保存] 正在保存最终 LoRA 权重...")

        # 获取保存精度
        save_dtype = self._resolve_save_dtype()

        # 提取LoRA权重
        lora_state_dict = self._extract_lora_weights(transformer, save_dtype)

        # 保存到safetensors
        save_file(lora_state_dict, save_path)

        # 打印保存信息
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        num_params = sum(v.numel() for v in lora_state_dict.values())

        self._log(f"[保存] ✓ LoRA 权重已保存: {save_path}")
        self._log(f"[保存] 文件大小: {file_size_mb:.2f} MB")
        self._log(f"[保存] 参数数量: {num_params:,}")
        self._log(f"[保存] 保存精度: {self.save_dtype}\n")

        # 清理旧的检查点
        if epoch is not None and self.save_total_limit is not None:
            self._cleanup_old_checkpoints(epoch)

        return save_path

    def _extract_lora_weights(self, transformer, save_dtype: torch.dtype) -> dict:
        """
        从Transformer中提取LoRA权重

        Args:
            transformer: Transformer模型
            save_dtype: 保存精度

        Returns:
            包含LoRA权重的字典
        """
        lora_state_dict = {}

        for name, param in transformer.named_parameters():
            # 只保存包含"lora_"的参数
            if "lora_" in name:
                # 转换到CPU并转换精度
                lora_state_dict[name] = param.detach().to("cpu", dtype=save_dtype)

        if len(lora_state_dict) == 0:
            self._log("[警告] 未找到任何LoRA参数！")

        return lora_state_dict

    def _cleanup_old_checkpoints(self, current_epoch: int):
        """
        清理旧的检查点，只保留最近的N个

        Args:
            current_epoch: 当前epoch编号
        """
        # 获取所有epoch检查点
        checkpoint_files = []
        for filename in os.listdir(self.output_dir):
            if filename.startswith("lora_epoch_") and filename.endswith(".safetensors"):
                try:
                    # 提取epoch编号
                    epoch_num = int(filename.replace("lora_epoch_", "").replace(".safetensors", ""))
                    filepath = os.path.join(self.output_dir, filename)
                    checkpoint_files.append((epoch_num, filepath))
                except ValueError:
                    continue

        # 按epoch编号排序
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)

        # 删除超出限制的检查点
        if len(checkpoint_files) > self.save_total_limit:
            for epoch_num, filepath in checkpoint_files[self.save_total_limit:]:
                try:
                    os.remove(filepath)
                    self._log(f"[清理] 删除旧检查点: lora_epoch_{epoch_num}.safetensors")
                except Exception as e:
                    self._log(f"[警告] 无法删除检查点 {filepath}: {e}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"output_dir={self.output_dir}, "
            f"save_dtype={self.save_dtype}, "
            f"save_total_limit={self.save_total_limit})"
        )


def create_saver_from_config(config):
    """
    从TrainingConfig创建ModelSaver

    Args:
        config: TrainingConfig对象

    Returns:
        ModelSaver实例
    """
    # 优先使用save配置，如果没有则使用train配置
    if hasattr(config, 'save'):
        output_dir = config.save.output_dir
        save_dtype = config.save.save_dtype
        save_total_limit = getattr(config.save, 'save_total_limit', None)
    else:
        # Fallback到简单配置
        output_dir = getattr(config, 'output_dir', './outputs/lora_training')
        save_dtype = getattr(config, 'save_dtype', 'float16')
        save_total_limit = None

    return ModelSaver(
        output_dir=output_dir,
        save_dtype=save_dtype,
        save_total_limit=save_total_limit,
        verbose=True,
    )
