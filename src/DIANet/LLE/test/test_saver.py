"""
ModelSaver测试

测试内容：
1. ModelSaver本身功能
   - 初始化
   - 保存精度验证
   - LoRA权重提取
   - 保存到safetensors
   - 检查点清理

2. 与其他组件的协作
   - 与配置系统的集成
   - 与Transformer模型的集成
   - 端到端保存流程
"""

import sys
import os
import torch
import torch.nn as nn
import tempfile
import shutil

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.saver import ModelSaver, create_saver_from_config
from core.config import TrainingConfig, ModelConfig, DataConfig, LoRAConfig, TrainConfig, SaveConfig


# ============================================================================
# Mock组件
# ============================================================================

class MockTransformerWithLoRA(nn.Module):
    """模拟带LoRA的Transformer"""
    def __init__(self):
        super().__init__()
        self.base_linear = nn.Linear(128, 128)
        # 添加LoRA参数
        self.lora_A = nn.Parameter(torch.randn(128, 16))
        self.lora_B = nn.Parameter(torch.randn(16, 128))

    def forward(self, x):
        return self.base_linear(x)


# ============================================================================
# 第一部分：ModelSaver本身功能测试
# ============================================================================

def test_saver_initialization():
    """测试1: ModelSaver初始化"""
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = ModelSaver(
            output_dir=tmpdir,
            save_dtype="float16",
            save_total_limit=3,
            verbose=False
        )

        assert saver.output_dir == tmpdir
        assert saver.save_dtype == "float16"
        assert saver.save_total_limit == 3
        assert saver.verbose == False

        # 检查目录是否创建
        assert os.path.exists(tmpdir)

    print("✓ 测试1: ModelSaver初始化")


def test_validate_save_dtype():
    """测试2: 保存精度验证"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 测试合法的精度
        valid_dtypes = ["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"]
        for dtype in valid_dtypes:
            saver = ModelSaver(output_dir=tmpdir, save_dtype=dtype, verbose=False)
            assert saver.save_dtype == dtype

        # 测试非法的精度
        try:
            saver = ModelSaver(output_dir=tmpdir, save_dtype="invalid", verbose=False)
            assert False, "应该抛出ValueError"
        except ValueError as e:
            assert "不支持的保存精度" in str(e)

    print("✓ 测试2: 保存精度验证")


def test_resolve_save_dtype():
    """测试3: 精度字符串转换"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_cases = [
            ("float16", torch.float16),
            ("fp16", torch.float16),
            ("bfloat16", torch.bfloat16),
            ("bf16", torch.bfloat16),
            ("float32", torch.float32),
            ("fp32", torch.float32),
        ]

        for dtype_str, expected_dtype in test_cases:
            saver = ModelSaver(output_dir=tmpdir, save_dtype=dtype_str, verbose=False)
            resolved = saver._resolve_save_dtype()
            assert resolved == expected_dtype, f"{dtype_str} should resolve to {expected_dtype}"

    print("✓ 测试3: 精度字符串转换")


def test_extract_lora_weights():
    """测试4: LoRA权重提取"""
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = ModelSaver(output_dir=tmpdir, save_dtype="float16", verbose=False)

        # 创建模拟模型
        model = MockTransformerWithLoRA()

        # 提取LoRA权重
        lora_weights = saver._extract_lora_weights(model, torch.float16)

        # 检查提取的权重
        assert "lora_A" in lora_weights
        assert "lora_B" in lora_weights
        assert "base_linear.weight" not in lora_weights  # 基础层不应该被提取
        assert "base_linear.bias" not in lora_weights

        # 检查精度转换
        assert lora_weights["lora_A"].dtype == torch.float16
        assert lora_weights["lora_B"].dtype == torch.float16

        # 检查形状
        assert lora_weights["lora_A"].shape == (128, 16)
        assert lora_weights["lora_B"].shape == (16, 128)

    print("✓ 测试4: LoRA权重提取")


def test_save_lora():
    """测试5: 保存LoRA权重"""
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = ModelSaver(output_dir=tmpdir, save_dtype="float16", verbose=False)

        # 创建模拟模型
        model = MockTransformerWithLoRA()

        # 测试epoch保存
        save_path = saver.save_lora(model, epoch=1)
        assert os.path.exists(save_path)
        assert "lora_epoch_1.safetensors" in save_path

        # 测试最终保存
        save_path = saver.save_lora(model, epoch=None)
        assert os.path.exists(save_path)
        assert "lora_final.safetensors" in save_path

        # 检查文件大小
        file_size = os.path.getsize(save_path)
        assert file_size > 0

    print("✓ 测试5: 保存LoRA权重")


def test_cleanup_old_checkpoints():
    """测试6: 检查点清理"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 设置保存限制为3
        saver = ModelSaver(
            output_dir=tmpdir,
            save_dtype="float16",
            save_total_limit=3,
            verbose=False
        )

        model = MockTransformerWithLoRA()

        # 保存5个epoch
        for epoch in range(1, 6):
            saver.save_lora(model, epoch=epoch)

        # 检查只保留了最近的3个
        saved_files = [f for f in os.listdir(tmpdir) if f.startswith("lora_epoch_")]
        assert len(saved_files) == 3

        # 检查保留的是最新的3个
        saved_epochs = sorted([
            int(f.replace("lora_epoch_", "").replace(".safetensors", ""))
            for f in saved_files
        ])
        assert saved_epochs == [3, 4, 5]

    print("✓ 测试6: 检查点清理")


def test_no_limit_cleanup():
    """测试7: 无限制保存"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 不设置保存限制
        saver = ModelSaver(
            output_dir=tmpdir,
            save_dtype="float16",
            save_total_limit=None,
            verbose=False
        )

        model = MockTransformerWithLoRA()

        # 保存5个epoch
        for epoch in range(1, 6):
            saver.save_lora(model, epoch=epoch)

        # 检查所有5个都保留了
        saved_files = [f for f in os.listdir(tmpdir) if f.startswith("lora_epoch_")]
        assert len(saved_files) == 5

    print("✓ 测试7: 无限制保存")


def test_repr():
    """测试8: __repr__方法"""
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = ModelSaver(
            output_dir=tmpdir,
            save_dtype="bfloat16",
            save_total_limit=5,
            verbose=False
        )

        repr_str = repr(saver)

        assert "ModelSaver" in repr_str
        assert tmpdir in repr_str
        assert "bfloat16" in repr_str
        assert "5" in repr_str

    print("✓ 测试8: __repr__方法")


# ============================================================================
# 第二部分：与其他组件的协作测试
# ============================================================================

def test_integration_with_config():
    """测试9: 与配置系统的集成"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            model=ModelConfig(base_model="test"),
            data=DataConfig(input_image_dir=".", target_image_dir="."),
            lora=LoRAConfig(),
            train=TrainConfig(),
            save=SaveConfig(
                output_dir=tmpdir,
                save_every_n_epochs=10,
                save_dtype="bfloat16"
            ),
            device="cpu"
        )

        saver = create_saver_from_config(config)

        assert saver.output_dir == tmpdir
        assert saver.save_dtype == "bfloat16"

    print("✓ 测试9: 与配置系统的集成")


def test_integration_save_and_load():
    """测试10: 保存并加载LoRA权重"""
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = ModelSaver(output_dir=tmpdir, save_dtype="float32", verbose=False)

        # 创建模型并保存
        model = MockTransformerWithLoRA()
        original_lora_A = model.lora_A.data.clone()
        original_lora_B = model.lora_B.data.clone()

        save_path = saver.save_lora(model, epoch=1)

        # 加载权重
        from safetensors.torch import load_file
        loaded_weights = load_file(save_path)

        # 检查加载的权重与原始权重一致
        assert torch.allclose(loaded_weights["lora_A"], original_lora_A, atol=1e-5)
        assert torch.allclose(loaded_weights["lora_B"], original_lora_B, atol=1e-5)

    print("✓ 测试10: 保存并加载LoRA权重")


def test_different_precisions():
    """测试11: 不同精度保存"""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = MockTransformerWithLoRA()

        # 测试不同精度
        test_cases = [
            ("float16", torch.float16),
            ("bfloat16", torch.bfloat16),
            ("float32", torch.float32),
        ]

        for dtype_str, expected_dtype in test_cases:
            saver = ModelSaver(output_dir=tmpdir, save_dtype=dtype_str, verbose=False)
            save_path = saver.save_lora(model, epoch=None)

            # 加载并检查精度
            from safetensors.torch import load_file
            loaded_weights = load_file(save_path)

            assert loaded_weights["lora_A"].dtype == expected_dtype
            assert loaded_weights["lora_B"].dtype == expected_dtype

    print("✓ 测试11: 不同精度保存")


def test_empty_lora_warning():
    """测试12: 空LoRA警告"""
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = ModelSaver(output_dir=tmpdir, save_dtype="float16", verbose=False)

        # 创建没有LoRA参数的模型
        class EmptyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

        model = EmptyModel()

        # 提取权重（应该为空）
        lora_weights = saver._extract_lora_weights(model, torch.float16)
        assert len(lora_weights) == 0

    print("✓ 测试12: 空LoRA警告")


# ============================================================================
# 运行所有测试
# ============================================================================

def main():
    """运行所有测试"""
    print("=" * 60)
    print("开始ModelSaver测试")
    print("=" * 60)

    # 第一部分：ModelSaver本身功能测试
    print("\n【第一部分：ModelSaver本身功能测试】")
    test_saver_initialization()
    test_validate_save_dtype()
    test_resolve_save_dtype()
    test_extract_lora_weights()
    test_save_lora()
    test_cleanup_old_checkpoints()
    test_no_limit_cleanup()
    test_repr()

    # 第二部分：与其他组件的协作测试
    print("\n【第二部分：与其他组件的协作测试】")
    test_integration_with_config()
    test_integration_save_and_load()
    test_different_precisions()
    test_empty_lora_warning()

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
