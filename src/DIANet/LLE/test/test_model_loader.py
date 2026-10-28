"""
模型加载器测试

测试内容：
1. ModelLoader本身功能
   - 初始化
   - 参数验证
   - 设备检查
   - __repr__方法

2. 与其他组件的协作
   - 与config.py的集成
   - 与LoRA配置器的协作（加载的模型应该可以注入LoRA）
   - 参数计数功能

3. 模拟加载测试
   - 由于实际加载需要大模型文件，这里只测试逻辑
   - 测试dtype转换
   - 测试设备转换

注意：
由于FLUX模型非常大（~50GB），实际加载测试需要：
1. 有足够的内存/显存
2. 有模型文件或网络访问HuggingFace
3. 安装diffusers和transformers库

因此，这些测试主要关注：
- 接口正确性
- 参数验证
- 与其他组件的协同工作
- 逻辑正确性
"""

import sys
import os
import torch

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.model_loader import ModelLoader, create_model_loader_from_config
from core.config import TrainingConfig


# ============================================================================
# 第一部分：ModelLoader本身功能测试
# ============================================================================

def test_loader_initialization():
    """测试1: ModelLoader初始化"""
    # 基本初始化
    loader = ModelLoader(
        model_path="black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        device="cpu",  # 使用CPU避免CUDA依赖
        verbose=False
    )

    assert loader.model_path == "black-forest-labs/FLUX.1-dev"
    assert loader.torch_dtype == torch.bfloat16
    assert loader.device == "cpu"
    assert loader.verbose == False

    print("✓ 测试1: ModelLoader初始化")


def test_loader_cuda_check():
    """测试2: CUDA可用性检查"""
    if not torch.cuda.is_available():
        # 如果CUDA不可用，应该抛出错误
        try:
            loader = ModelLoader(
                model_path="test/model",
                device="cuda"
            )
            # 注意：初始化时不检查，只在__init__中检查
        except ValueError as e:
            assert "CUDA is not available" in str(e)
            print("✓ 测试2: CUDA可用性检查（CUDA不可用）")
            return

    # 如果CUDA可用，应该可以正常创建
    loader = ModelLoader(
        model_path="test/model",
        device="cuda",
        verbose=False
    )
    assert loader.device == "cuda"
    print("✓ 测试2: CUDA可用性检查（CUDA可用）")


def test_loader_dtype_options():
    """测试3: 不同的dtype选项"""
    # bfloat16
    loader_bf16 = ModelLoader(
        model_path="test/model",
        torch_dtype=torch.bfloat16,
        device="cpu",
        verbose=False
    )
    assert loader_bf16.torch_dtype == torch.bfloat16

    # float16
    loader_fp16 = ModelLoader(
        model_path="test/model",
        torch_dtype=torch.float16,
        device="cpu",
        verbose=False
    )
    assert loader_fp16.torch_dtype == torch.float16

    # float32
    loader_fp32 = ModelLoader(
        model_path="test/model",
        torch_dtype=torch.float32,
        device="cpu",
        verbose=False
    )
    assert loader_fp32.torch_dtype == torch.float32

    print("✓ 测试3: 不同的dtype选项")


def test_loader_repr():
    """测试4: __repr__方法"""
    loader = ModelLoader(
        model_path="test/model",
        torch_dtype=torch.bfloat16,
        device="cpu",
        verbose=False
    )

    repr_str = repr(loader)

    assert "ModelLoader" in repr_str
    assert "test/model" in repr_str
    assert "bfloat16" in repr_str
    assert "cpu" in repr_str

    print("✓ 测试4: __repr__方法")


def test_count_parameters():
    """测试5: 参数计数功能"""
    loader = ModelLoader(
        model_path="test/model",
        device="cpu",
        verbose=False
    )

    # 创建一个简单的测试模型
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)  # 10*5 + 5 = 55 参数

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    param_count = loader._count_parameters(model)

    assert param_count == 55, f"应该有55个参数，实际{param_count}"

    print("✓ 测试5: 参数计数功能")


# ============================================================================
# 第二部分：与其他组件的协作测试
# ============================================================================

def test_integration_with_config():
    """测试6: 与配置系统的集成"""
    # 创建配置
    config_dict = {
        'job_name': 'test_model_loader',
        'device': 'cpu',
        'model': {
            'base_model': 'black-forest-labs/FLUX.1-dev',
            'dtype': 'bfloat16',
        },
        'lora': {
            'rank': 16,
            'alpha': 16.0,
            'target_modules': ["*.attn.to_q"]
        },
        'data': {
            'input_image_dir': './data/input',
            'target_image_dir': './data/target',
        },
        'train': {
            'num_epochs': 10,
            'batch_size': 1,
            'learning_rate': 1e-4,
        }
    }

    config = TrainingConfig.from_dict(config_dict)

    # 从配置创建ModelLoader
    loader = create_model_loader_from_config(config)

    assert loader.model_path == 'black-forest-labs/FLUX.1-dev'
    assert loader.torch_dtype == torch.bfloat16
    assert loader.device == 'cpu'

    print("✓ 测试6: 与配置系统的集成")


def test_dtype_string_conversion():
    """测试7: dtype字符串转换"""
    # 测试不同的dtype字符串
    test_cases = [
        ('bfloat16', torch.bfloat16),
        ('float16', torch.float16),
        ('float32', torch.float32),
    ]

    for dtype_str, expected_dtype in test_cases:
        config_dict = {
            'job_name': 'test',
            'device': 'cpu',
            'model': {
                'base_model': 'test/model',
                'dtype': dtype_str,
            },
            'lora': {
                'rank': 16,
                'target_modules': ["*.attn.to_q"]
            },
            'data': {
                'input_image_dir': './data/input',
                'target_image_dir': './data/target',
            },
            'train': {
                'num_epochs': 10,
                'learning_rate': 1e-4,
            }
        }

        config = TrainingConfig.from_dict(config_dict)
        loader = create_model_loader_from_config(config)

        assert loader.torch_dtype == expected_dtype, \
            f"dtype_str={dtype_str}应该转换为{expected_dtype}"

    print("✓ 测试7: dtype字符串转换")


def test_integration_with_lora_configurator():
    """测试8: 与LoRA配置器的协作"""
    # 这个测试验证：加载的模型应该可以与LoRA配置器协同工作
    # 由于实际加载需要大模型，这里创建一个简单的mock模型测试逻辑

    import torch.nn as nn
    from core.lora.configurator import LoRAConfigurator

    # 创建一个简单的transformer模拟
    class MockTransformerBlock(nn.Module):
        def __init__(self, dim=128):
            super().__init__()
            self.attn_to_q = nn.Linear(dim, dim)
            self.attn_to_k = nn.Linear(dim, dim)
            self.attn_to_v = nn.Linear(dim, dim)

        def forward(self, x):
            return x

    class MockTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.single_transformer_blocks = nn.ModuleList([
                MockTransformerBlock() for _ in range(3)
            ])

        def forward(self, x):
            for block in self.single_transformer_blocks:
                x = block(x)
            return x

    # 创建mock transformer
    mock_transformer = MockTransformer()

    # 创建LoRA配置器
    configurator = LoRAConfigurator(
        target_modules=["single_transformer_blocks.*.attn_to_q"],
        rank=8,
        verbose=False
    )

    # 注入LoRA
    mock_transformer, stats = configurator.inject_lora(mock_transformer)

    # 验证LoRA注入成功
    assert stats.lora_injected == 3, "应该注入3个LoRA层"

    print("✓ 测试8: 与LoRA配置器的协作")


def test_model_components_compatibility():
    """测试9: 模型组件兼容性测试"""
    # 测试加载器返回的组件类型是否正确
    # 这是一个逻辑测试，不实际加载模型

    loader = ModelLoader(
        model_path="test/model",
        device="cpu",
        verbose=False
    )

    # 验证方法存在
    assert hasattr(loader, 'load_all'), "应该有load_all方法"
    assert hasattr(loader, 'load_transformer_only'), "应该有load_transformer_only方法"
    assert hasattr(loader, '_count_parameters'), "应该有_count_parameters方法"

    print("✓ 测试9: 模型组件兼容性测试")


def test_verbose_mode():
    """测试10: verbose模式"""
    # 测试verbose=True时应该打印信息
    # 这里不实际加载，只测试verbose参数被正确保存

    loader_verbose = ModelLoader(
        model_path="test/model",
        device="cpu",
        verbose=True
    )
    assert loader_verbose.verbose == True

    loader_quiet = ModelLoader(
        model_path="test/model",
        device="cpu",
        verbose=False
    )
    assert loader_quiet.verbose == False

    print("✓ 测试10: verbose模式")


def test_parameter_counting_various_models():
    """测试11: 各种模型的参数计数"""
    import torch.nn as nn

    loader = ModelLoader(
        model_path="test/model",
        device="cpu",
        verbose=False
    )

    # 测试1: 空模型
    class EmptyModel(nn.Module):
        def forward(self, x):
            return x

    empty_model = EmptyModel()
    assert loader._count_parameters(empty_model) == 0

    # 测试2: 单层Linear
    class SingleLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)

    single_linear = SingleLinear()
    # 10*5 + 5 = 55
    assert loader._count_parameters(single_linear) == 55

    # 测试3: 多层模型
    class MultiLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)  # 10*20 + 20 = 220
            self.fc2 = nn.Linear(20, 5)   # 20*5 + 5 = 105

        def forward(self, x):
            return self.fc2(self.fc1(x))

    multi_layer = MultiLayer()
    # 220 + 105 = 325
    assert loader._count_parameters(multi_layer) == 325

    print("✓ 测试11: 各种模型的参数计数")


def test_end_to_end_workflow_simulation():
    """测试12: 端到端工作流模拟"""
    # 模拟完整的工作流：配置 → 加载器 → (模拟)LoRA注入

    # 步骤1: 创建配置
    config_dict = {
        'job_name': 'test_e2e',
        'device': 'cpu',
        'model': {
            'base_model': 'test/model',
            'dtype': 'float32',
        },
        'lora': {
            'rank': 8,
            'alpha': 8.0,
            'target_modules': ["*.attn.to_q"]
        },
        'data': {
            'input_image_dir': './data/input',
            'target_image_dir': './data/target',
        },
        'train': {
            'num_epochs': 10,
            'batch_size': 1,
            'learning_rate': 1e-4,
        }
    }
    config = TrainingConfig.from_dict(config_dict)

    # 步骤2: 从配置创建加载器
    loader = create_model_loader_from_config(config)

    # 验证加载器配置
    assert loader.model_path == 'test/model'
    assert loader.torch_dtype == torch.float32
    assert loader.device == 'cpu'

    # 步骤3: 创建LoRA配置器（模拟后续流程）
    from core.lora.configurator import create_lora_configurator_from_config

    lora_configurator = create_lora_configurator_from_config(config)

    # 验证LoRA配置器
    assert lora_configurator.rank == 8
    assert lora_configurator.alpha == 8.0

    # 在实际使用中，这两个组件会协同工作：
    # transformer, ... = loader.load_all()
    # transformer, stats = lora_configurator.inject_lora(transformer)

    print("✓ 测试12: 端到端工作流模拟")


def test_different_model_paths():
    """测试13: 不同的模型路径格式"""
    # HuggingFace模型ID
    loader_hf = ModelLoader(
        model_path="black-forest-labs/FLUX.1-dev",
        device="cpu",
        verbose=False
    )
    assert loader_hf.model_path == "black-forest-labs/FLUX.1-dev"

    # 本地路径
    loader_local = ModelLoader(
        model_path="/path/to/local/model",
        device="cpu",
        verbose=False
    )
    assert loader_local.model_path == "/path/to/local/model"

    # 相对路径
    loader_relative = ModelLoader(
        model_path="./models/flux",
        device="cpu",
        verbose=False
    )
    assert loader_relative.model_path == "./models/flux"

    print("✓ 测试13: 不同的模型路径格式")


def test_device_cpu_vs_cuda():
    """测试14: CPU vs CUDA设备"""
    # CPU应该总是可用
    loader_cpu = ModelLoader(
        model_path="test/model",
        device="cpu",
        verbose=False
    )
    assert loader_cpu.device == "cpu"

    # CUDA只在可用时测试
    if torch.cuda.is_available():
        loader_cuda = ModelLoader(
            model_path="test/model",
            device="cuda",
            verbose=False
        )
        assert loader_cuda.device == "cuda"
        print("✓ 测试14: CPU vs CUDA设备（CUDA可用）")
    else:
        print("✓ 测试14: CPU vs CUDA设备（CUDA不可用，只测试CPU）")


def test_config_integration_all_dtypes():
    """测试15: 配置集成测试所有dtype"""
    dtypes = ['bfloat16', 'float16', 'float32']

    for dtype_str in dtypes:
        config_dict = {
            'job_name': f'test_{dtype_str}',
            'device': 'cpu',
            'model': {
                'base_model': 'test/model',
                'dtype': dtype_str,
            },
            'lora': {
                'rank': 16,
                'target_modules': ["*.attn.to_q"]
            },
            'data': {
                'input_image_dir': './data/input',
                'target_image_dir': './data/target',
            },
            'train': {
                'num_epochs': 10,
                'learning_rate': 1e-4,
            }
        }

        config = TrainingConfig.from_dict(config_dict)
        loader = create_model_loader_from_config(config)

        # 验证dtype正确转换
        expected_dtype = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
        }[dtype_str]

        assert loader.torch_dtype == expected_dtype

    print("✓ 测试15: 配置集成测试所有dtype")


# ============================================================================
# 运行所有测试
# ============================================================================

def main():
    """运行所有测试"""
    print("=" * 60)
    print("开始模型加载器测试")
    print("=" * 60)
    print("\n注意：这些测试主要验证接口和逻辑正确性")
    print("实际的模型加载需要大量内存和网络访问\n")

    # 第一部分：ModelLoader本身功能测试
    print("\n【第一部分：ModelLoader本身功能测试】")
    test_loader_initialization()
    test_loader_cuda_check()
    test_loader_dtype_options()
    test_loader_repr()
    test_count_parameters()

    # 第二部分：与其他组件的协作测试
    print("\n【第二部分：与其他组件的协作测试】")
    test_integration_with_config()
    test_dtype_string_conversion()
    test_integration_with_lora_configurator()
    test_model_components_compatibility()
    test_verbose_mode()
    test_parameter_counting_various_models()
    test_end_to_end_workflow_simulation()
    test_different_model_paths()
    test_device_cpu_vs_cuda()
    test_config_integration_all_dtypes()

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    print("\n提示：要测试实际模型加载，请确保：")
    print("  1. 安装diffusers和transformers库")
    print("  2. 有足够的内存/显存（~50GB）")
    print("  3. 有网络访问HuggingFace或本地模型文件")
    print("=" * 60)


if __name__ == "__main__":
    main()
