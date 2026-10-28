"""
LoRA配置器测试

测试内容：
1. 配置器本身功能
   - 初始化
   - 通配符模式匹配
   - LoRA注入
   - 统计信息
   - 参数冻结

2. 与其他组件的协作
   - 与config.py的集成
   - 与standard_lora.py的集成
   - 端到端测试（配置文件 → 注入LoRA）
"""

import sys
import os
import torch
import torch.nn as nn

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.lora.configurator import (
    LoRAConfigurator,
    LoRAInjectionStats,
    create_lora_configurator_from_config
)
from core.lora.standard.lora_layer import StandardLoRALayer
from core.config import TrainingConfig


# ============================================================================
# 测试辅助函数
# ============================================================================

def create_test_model():
    """创建测试用的模型结构"""
    class Attention(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.to_q = nn.Linear(dim, dim)
            self.to_k = nn.Linear(dim, dim)
            self.to_v = nn.Linear(dim, dim)
            self.to_out = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Dropout(0.1)
            )

        def forward(self, x):
            return x

    class FFN(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )

        def forward(self, x):
            return self.net(x)

    class TransformerBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attn = Attention(dim)
            self.ff = FFN(dim)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.ff(self.norm2(x))
            return x

    class TestModel(nn.Module):
        def __init__(self, num_blocks=3, dim=128):
            super().__init__()
            self.single_transformer_blocks = nn.ModuleList([
                TransformerBlock(dim) for _ in range(num_blocks)
            ])
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(dim) for _ in range(2)
            ])
            self.proj_out = nn.Linear(dim, dim)

        def forward(self, x):
            for block in self.single_transformer_blocks:
                x = block(x)
            for block in self.transformer_blocks:
                x = block(x)
            return self.proj_out(x)

    return TestModel()


# ============================================================================
# 第一部分：配置器本身功能测试
# ============================================================================

def test_configurator_initialization():
    """测试1: 配置器初始化"""
    configurator = LoRAConfigurator(
        target_modules=["*.attn.to_q", "*.attn.to_k"],
        rank=16,
        alpha=16.0,
        dropout=0.0,
        verbose=False
    )

    assert configurator.rank == 16, "rank应该为16"
    assert configurator.alpha == 16.0, "alpha应该为16.0"
    assert configurator.dropout == 0.0, "dropout应该为0.0"
    assert len(configurator.target_modules) == 2, "target_modules应该有2个"
    print("✓ 测试1: 配置器初始化")


def test_pattern_matching():
    """测试2: 通配符模式匹配"""
    configurator = LoRAConfigurator(
        target_modules=["single_transformer_blocks.*.attn.to_q"],
        rank=16,
        verbose=False
    )

    # 应该匹配
    assert configurator._should_inject_lora("single_transformer_blocks.0.attn.to_q")
    assert configurator._should_inject_lora("single_transformer_blocks.1.attn.to_q")
    assert configurator._should_inject_lora("single_transformer_blocks.999.attn.to_q")

    # 不应该匹配
    assert not configurator._should_inject_lora("single_transformer_blocks.0.attn.to_k")
    assert not configurator._should_inject_lora("transformer_blocks.0.attn.to_q")
    assert not configurator._should_inject_lora("proj_out")

    print("✓ 测试2: 通配符模式匹配")


def test_pattern_matching_multiple():
    """测试3: 多个通配符模式"""
    configurator = LoRAConfigurator(
        target_modules=[
            "single_transformer_blocks.*.attn.to_q",
            "single_transformer_blocks.*.attn.to_k",
            "*.attn.to_v"
        ],
        rank=16,
        verbose=False
    )

    # 第一个模式匹配
    assert configurator._should_inject_lora("single_transformer_blocks.0.attn.to_q")
    # 第二个模式匹配
    assert configurator._should_inject_lora("single_transformer_blocks.0.attn.to_k")
    # 第三个模式匹配
    assert configurator._should_inject_lora("single_transformer_blocks.0.attn.to_v")
    assert configurator._should_inject_lora("transformer_blocks.0.attn.to_v")
    assert configurator._should_inject_lora("any_module.attn.to_v")

    # 不匹配
    assert not configurator._should_inject_lora("single_transformer_blocks.0.attn.to_out.0")

    print("✓ 测试3: 多个通配符模式")


def test_lora_injection_basic():
    """测试4: 基础LoRA注入"""
    model = create_test_model()
    configurator = LoRAConfigurator(
        target_modules=["single_transformer_blocks.0.attn.to_q"],
        rank=8,
        alpha=8.0,
        verbose=False
    )

    model, stats = configurator.inject_lora(model)

    # 检查统计信息
    assert stats.lora_injected == 1, "应该注入1个LoRA层"
    assert "single_transformer_blocks.0.attn.to_q" in stats.injected_module_names

    # 检查模块是否被替换为LoRA层
    assert isinstance(
        model.single_transformer_blocks[0].attn.to_q,
        StandardLoRALayer
    ), "to_q应该被替换为StandardLoRALayer"

    # 检查其他模块没有被替换
    assert isinstance(
        model.single_transformer_blocks[0].attn.to_k,
        nn.Linear
    ), "to_k应该仍然是nn.Linear"

    print("✓ 测试4: 基础LoRA注入")


def test_lora_injection_wildcard():
    """测试5: 使用通配符注入多个LoRA"""
    model = create_test_model()
    configurator = LoRAConfigurator(
        target_modules=["single_transformer_blocks.*.attn.to_q"],
        rank=8,
        verbose=False
    )

    model, stats = configurator.inject_lora(model)

    # 模型有3个single_transformer_blocks，每个都应该注入to_q
    assert stats.lora_injected == 3, f"应该注入3个LoRA层，实际注入了{stats.lora_injected}个"

    # 检查所有block的to_q都被替换
    for i in range(3):
        assert isinstance(
            model.single_transformer_blocks[i].attn.to_q,
            StandardLoRALayer
        ), f"block {i}的to_q应该被替换为StandardLoRALayer"

    print("✓ 测试5: 使用通配符注入多个LoRA")


def test_lora_injection_multiple_patterns():
    """测试6: 使用多个模式注入LoRA"""
    model = create_test_model()
    configurator = LoRAConfigurator(
        target_modules=[
            "single_transformer_blocks.*.attn.to_q",
            "single_transformer_blocks.*.attn.to_k",
            "single_transformer_blocks.*.attn.to_v"
        ],
        rank=8,
        verbose=False
    )

    model, stats = configurator.inject_lora(model)

    # 3个blocks * 3个attention层 = 9个LoRA
    assert stats.lora_injected == 9, f"应该注入9个LoRA层，实际注入了{stats.lora_injected}个"

    # 检查所有attention层都被替换
    for i in range(3):
        block = model.single_transformer_blocks[i]
        assert isinstance(block.attn.to_q, StandardLoRALayer)
        assert isinstance(block.attn.to_k, StandardLoRALayer)
        assert isinstance(block.attn.to_v, StandardLoRALayer)
        # to_out[0]没有被替换
        assert isinstance(block.attn.to_out[0], nn.Linear)

    print("✓ 测试6: 使用多个模式注入LoRA")


def test_parameter_freezing():
    """测试7: 参数冻结"""
    model = create_test_model()
    configurator = LoRAConfigurator(
        target_modules=["single_transformer_blocks.0.attn.to_q"],
        rank=8,
        verbose=False
    )

    model, stats = configurator.inject_lora(model)

    # 检查基础模型的参数被冻结
    base_layer = model.single_transformer_blocks[0].attn.to_q.base_layer
    for param in base_layer.parameters():
        assert not param.requires_grad, "基础层的参数应该被冻结"

    # 检查LoRA参数可训练
    lora_layer = model.single_transformer_blocks[0].attn.to_q
    assert lora_layer.lora_A.requires_grad, "lora_A应该可训练"
    assert lora_layer.lora_B.requires_grad, "lora_B应该可训练"

    # 检查未注入LoRA的模块参数被冻结
    for param in model.single_transformer_blocks[0].attn.to_k.parameters():
        assert not param.requires_grad, "未注入LoRA的模块应该被冻结"

    print("✓ 测试7: 参数冻结")


def test_get_trainable_parameters():
    """测试8: 获取可训练参数"""
    model = create_test_model()
    configurator = LoRAConfigurator(
        target_modules=["single_transformer_blocks.*.attn.to_q"],
        rank=8,
        verbose=False
    )

    model, stats = configurator.inject_lora(model)

    trainable_params = configurator.get_trainable_parameters(model)

    # 应该只有LoRA参数可训练
    # 3个blocks，每个都有to_q的lora_A和lora_B
    # 每个LoRA层：lora_A (128, 8) + lora_B (8, 128)
    expected_trainable_params = 3 * 2  # 3个LoRA层，每个2个参数（A和B）
    assert len(trainable_params) == expected_trainable_params, \
        f"应该有{expected_trainable_params}个可训练参数，实际有{len(trainable_params)}个"

    # 检查所有可训练参数都包含"lora"
    for param in trainable_params:
        # 通过检查形状判断是LoRA参数
        assert param.shape[0] == 128 or param.shape[1] == 128, "可训练参数应该是LoRA参数"

    print("✓ 测试8: 获取可训练参数")


def test_injection_stats():
    """测试9: 注入统计信息"""
    model = create_test_model()
    configurator = LoRAConfigurator(
        target_modules=[
            "single_transformer_blocks.*.attn.to_q",
            "single_transformer_blocks.*.attn.to_k"
        ],
        rank=8,
        verbose=False
    )

    model, stats = configurator.inject_lora(model)

    # 检查统计信息
    assert isinstance(stats, LoRAInjectionStats), "stats应该是LoRAInjectionStats类型"
    assert stats.lora_injected == 6, f"应该注入6个LoRA（3 blocks * 2 layers），实际{stats.lora_injected}"
    assert stats.moe_injected == 0, "不应该注入MoE-LoRA"
    assert stats.total_modules > 0, "总模块数应该大于0"
    assert len(stats.injected_module_names) == 6, "注入的模块名列表应该有6个"

    # 检查注入的模块名
    expected_names = [
        "single_transformer_blocks.0.attn.to_q",
        "single_transformer_blocks.0.attn.to_k",
        "single_transformer_blocks.1.attn.to_q",
        "single_transformer_blocks.1.attn.to_k",
        "single_transformer_blocks.2.attn.to_q",
        "single_transformer_blocks.2.attn.to_k",
    ]
    for name in expected_names:
        assert name in stats.injected_module_names, f"{name}应该在注入列表中"

    print("✓ 测试9: 注入统计信息")


# ============================================================================
# 第二部分：与其他组件的协作测试
# ============================================================================

def test_integration_with_config():
    """测试10: 与配置系统的集成"""
    # 创建配置
    config_dict = {
        'job_name': 'test_configurator',
        'device': 'cpu',
        'lora': {
            'rank': 16,
            'alpha': 16.0,
            'dropout': 0.0,
            'target_modules': [
                "single_transformer_blocks.*.attn.to_q",
                "single_transformer_blocks.*.attn.to_k"
            ]
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

    # 从配置创建configurator
    configurator = create_lora_configurator_from_config(config)

    # 检查configurator的参数
    assert configurator.rank == 16, "rank应该从config读取"
    assert configurator.alpha == 16.0, "alpha应该从config读取"
    assert configurator.dropout == 0.0, "dropout应该从config读取"
    assert len(configurator.target_modules) == 2, "target_modules应该从config读取"

    print("✓ 测试10: 与配置系统的集成")


def test_integration_with_standard_lora():
    """测试11: 与StandardLoRALayer的集成"""
    model = create_test_model()
    configurator = LoRAConfigurator(
        target_modules=["single_transformer_blocks.0.attn.to_q"],
        rank=8,
        alpha=16.0,
        dropout=0.1,
        verbose=False
    )

    model, stats = configurator.inject_lora(model)

    # 检查注入的LoRA层
    lora_layer = model.single_transformer_blocks[0].attn.to_q
    assert isinstance(lora_layer, StandardLoRALayer)

    # 检查LoRA层的参数
    assert lora_layer.rank == 8, "rank应该正确传递"
    assert lora_layer.alpha == 16.0, "alpha应该正确传递"
    assert lora_layer.scaling == 16.0 / 8, "scaling应该正确计算"

    # 检查LoRA层可以正常前向传播
    x = torch.randn(2, 10, 128)
    output = lora_layer(x)
    assert output.shape == (2, 10, 128), "输出形状应该正确"

    print("✓ 测试11: 与StandardLoRALayer的集成")


def test_end_to_end_from_config():
    """测试12: 端到端测试（配置文件 → 注入LoRA → 前向传播）"""
    # 步骤1: 创建配置
    config_dict = {
        'job_name': 'test_e2e',
        'device': 'cpu',
        'lora': {
            'rank': 8,
            'alpha': 8.0,
            'dropout': 0.0,
            'target_modules': [
                "single_transformer_blocks.*.attn.to_q",
                "single_transformer_blocks.*.attn.to_k",
                "single_transformer_blocks.*.attn.to_v",
            ]
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

    # 步骤2: 创建模型
    model = create_test_model()

    # 步骤3: 从配置创建configurator并注入LoRA
    configurator = create_lora_configurator_from_config(config)
    model, stats = configurator.inject_lora(model)

    # 步骤4: 验证注入
    assert stats.lora_injected == 9, "应该注入9个LoRA层（3 blocks * 3 layers）"

    # 步骤5: 前向传播测试
    x = torch.randn(2, 10, 128)
    output = model(x)
    assert output.shape == (2, 10, 128), "输出形状应该正确"

    # 步骤6: 检查requires_grad属性
    # 只有LoRA参数应该requires_grad=True
    lora_param_count = 0
    frozen_param_count = 0
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            assert param.requires_grad, f"{name}应该requires_grad=True"
            lora_param_count += 1
        else:
            # 基础模型的参数应该被冻结
            assert not param.requires_grad, f"{name}应该requires_grad=False（被冻结）"
            frozen_param_count += 1

    # 验证参数数量
    assert lora_param_count == 18, f"应该有18个LoRA参数（9层 * 2），实际{lora_param_count}"
    assert frozen_param_count > 0, "应该有冻结的参数"

    print("✓ 测试12: 端到端测试")


def test_trainable_parameter_statistics():
    """测试13: 可训练参数统计"""
    model = create_test_model()
    configurator = LoRAConfigurator(
        target_modules=["single_transformer_blocks.*.attn.to_q"],
        rank=8,
        verbose=False
    )

    # 注入前的参数数量
    total_params_before = sum(p.numel() for p in model.parameters())
    trainable_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 注入LoRA
    model, stats = configurator.inject_lora(model)

    # 注入后的参数数量
    total_params_after = sum(p.numel() for p in model.parameters())
    trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 总参数应该增加（添加了LoRA参数）
    assert total_params_after > total_params_before, "总参数应该增加"

    # 可训练参数应该远小于总参数（只有LoRA参数可训练）
    assert trainable_params_after < total_params_after * 0.1, \
        "可训练参数应该远小于总参数（< 10%）"

    # LoRA参数数量计算
    # 每个LoRA层：lora_A (128, 8) + lora_B (8, 128) = 1024 + 1024 = 2048
    # 3个LoRA层 = 6144
    expected_lora_params = 3 * (128 * 8 + 8 * 128)
    assert trainable_params_after == expected_lora_params, \
        f"可训练参数应该是{expected_lora_params}，实际是{trainable_params_after}"

    print("✓ 测试13: 可训练参数统计")


def test_lora_injection_correctness():
    """测试14: LoRA注入正确性（前向传播结果）"""
    model = create_test_model()

    # 保存原始权重的输出
    x = torch.randn(2, 10, 128)
    with torch.no_grad():
        original_q = model.single_transformer_blocks[0].attn.to_q(x).clone()

    # 注入LoRA
    configurator = LoRAConfigurator(
        target_modules=["single_transformer_blocks.0.attn.to_q"],
        rank=8,
        verbose=False
    )
    model, stats = configurator.inject_lora(model)

    # LoRA层的输出应该与原始层不同（因为添加了LoRA）
    with torch.no_grad():
        lora_q = model.single_transformer_blocks[0].attn.to_q(x)

    # 但是由于LoRA初始化后权重很小，差异应该不会太大
    diff = (lora_q - original_q).abs().max()
    assert diff < 1.0, "LoRA初始化后的输出差异应该不会太大"

    print("✓ 测试14: LoRA注入正确性")


def test_multiple_injections():
    """测试15: 多次注入（应该重置统计信息）"""
    model = create_test_model()
    configurator = LoRAConfigurator(
        target_modules=["single_transformer_blocks.0.attn.to_q"],
        rank=8,
        verbose=False
    )

    # 第一次注入
    model1, stats1 = configurator.inject_lora(model)
    assert stats1.lora_injected == 1

    # 创建新模型，第二次注入
    model2 = create_test_model()
    model2, stats2 = configurator.inject_lora(model2)
    assert stats2.lora_injected == 1, "第二次注入的统计应该重置"

    print("✓ 测试15: 多次注入")


# ============================================================================
# 运行所有测试
# ============================================================================

def main():
    """运行所有测试"""
    print("=" * 60)
    print("开始LoRA配置器测试")
    print("=" * 60)

    # 第一部分：配置器本身功能测试
    print("\n【第一部分：配置器本身功能测试】")
    test_configurator_initialization()
    test_pattern_matching()
    test_pattern_matching_multiple()
    test_lora_injection_basic()
    test_lora_injection_wildcard()
    test_lora_injection_multiple_patterns()
    test_parameter_freezing()
    test_get_trainable_parameters()
    test_injection_stats()

    # 第二部分：与其他组件的协作测试
    print("\n【第二部分：与其他组件的协作测试】")
    test_integration_with_config()
    test_integration_with_standard_lora()
    test_end_to_end_from_config()
    test_trainable_parameter_statistics()
    test_lora_injection_correctness()
    test_multiple_injections()

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
