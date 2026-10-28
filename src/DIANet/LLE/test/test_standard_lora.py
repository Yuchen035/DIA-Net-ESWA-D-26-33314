"""
标准LoRA层测试

测试覆盖：
1. LoRA层创建和初始化
2. 前向传播（正确性、形状）
3. 参数冻结检查
4. 可训练参数获取
5. 权重合并
6. 参数重置
7. 边界情况和错误处理
8. 工厂函数
9. 批量替换功能
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.lora.standard.lora_layer import (
    StandardLoRALayer,
    create_lora_layer,
    replace_linear_with_lora,
)
from core.lora.base.base_lora import BaseLoRALayer


class TestStandardLoRALayer:
    """测试StandardLoRALayer类"""

    def test_initialization(self):
        """测试LoRA层初始化"""
        print("测试1: LoRA层初始化...")
        base_layer = nn.Linear(512, 1024, bias=True)
        lora_layer = StandardLoRALayer(
            base_layer=base_layer,
            rank=16,
            alpha=16.0,
            dropout=0.1,
        )

        # 检查属性
        assert lora_layer.in_features == 512
        assert lora_layer.out_features == 1024
        assert lora_layer.rank == 16
        assert lora_layer.alpha == 16.0
        assert lora_layer.dropout == 0.1
        assert lora_layer.scaling == 1.0  # alpha / rank = 16 / 16 = 1
        assert lora_layer.is_initialized

        # 检查LoRA权重形状
        assert lora_layer.lora_A.shape == (512, 16)
        assert lora_layer.lora_B.shape == (16, 1024)

        # 检查基础层被冻结
        for param in base_layer.parameters():
            assert not param.requires_grad

        print("✓ LoRA层初始化正确")

    def test_invalid_base_layer(self):
        """测试无效的基础层类型"""
        print("\n测试2: 无效的基础层类型...")
        invalid_layer = nn.Conv2d(3, 64, 3)
        try:
            StandardLoRALayer(base_layer=invalid_layer, rank=16, alpha=16.0)
            assert False, "应该抛出ValueError"
        except ValueError as e:
            assert "only supports nn.Linear" in str(e)
            print(f"✓ 正确捕获无效基础层: {e}")

    def test_invalid_rank(self):
        """测试无效的rank值"""
        print("\n测试3: 无效的rank值...")
        base_layer = nn.Linear(512, 1024)

        # 测试负数rank
        try:
            StandardLoRALayer(base_layer=base_layer, rank=-1, alpha=16.0)
            assert False, "应该抛出ValueError"
        except ValueError as e:
            assert "rank must be positive" in str(e)
            print(f"✓ 正确捕获负数rank: {e}")

        # 测试零rank
        try:
            StandardLoRALayer(base_layer=base_layer, rank=0, alpha=16.0)
            assert False, "应该抛出ValueError"
        except ValueError as e:
            assert "rank must be positive" in str(e)
            print(f"✓ 正确捕获零rank: {e}")

    def test_invalid_alpha(self):
        """测试无效的alpha值"""
        print("\n测试4: 无效的alpha值...")
        base_layer = nn.Linear(512, 1024)

        try:
            StandardLoRALayer(base_layer=base_layer, rank=16, alpha=-1.0)
            assert False, "应该抛出ValueError"
        except ValueError as e:
            assert "alpha must be positive" in str(e)
            print(f"✓ 正确捕获负数alpha: {e}")

    def test_invalid_dropout(self):
        """测试无效的dropout值"""
        print("\n测试5: 无效的dropout值...")
        base_layer = nn.Linear(512, 1024)

        # dropout = 1.0 应该失败
        try:
            StandardLoRALayer(base_layer=base_layer, rank=16, alpha=16.0, dropout=1.0)
            assert False, "应该抛出ValueError"
        except ValueError as e:
            assert "dropout must be in [0, 1)" in str(e)
            print(f"✓ 正确捕获无效dropout=1.0: {e}")

        # dropout = -0.1 应该失败
        try:
            StandardLoRALayer(base_layer=base_layer, rank=16, alpha=16.0, dropout=-0.1)
            assert False, "应该抛出ValueError"
        except ValueError as e:
            assert "dropout must be in [0, 1)" in str(e)
            print(f"✓ 正确捕获无效dropout=-0.1: {e}")

    def test_forward_pass_shape(self):
        """测试前向传播的输出形状"""
        print("\n测试6: 前向传播形状...")
        base_layer = nn.Linear(512, 1024)
        lora_layer = StandardLoRALayer(base_layer=base_layer, rank=16, alpha=16.0)

        # 测试2D输入
        x = torch.randn(4, 512)
        output = lora_layer(x)
        assert output.shape == (4, 1024), f"Expected (4, 1024), got {output.shape}"
        print("✓ 2D输入形状正确")

        # 测试3D输入
        x = torch.randn(2, 8, 512)
        output = lora_layer(x)
        assert output.shape == (2, 8, 1024), f"Expected (2, 8, 1024), got {output.shape}"
        print("✓ 3D输入形状正确")

    def test_forward_pass_correctness(self):
        """测试前向传播的计算正确性"""
        print("\n测试7: 前向传播计算正确性...")
        base_layer = nn.Linear(128, 256, bias=False)
        lora_layer = StandardLoRALayer(base_layer=base_layer, rank=8, alpha=8.0)

        x = torch.randn(2, 128)

        # 手动计算预期输出
        with torch.no_grad():
            base_output = base_layer(x)
            lora_output = (x @ lora_layer.lora_A) @ lora_layer.lora_B
            expected = base_output + lora_layer.scaling * lora_output

        # 实际输出
        actual = lora_layer(x)

        # 检查是否相等
        assert torch.allclose(actual, expected, atol=1e-6), "前向传播计算不正确"
        print("✓ 前向传播计算正确")

    def test_trainable_parameters(self):
        """测试可训练参数获取"""
        print("\n测试8: 可训练参数...")
        base_layer = nn.Linear(512, 1024)
        lora_layer = StandardLoRALayer(base_layer=base_layer, rank=16, alpha=16.0)

        trainable_params = list(lora_layer.get_trainable_parameters())
        assert len(trainable_params) == 2, f"Expected 2 trainable params, got {len(trainable_params)}"
        assert trainable_params[0] is lora_layer.lora_A
        assert trainable_params[1] is lora_layer.lora_B

        # 检查参数可训练
        assert trainable_params[0].requires_grad
        assert trainable_params[1].requires_grad

        print("✓ 可训练参数正确")

    def test_base_layer_frozen(self):
        """测试基础层被冻结"""
        print("\n测试9: 基础层冻结...")
        base_layer = nn.Linear(512, 1024, bias=True)
        lora_layer = StandardLoRALayer(base_layer=base_layer, rank=16, alpha=16.0)

        for param in base_layer.parameters():
            assert not param.requires_grad, "基础层参数应该被冻结"

        print("✓ 基础层正确冻结")

    def test_merge_weights(self):
        """测试权重合并"""
        print("\n测试10: 权重合并...")
        base_layer = nn.Linear(128, 256, bias=True)
        lora_layer = StandardLoRALayer(base_layer=base_layer, rank=8, alpha=8.0)

        # 合并权重
        merged_layer = lora_layer.merge_weights()

        # 检查merged_layer类型
        assert isinstance(merged_layer, nn.Linear)
        assert merged_layer.in_features == 128
        assert merged_layer.out_features == 256

        # 检查合并后的输出与LoRA层相同
        x = torch.randn(4, 128)
        with torch.no_grad():
            lora_output = lora_layer(x)
            merged_output = merged_layer(x)

        assert torch.allclose(lora_output, merged_output, atol=1e-5), "合并后输出不一致"
        print("✓ 权重合并正确")

    def test_reset_parameters(self):
        """测试参数重置"""
        print("\n测试11: 参数重置...")
        base_layer = nn.Linear(512, 1024)
        lora_layer = StandardLoRALayer(base_layer=base_layer, rank=16, alpha=16.0)

        # 保存初始值
        lora_A_before = lora_layer.lora_A.data.clone()
        lora_B_before = lora_layer.lora_B.data.clone()

        # 修改参数
        lora_layer.lora_A.data.fill_(1.0)
        lora_layer.lora_B.data.fill_(2.0)

        # 重置参数
        lora_layer.reset_lora_parameters(init_method="zero")

        # 检查是否重置为0
        assert torch.allclose(lora_layer.lora_A, torch.zeros_like(lora_layer.lora_A))
        assert torch.allclose(lora_layer.lora_B, torch.zeros_like(lora_layer.lora_B))

        print("✓ 参数重置正确")

    def test_different_ranks(self):
        """测试不同的rank值"""
        print("\n测试12: 不同rank值...")
        base_layer = nn.Linear(512, 1024)

        for rank in [4, 8, 16, 32, 64]:
            lora_layer = StandardLoRALayer(base_layer=base_layer, rank=rank, alpha=float(rank))
            assert lora_layer.lora_A.shape == (512, rank)
            assert lora_layer.lora_B.shape == (rank, 1024)
            assert lora_layer.scaling == 1.0  # alpha / rank

        print("✓ 不同rank值正确")

    def test_different_scaling(self):
        """测试不同的scaling值"""
        print("\n测试13: 不同scaling值...")
        base_layer = nn.Linear(512, 1024)

        # alpha != rank，scaling应该不为1
        lora_layer = StandardLoRALayer(base_layer=base_layer, rank=16, alpha=32.0)
        assert lora_layer.scaling == 2.0  # 32 / 16

        lora_layer = StandardLoRALayer(base_layer=base_layer, rank=32, alpha=16.0)
        assert lora_layer.scaling == 0.5  # 16 / 32

        print("✓ 不同scaling值正确")

    def test_repr(self):
        """测试__repr__方法"""
        print("\n测试14: __repr__方法...")
        base_layer = nn.Linear(512, 1024)
        lora_layer = StandardLoRALayer(base_layer=base_layer, rank=16, alpha=16.0, dropout=0.1)

        repr_str = repr(lora_layer)
        assert "StandardLoRALayer" in repr_str
        assert "512" in repr_str  # in_features
        assert "1024" in repr_str  # out_features
        assert "16" in repr_str  # rank
        assert "0.1" in repr_str  # dropout

        print("✓ __repr__方法正确")


class TestCreateLoRALayer:
    """测试create_lora_layer工厂函数"""

    def test_factory_function(self):
        """测试工厂函数"""
        print("\n测试15: 工厂函数...")
        base_layer = nn.Linear(512, 1024)
        lora_layer = create_lora_layer(base_layer, rank=8, alpha=8.0)

        assert isinstance(lora_layer, StandardLoRALayer)
        assert lora_layer.rank == 8
        assert lora_layer.alpha == 8.0

        print("✓ 工厂函数正确")


class TestReplaceLinearWithLoRA:
    """测试replace_linear_with_lora函数"""

    def test_replace_all_layers(self):
        """测试替换所有Linear层"""
        print("\n测试16: 替换所有Linear层...")

        # 创建简单模型
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
        )

        # 替换所有层
        num_replaced = replace_linear_with_lora(model, ["*"], rank=16, alpha=16.0)

        assert num_replaced == 3, f"Expected 3 replaced, got {num_replaced}"

        # 检查所有Linear层都被替换
        linear_count = 0
        lora_count = 0
        for module in model.modules():
            if isinstance(module, nn.Linear) and not isinstance(module, StandardLoRALayer):
                # 这应该是被包装在StandardLoRALayer内的base_layer
                pass
            elif isinstance(module, StandardLoRALayer):
                lora_count += 1

        assert lora_count == 3, f"Expected 3 LoRA layers, got {lora_count}"

        print("✓ 替换所有Linear层正确")

    def test_replace_specific_layers(self):
        """测试替换特定层"""
        print("\n测试17: 替换特定层...")

        # 创建带命名层的模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(128, 256)
                self.layer2 = nn.Linear(256, 512)
                self.layer3 = nn.Linear(512, 1024)

        model = SimpleModel()

        # 只替换layer1和layer3
        num_replaced = replace_linear_with_lora(
            model, ["layer1", "layer3"], rank=16, alpha=16.0
        )

        assert num_replaced == 2, f"Expected 2 replaced, got {num_replaced}"

        # 检查
        assert isinstance(model.layer1, StandardLoRALayer)
        assert isinstance(model.layer2, nn.Linear) and not isinstance(model.layer2, StandardLoRALayer)
        assert isinstance(model.layer3, StandardLoRALayer)

        print("✓ 替换特定层正确")

    def test_replace_with_wildcard(self):
        """测试使用通配符替换"""
        print("\n测试18: 使用通配符替换...")

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_q = nn.Linear(128, 128)
                self.attn_k = nn.Linear(128, 128)
                self.attn_v = nn.Linear(128, 128)
                self.ffn = nn.Linear(128, 512)

        model = Model()

        # 只替换attn_*层
        num_replaced = replace_linear_with_lora(
            model, ["attn_*"], rank=16, alpha=16.0
        )

        assert num_replaced == 3, f"Expected 3 replaced, got {num_replaced}"

        assert isinstance(model.attn_q, StandardLoRALayer)
        assert isinstance(model.attn_k, StandardLoRALayer)
        assert isinstance(model.attn_v, StandardLoRALayer)
        assert not isinstance(model.ffn, StandardLoRALayer)

        print("✓ 通配符替换正确")

    def test_replace_nested_modules(self):
        """测试替换嵌套模块"""
        print("\n测试19: 替换嵌套模块...")

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(128, 256)
                self.linear2 = nn.Linear(256, 128)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = Block()
                self.block2 = Block()

        model = Model()

        # 替换所有Linear层
        num_replaced = replace_linear_with_lora(model, ["*"], rank=16, alpha=16.0)

        assert num_replaced == 4, f"Expected 4 replaced, got {num_replaced}"

        print("✓ 嵌套模块替换正确")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("开始标准LoRA层测试")
    print("=" * 60)

    try:
        # 测试StandardLoRALayer
        test_lora = TestStandardLoRALayer()
        test_lora.test_initialization()
        test_lora.test_invalid_base_layer()
        test_lora.test_invalid_rank()
        test_lora.test_invalid_alpha()
        test_lora.test_invalid_dropout()
        test_lora.test_forward_pass_shape()
        test_lora.test_forward_pass_correctness()
        test_lora.test_trainable_parameters()
        test_lora.test_base_layer_frozen()
        test_lora.test_merge_weights()
        test_lora.test_reset_parameters()
        test_lora.test_different_ranks()
        test_lora.test_different_scaling()
        test_lora.test_repr()

        # 测试工厂函数
        test_factory = TestCreateLoRALayer()
        test_factory.test_factory_function()

        # 测试批量替换
        test_replace = TestReplaceLinearWithLoRA()
        test_replace.test_replace_all_layers()
        test_replace.test_replace_specific_layers()
        test_replace.test_replace_with_wildcard()
        test_replace.test_replace_nested_modules()

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
