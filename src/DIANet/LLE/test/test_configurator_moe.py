"""
测试LoRAConfigurator的MoE-LoRA支持
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.lora.configurator import LoRAConfigurator
from core.lora.moe.moe_lora_layer import MoELoRALayer
from core.lora.standard.lora_layer import StandardLoRALayer


def create_simple_model():
    """创建一个简单的测试模型"""
    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_q = nn.Linear(64, 64)
            self.attn_k = nn.Linear(64, 64)
            self.attn_v = nn.Linear(64, 64)
            self.attn_out = nn.Linear(64, 64)
            self.ff1 = nn.Linear(64, 256)
            self.ff2 = nn.Linear(256, 64)
            self.proj_out = nn.Linear(64, 64)

        def forward(self, x):
            # 简单的前向传播
            q = self.attn_q(x)
            k = self.attn_k(x)
            v = self.attn_v(x)
            attn_out = self.attn_out(v)
            ff_out = self.ff2(self.ff1(attn_out))
            return self.proj_out(ff_out)

    return SimpleTransformer()


def test_standard_lora_injection():
    """测试标准LoRA注入（兼容旧API）"""
    print("\n=== 测试标准LoRA注入（兼容旧API）===")

    model = create_simple_model()

    # 使用旧API：只提供target_modules
    configurator = LoRAConfigurator(
        target_modules=["attn_q", "attn_k"],
        rank=8,
        alpha=8.0,
        verbose=False
    )

    model, stats = configurator.inject_lora(model)

    print(f"总模块数: {stats.total_modules}")
    print(f"注入标准LoRA: {stats.lora_injected}")
    print(f"注入MoE-LoRA: {stats.moe_injected}")
    print(f"冻结模块: {stats.frozen}")

    # 验证
    assert stats.lora_injected == 2, f"应该注入2个LoRA，实际: {stats.lora_injected}"
    assert stats.moe_injected == 0, f"不应该注入MoE-LoRA，实际: {stats.moe_injected}"
    assert isinstance(model.attn_q, StandardLoRALayer), "attn_q应该是StandardLoRALayer"
    assert isinstance(model.attn_k, StandardLoRALayer), "attn_k应该是StandardLoRALayer"
    assert isinstance(model.attn_v, nn.Linear), "attn_v应该保持为nn.Linear"

    print("✅ 标准LoRA注入测试通过")


def test_moe_lora_injection():
    """测试MoE-LoRA注入"""
    print("\n=== 测试MoE-LoRA注入 ===")

    model = create_simple_model()

    # 使用新API：使用module_modes指定MoE
    configurator = LoRAConfigurator(
        module_modes={
            "attn_q": "moe",
            "attn_k": "moe",
            "attn_v": "lora",
            "attn_out": "freeze"
        },
        rank=8,
        alpha=8.0,
        moe_config={
            "num_experts": 4,
            "routing_mode": "single",
            "router_config": {"top_k": 2}
        },
        verbose=False
    )

    model, stats = configurator.inject_lora(model)

    print(f"总模块数: {stats.total_modules}")
    print(f"注入标准LoRA: {stats.lora_injected}")
    print(f"注入MoE-LoRA: {stats.moe_injected}")
    print(f"冻结模块: {stats.frozen}")

    # 验证
    assert stats.moe_injected == 2, f"应该注入2个MoE-LoRA，实际: {stats.moe_injected}"
    assert stats.lora_injected == 1, f"应该注入1个标准LoRA，实际: {stats.lora_injected}"
    assert isinstance(model.attn_q, MoELoRALayer), "attn_q应该是MoELoRALayer"
    assert isinstance(model.attn_k, MoELoRALayer), "attn_k应该是MoELoRALayer"
    assert isinstance(model.attn_v, StandardLoRALayer), "attn_v应该是StandardLoRALayer"
    assert isinstance(model.attn_out, nn.Linear), "attn_out应该保持为nn.Linear（冻结）"

    # 验证MoE-LoRA参数
    assert model.attn_q.num_experts == 4, "num_experts应该是4"
    assert model.attn_q.routing_mode == "single", "routing_mode应该是single"

    print("✅ MoE-LoRA注入测试通过")


def test_mixed_injection():
    """测试混合注入（LoRA + MoE-LoRA + Freeze）"""
    print("\n=== 测试混合注入 ===")

    model = create_simple_model()

    configurator = LoRAConfigurator(
        module_modes={
            "attn_q": "moe",
            "attn_k": "lora",
            "attn_v": "lora",
            "attn_out": "moe",
            "ff1": "lora",
            "ff2": "freeze",
            "proj_out": "freeze"
        },
        rank=16,
        alpha=16.0,
        moe_config={
            "num_experts": 6,
            "routing_mode": "dual",
            "router_config": {
                "num_specific": 3,
                "shared_top_k": 2
            }
        },
        verbose=False
    )

    model, stats = configurator.inject_lora(model)

    print(f"总模块数: {stats.total_modules}")
    print(f"注入标准LoRA: {stats.lora_injected}")
    print(f"注入MoE-LoRA: {stats.moe_injected}")
    print(f"冻结模块: {stats.frozen}")
    print(f"MoE模块: {stats.moe_module_names}")

    # 验证
    assert stats.lora_injected == 3, f"应该注入3个标准LoRA，实际: {stats.lora_injected}"
    assert stats.moe_injected == 2, f"应该注入2个MoE-LoRA，实际: {stats.moe_injected}"

    # 验证每个模块的类型
    assert isinstance(model.attn_q, MoELoRALayer), "attn_q应该是MoELoRALayer"
    assert isinstance(model.attn_k, StandardLoRALayer), "attn_k应该是StandardLoRALayer"
    assert isinstance(model.attn_v, StandardLoRALayer), "attn_v应该是StandardLoRALayer"
    assert isinstance(model.attn_out, MoELoRALayer), "attn_out应该是MoELoRALayer"
    assert isinstance(model.ff1, StandardLoRALayer), "ff1应该是StandardLoRALayer"
    assert isinstance(model.ff2, nn.Linear), "ff2应该保持为nn.Linear（冻结）"
    assert isinstance(model.proj_out, nn.Linear), "proj_out应该保持为nn.Linear（冻结）"

    # 验证双路由模式
    assert model.attn_q.routing_mode == "dual", "routing_mode应该是dual"
    assert model.attn_q.num_experts == 6, "num_experts应该是6"
    assert model.attn_q.num_specific == 3, "num_specific应该是3"

    print("✅ 混合注入测试通过")


def test_forward_pass():
    """测试前向传播"""
    print("\n=== 测试前向传播 ===")

    model = create_simple_model()

    configurator = LoRAConfigurator(
        module_modes={
            "attn_q": "moe",
            "attn_k": "lora",
        },
        rank=8,
        alpha=8.0,
        moe_config={
            "num_experts": 4,
            "routing_mode": "single",
            "router_config": {"top_k": 2}
        },
        verbose=False
    )

    model, stats = configurator.inject_lora(model)

    # 前向传播测试
    x = torch.randn(4, 10, 64)
    output = model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    assert output.shape == x.shape, f"输出形状应该与输入相同，实际: {output.shape}"

    print("✅ 前向传播测试通过")


def test_trainable_parameters():
    """测试可训练参数统计"""
    print("\n=== 测试可训练参数统计 ===")

    model = create_simple_model()

    # 先统计原始参数
    original_params = sum(p.numel() for p in model.parameters())
    print(f"原始模型参数: {original_params:,}")

    configurator = LoRAConfigurator(
        module_modes={
            "attn_q": "moe",
            "attn_k": "lora",
        },
        rank=8,
        alpha=8.0,
        moe_config={
            "num_experts": 4,
            "routing_mode": "single",
            "router_config": {"top_k": 2}
        },
        verbose=False
    )

    model, stats = configurator.inject_lora(model)

    # 统计注入后的可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"注入后总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"可训练比例: {100 * trainable_params / total_params:.2f}%")

    # MoE-LoRA的参数应该比标准LoRA多（因为有多个专家）
    assert trainable_params > 0, "应该有可训练参数"
    assert trainable_params < total_params, "不应该所有参数都可训练"

    print("✅ 可训练参数统计测试通过")


def main():
    """运行所有测试"""
    print("=" * 70)
    print("开始测试LoRAConfigurator的MoE-LoRA支持")
    print("=" * 70)

    try:
        test_standard_lora_injection()
        test_moe_lora_injection()
        test_mixed_injection()
        test_forward_pass()
        test_trainable_parameters()

        print("\n" + "=" * 70)
        print("✅ 所有测试通过！")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ 测试失败: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
