"""
MoE-LoRA层简单测试
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.lora.moe.moe_lora_layer import MoELoRALayer


def test_single_route_mode():
    """测试单路由模式"""
    print("\n=== 测试单路由模式 MoE-LoRA ===")

    # 创建基础层
    base_layer = nn.Linear(128, 256)

    # 创建MoE-LoRA层（单路由模式）
    moe_lora = MoELoRALayer(
        base_layer=base_layer,
        num_experts=4,
        rank=16,
        alpha=16.0,
        routing_mode="single",
        router_config={'top_k': 2}
    )

    print(f"基础层: {base_layer.in_features} -> {base_layer.out_features}")
    print(f"专家数量: {moe_lora.num_experts}")
    print(f"Rank: {moe_lora.rank}")
    print(f"路由模式: {moe_lora.routing_mode}")

    # 测试2D输入
    x_2d = torch.randn(8, 128)
    output_2d = moe_lora(x_2d)

    print(f"\n2D输入形状: {x_2d.shape}")
    print(f"2D输出形状: {output_2d.shape}")

    assert output_2d.shape == (8, 256), f"2D输出形状错误: {output_2d.shape}"

    # 测试3D输入
    x_3d = torch.randn(8, 16, 128)
    output_3d = moe_lora(x_3d)

    print(f"\n3D输入形状: {x_3d.shape}")
    print(f"3D输出形状: {output_3d.shape}")

    assert output_3d.shape == (8, 16, 256), f"3D输出形状错误: {output_3d.shape}"

    # 测试路由统计
    moe_lora.reset_routing_stats()
    moe_lora.train()

    x = torch.randn(10, 5, 128)
    output = moe_lora(x)

    stats = moe_lora.get_routing_stats()
    print(f"\n路由统计:")
    print(f"  总调用次数: {stats['total_calls']}")
    print(f"  专家使用次数: {stats['expert_usage']}")
    print(f"  专家使用率: {[f'{r:.3f}' for r in stats['expert_usage_ratio']]}")

    assert stats['total_calls'] == 20, f"调用次数错误: {stats['total_calls']}"

    print("✅ 单路由模式测试通过")


def test_dual_route_mode():
    """测试双路由模式"""
    print("\n=== 测试双路由模式 MoE-LoRA ===")

    # 创建基础层
    base_layer = nn.Linear(128, 256)

    # 创建MoE-LoRA层（双路由模式）
    # 6个专家：前3个是Specific专家，后3个是Shared专家
    moe_lora = MoELoRALayer(
        base_layer=base_layer,
        num_experts=6,
        rank=16,
        alpha=16.0,
        routing_mode="dual",
        router_config={
            'num_specific': 3,
            'shared_top_k': 2,
            'shared_hidden_dim': 256
        }
    )

    print(f"基础层: {base_layer.in_features} -> {base_layer.out_features}")
    print(f"专家数量: {moe_lora.num_experts}")
    print(f"Specific专家: 0-2")
    print(f"Shared专家: 3-5")
    print(f"路由模式: {moe_lora.routing_mode}")

    # 测试前向传播（需要exposure_category）
    x = torch.randn(12, 10, 128)
    exposure_category = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    output = moe_lora(x, exposure_category=exposure_category)

    print(f"\n输入形状: {x.shape}")
    print(f"曝光度类别: {exposure_category}")
    print(f"输出形状: {output.shape}")

    assert output.shape == (12, 10, 256), f"输出形状错误: {output.shape}"

    # 测试缺少exposure_category时报错
    try:
        output = moe_lora(x)
        assert False, "应该抛出ValueError"
    except ValueError as e:
        print(f"\n✅ 正确抛出错误: {e}")

    # 测试路由统计
    moe_lora.reset_routing_stats()
    moe_lora.train()

    output = moe_lora(x, exposure_category=exposure_category)

    stats = moe_lora.get_routing_stats()
    print(f"\n路由统计:")
    print(f"Specific路由:")
    print(f"  总调用次数: {stats['specific']['total_calls']}")
    print(f"  专家使用次数: {stats['specific']['expert_usage']}")
    print(f"  专家使用率: {[f'{r:.3f}' for r in stats['specific']['expert_usage_ratio']]}")

    print(f"Shared路由:")
    print(f"  总调用次数: {stats['shared']['total_calls']}")
    print(f"  专家使用次数: {stats['shared']['expert_usage']}")
    print(f"  专家使用率: {[f'{r:.3f}' for r in stats['shared']['expert_usage_ratio']]}")

    assert stats['specific']['total_calls'] == 12, "Specific调用次数错误"
    assert stats['shared']['total_calls'] == 24, "Shared调用次数错误"  # 12 * 2(top_k)

    print("✅ 双路由模式测试通过")


def test_gradient_flow():
    """测试梯度流"""
    print("\n=== 测试梯度流 ===")

    base_layer = nn.Linear(64, 128)

    moe_lora = MoELoRALayer(
        base_layer=base_layer,
        num_experts=4,
        rank=8,
        alpha=8.0,
        routing_mode="single",
        router_config={'top_k': 4}  # 选择所有专家确保都被使用
    )

    # 创建带梯度的输入，增大batch size确保所有专家都被选中
    x = torch.randn(20, 10, 64, requires_grad=True)

    # 前向传播
    output = moe_lora(x)

    # 计算损失并反向传播
    loss = output.sum()
    loss.backward()

    # 检查输入梯度
    assert x.grad is not None, "输入梯度为None"
    assert not torch.all(x.grad == 0), "输入梯度全为0"

    # 检查LoRA参数梯度（由于top_k=4，所有专家都应该被选中）
    lora_params = moe_lora.get_lora_parameters()
    print(f"LoRA参数数量: {len(lora_params)}")

    grad_count = 0
    for i, param in enumerate(lora_params):
        if param.grad is not None:
            grad_count += 1

    print(f"有梯度的参数: {grad_count}/{len(lora_params)}")
    assert grad_count > 0, "没有任何LoRA参数有梯度"

    # 检查Router参数梯度
    router_params = list(moe_lora.router.parameters())
    print(f"Router参数数量: {len(router_params)}")

    for param in router_params:
        assert param.grad is not None, "Router参数梯度为None"

    print("✅ 梯度流测试通过")


def test_device_compatibility():
    """测试设备兼容性"""
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA不可用，跳过CUDA测试")
        return

    print("\n=== 测试 CUDA 兼容性 ===")

    # 创建基础层并移到CUDA
    base_layer = nn.Linear(128, 256).cuda()

    # 创建MoE-LoRA层并移到CUDA
    moe_lora = MoELoRALayer(
        base_layer=base_layer,
        num_experts=4,
        rank=16,
        alpha=16.0,
        routing_mode="single",
        router_config={'top_k': 2}
    ).cuda()

    # 验证LoRA参数在CUDA上
    for i in range(moe_lora.num_experts):
        assert moe_lora.lora_A[i].device.type == 'cuda', f"lora_A[{i}]不在CUDA上"
        assert moe_lora.lora_B[i].device.type == 'cuda', f"lora_B[{i}]不在CUDA上"

    # 验证Router在CUDA上
    for param in moe_lora.router.parameters():
        assert param.device.type == 'cuda', "Router参数不在CUDA上"

    # 测试前向传播
    x = torch.randn(4, 10, 128).cuda()
    output = moe_lora(x)

    assert output.device.type == 'cuda', "输出不在CUDA上"
    assert output.shape == (4, 10, 256), f"输出形状错误: {output.shape}"

    print("✅ CUDA兼容性测试通过")


def test_initialization_methods():
    """测试不同的初始化方法"""
    print("\n=== 测试初始化方法 ===")

    base_layer = nn.Linear(64, 128)

    for init_method in ['kaiming', 'normal', 'zero']:
        moe_lora = MoELoRALayer(
            base_layer=base_layer,
            num_experts=2,
            rank=8,
            alpha=8.0,
            routing_mode="single",
            router_config={'top_k': 1},
            init_method=init_method
        )

        print(f"\n初始化方法: {init_method}")

        # 检查初始化
        for i in range(moe_lora.num_experts):
            lora_A = moe_lora.lora_A[i].data
            lora_B = moe_lora.lora_B[i].data

            if init_method == 'zero':
                # 全零初始化
                assert torch.all(lora_A == 0), f"lora_A[{i}]不全为0"
                assert torch.all(lora_B == 0), f"lora_B[{i}]不全为0"
            else:
                # 其他初始化方法，lora_B应该为0，lora_A不为0
                assert torch.all(lora_B == 0), f"lora_B[{i}]不全为0"
                assert not torch.all(lora_A == 0), f"lora_A[{i}]全为0"

        print(f"✅ {init_method} 初始化正确")


def test_dropout():
    """测试Dropout"""
    print("\n=== 测试Dropout ===")

    base_layer = nn.Linear(64, 128)

    moe_lora = MoELoRALayer(
        base_layer=base_layer,
        num_experts=4,
        rank=8,
        alpha=8.0,
        dropout=0.5,  # 使用更高的dropout概率
        routing_mode="single",
        router_config={'top_k': 2},
        init_method='normal'  # 使用normal初始化，lora_B也会被初始化但然后设为0，所以手动设置
    )

    # 手动设置lora_B为非零，以便dropout能有效果
    for i in range(moe_lora.num_experts):
        nn.init.normal_(moe_lora.lora_B[i], std=0.02)

    assert moe_lora.dropout_layer is not None, "Dropout层未创建"

    # 训练模式下，多次前向传播应该有不同结果（由于dropout）
    moe_lora.train()
    x = torch.randn(20, 10, 64)  # 增大batch size

    outputs = []
    for _ in range(10):
        outputs.append(moe_lora(x))

    # 检查至少有一些输出是不同的
    different_count = 0
    for out in outputs[1:]:
        if not torch.allclose(outputs[0], out, atol=1e-6):
            different_count += 1

    print(f"不同输出数量: {different_count}/9")
    assert different_count > 0, "所有输出都相同，Dropout可能未生效"

    # 评估模式下，多次前向传播应该有相同结果
    moe_lora.eval()
    outputs_eval = []
    for _ in range(3):
        outputs_eval.append(moe_lora(x))

    for i, out in enumerate(outputs_eval[1:], 1):
        assert torch.equal(outputs_eval[0], out), f"评估模式下输出{i}与输出0不同"

    print("✅ Dropout测试通过")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("开始MoE-LoRA层测试")
    print("=" * 60)

    try:
        test_single_route_mode()
        test_dual_route_mode()
        test_gradient_flow()
        test_device_compatibility()
        test_initialization_methods()
        test_dropout()

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
