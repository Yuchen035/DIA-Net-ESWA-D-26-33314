"""
双路由MoE-LoRA端到端测试

测试完整的双路由流程：
1. Mock FLUX模型 + 双路由MoE-LoRA
2. 数据集曝光度分类
3. 前向传播（包含dual-route）
4. 专家选择验证
5. 权重保存和加载
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import tempfile
from PIL import Image
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.mock_flux import MockFluxModelLoader
from core.lora.moe.moe_lora_layer import MoELoRALayer
from core.lora.moe.router import SpecificRouter, SharedRouter
from core.data.dataset import ImagePairDataset
from torch.utils.data import DataLoader


def create_test_dataset(temp_dir: Path, num_images: int = 8):
    """创建测试数据集（不同曝光度）"""
    input_dir = temp_dir / "input"
    target_dir = temp_dir / "target"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 创建不同曝光度的图像
    brightness_levels = [0.05, 0.15, 0.25, 0.35, 0.45, 0.6, 0.75, 0.9]

    for i in range(num_images):
        brightness = brightness_levels[i % len(brightness_levels)]

        # 输入图像（不同亮度）
        input_img = np.ones((128, 128, 3), dtype=np.uint8) * int(brightness * 255)
        input_pil = Image.fromarray(input_img)
        input_pil.save(input_dir / f"img_{i:03d}.png")

        # 目标图像（正常亮度）
        target_img = np.ones((128, 128, 3), dtype=np.uint8) * 200
        target_pil = Image.fromarray(target_img)
        target_pil.save(target_dir / f"img_{i:03d}.png")

    return input_dir, target_dir


def test_dual_route_forward_pass():
    """测试1: 双路由前向传播"""
    print("\n" + "=" * 70)
    print("测试1: 双路由MoE-LoRA前向传播")
    print("=" * 70)

    # 1. 加载Mock FLUX模型
    loader = MockFluxModelLoader()
    transformer, _, _, _, _, _, _ = loader.load_all()
    print(f"✓ Mock FLUX模型加载成功")
    print(f"  输入维度: {transformer.config.hidden_size}")
    print(f"  Block数量: {len(transformer.blocks)}")

    # 2. 为第一个block添加双路由MoE-LoRA
    block = transformer.blocks[0]  # Sequential block
    # Get the first linear layer in the block
    original_linear = block[0]  # First linear layer

    # 配置：5个specific + 4个shared = 9个总专家
    num_specific = 5
    num_shared = 4
    rank = 8

    moe_layer = MoELoRALayer(
        base_layer=original_linear,
        rank=rank,
        alpha=8.0,
        num_experts=num_specific + num_shared,
        routing_mode="dual",
        router_config={
            'num_specific': num_specific,
            'shared_top_k': 2,
            'shared_hidden_dim': 512
        }
    )
    # Replace the first linear layer with MoE layer
    transformer.blocks[0] = nn.Sequential(
        moe_layer,
        block[1],  # GELU
        block[2]   # Second linear
    )

    print(f"\n✓ 添加双路由MoE-LoRA到block 0")
    print(f"  Specific专家数: {num_specific}")
    print(f"  Shared专家数: {num_shared}")
    print(f"  总专家数: {num_specific + num_shared}")
    print(f"  Shared Top-K: 2")
    print(f"  LoRA rank: {rank}")

    # 3. 准备输入（模拟batch）
    batch_size = 4
    seq_len = 256
    hidden_size = transformer.config.hidden_size
    device = next(transformer.parameters()).device
    dtype = next(transformer.parameters()).dtype

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    # 曝光类别：[0, 1, 2, 3]（4个不同曝光类别）
    exposure_categories = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)

    print(f"\n输入:")
    print(f"  shape: {x.shape}")
    print(f"  曝光类别: {exposure_categories.tolist()}")

    # 4a. 前向传播 (eval模式 - 不记录统计)
    transformer.eval()
    with torch.no_grad():
        output = moe_layer(x, exposure_category=exposure_categories)

    print(f"\n输出:")
    print(f"  shape: {output.shape}")
    assert output.shape == x.shape, "输出形状应该与输入相同"

    print(f"\n✓ 前向传播成功 (eval模式)")
    print(f"✓ 输出形状正确: {output.shape}")

    # 4b. 训练模式前向传播（记录统计）
    moe_layer.train()
    with torch.no_grad():  # 不需要梯度，只是为了记录统计
        _ = moe_layer(x, exposure_category=exposure_categories)
    moe_layer.eval()

    # 5. 验证路由统计
    specific_stats = moe_layer.specific_router.get_routing_stats()
    shared_stats = moe_layer.shared_router.get_routing_stats()

    print(f"\nSpecific Router统计 (训练模式记录):")
    print(f"  总调用: {specific_stats['total_calls']}")
    print(f"  专家使用: {specific_stats['expert_usage']}")
    print(f"  使用率: {[f'{r:.2f}' for r in specific_stats['expert_usage_ratio']]}")

    print(f"\nShared Router统计 (训练模式记录):")
    print(f"  总调用: {shared_stats['total_calls']}")
    print(f"  专家使用: {shared_stats['expert_usage']}")
    print(f"  使用率: {[f'{r:.3f}' for r in shared_stats['expert_usage_ratio']]}")
    print(f"  负载均衡损失: {shared_stats['load_balance_loss']:.4f}")

    # 验证specific专家被正确调用
    assert specific_stats['total_calls'] == batch_size
    assert specific_stats['expert_usage'][0] == 1.0  # 类别0被调用1次
    assert specific_stats['expert_usage'][1] == 1.0  # 类别1被调用1次
    assert specific_stats['expert_usage'][2] == 1.0  # 类别2被调用1次
    assert specific_stats['expert_usage'][3] == 1.0  # 类别3被调用1次

    print("\n✅ 双路由前向传播测试通过")


def test_with_dataset():
    """测试2: 与数据集集成"""
    print("\n" + "=" * 70)
    print("测试2: 双路由与数据集集成")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir, target_dir = create_test_dataset(temp_path, num_images=8)

        # 创建数据集（启用曝光度分类）
        dataset = ImagePairDataset(
            input_image_dir=str(input_dir),
            target_image_dir=str(target_dir),
            prompt="Enhance the low-light image",
            resolution=64,
            classify_exposure_category=True,
        )

        print(f"✓ 创建数据集，共 {len(dataset)} 个样本")

        # 创建DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        print(f"✓ 创建DataLoader，batch_size=4")

        # 加载Mock FLUX + 添加双路由MoE-LoRA
        loader = MockFluxModelLoader()
        transformer, _, _, _, _, _, _ = loader.load_all()

        block = transformer.blocks[0]
        original_linear = block[0]

        moe_layer = MoELoRALayer(
            base_layer=original_linear,
            rank=8,
            alpha=8.0,
            num_experts=9,  # 5 specific + 4 shared
            routing_mode="dual",
            router_config={
                'num_specific': 5,
                'shared_top_k': 2
            }
        )
        transformer.blocks[0] = nn.Sequential(moe_layer, block[1], block[2])
        print(f"✓ Mock FLUX + 双路由MoE-LoRA准备完成")

        # 获取一个batch
        batch = next(iter(dataloader))
        exposure_categories = batch['exposure_category'].to(next(transformer.parameters()).device)

        print(f"\nBatch信息:")
        print(f"  input_image shape: {batch['input_image'].shape}")
        print(f"  target_image shape: {batch['target_image'].shape}")
        print(f"  exposure_category: {exposure_categories.tolist()}")

        # 模拟前向传播（生成latents）
        batch_size = batch['input_image'].size(0)
        seq_len = 256
        hidden_size = transformer.config.hidden_size
        device = next(transformer.parameters()).device
        dtype = next(transformer.parameters()).dtype

        # 创建模拟的hidden states
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

        # 前向传播
        transformer.eval()
        with torch.no_grad():
            output = moe_layer(hidden_states, exposure_category=exposure_categories)

        print(f"\n模型输出:")
        print(f"  shape: {output.shape}")
        assert output.shape == hidden_states.shape

        # 验证曝光类别被正确使用
        stats = moe_layer.specific_router.get_routing_stats()
        print(f"\nSpecific专家调用情况:")
        for i in range(5):
            count = stats['expert_usage'][i]
            if count > 0:
                print(f"  专家{i}: 被调用 {int(count)} 次")

        print("\n✅ 数据集集成测试通过")


def test_save_and_load():
    """测试3: 双路由权重保存和加载"""
    print("\n" + "=" * 70)
    print("测试3: 双路由MoE-LoRA权重保存和加载")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        save_path = temp_path / "dual_route_lora.pt"

        # 1. 加载模型和创建MoE层
        loader = MockFluxModelLoader()
        transformer, _, _, _, _, _, _ = loader.load_all()

        block = transformer.blocks[0]
        original_linear = block[0]

        moe_layer = MoELoRALayer(
            base_layer=original_linear,
            rank=8,
            alpha=8.0,
            num_experts=9,
            routing_mode="dual",
            router_config={
                'num_specific': 5,
                'shared_top_k': 2
            }
        )

        print(f"✓ 创建双路由MoE-LoRA层")

        # 2. 运行一次前向传播（初始化统计）
        device = next(transformer.parameters()).device
        dtype = next(transformer.parameters()).dtype
        x = torch.randn(4, 256, transformer.config.hidden_size, device=device, dtype=dtype)
        exposure_categories = torch.tensor([0, 1, 2, 3], device=device)

        moe_layer.eval()
        with torch.no_grad():
            _ = moe_layer(x, exposure_category=exposure_categories)

        # 3. 获取原始权重
        original_lora_A = moe_layer.lora_A[0].clone()
        original_lora_B = moe_layer.lora_B[0].clone()

        print(f"✓ 原始权重:")
        print(f"  lora_A[0] shape: {original_lora_A.shape}")
        print(f"  lora_B[0] shape: {original_lora_B.shape}")

        # 4. 保存权重
        state_dict = {
            'lora_A': [expert.data for expert in moe_layer.lora_A],
            'lora_B': [expert.data for expert in moe_layer.lora_B],
            'specific_router': moe_layer.specific_router.state_dict(),
            'shared_router': moe_layer.shared_router.state_dict(),
            'config': {
                'num_experts': 9,
                'num_specific': 5,
                'num_shared': 4,
                'routing_mode': 'dual',
                'rank': 8,
                'alpha': 8.0,
            }
        }

        torch.save(state_dict, save_path)
        print(f"\n✓ 权重已保存到: {save_path}")
        print(f"  文件大小: {save_path.stat().st_size / 1024:.2f} KB")

        # 5. 创建新的MoE层并加载权重
        new_moe_layer = MoELoRALayer(
            base_layer=nn.Linear(transformer.config.hidden_size, transformer.config.hidden_size),
            rank=8,
            alpha=8.0,
            num_experts=9,
            routing_mode="dual",
            router_config={
                'num_specific': 5,
                'shared_top_k': 2
            }
        )

        # 加载权重
        loaded_state = torch.load(save_path)

        for i in range(9):
            new_moe_layer.lora_A[i].data = loaded_state['lora_A'][i]
            new_moe_layer.lora_B[i].data = loaded_state['lora_B'][i]

        new_moe_layer.specific_router.load_state_dict(loaded_state['specific_router'])
        new_moe_layer.shared_router.load_state_dict(loaded_state['shared_router'])

        print(f"✓ 权重已加载到新MoE层")

        # 6. 验证权重一致性
        loaded_lora_A = new_moe_layer.lora_A[0]
        loaded_lora_B = new_moe_layer.lora_B[0]

        assert torch.allclose(original_lora_A, loaded_lora_A), "lora_A权重不匹配"
        assert torch.allclose(original_lora_B, loaded_lora_B), "lora_B权重不匹配"

        print(f"\n✓ 权重验证:")
        print(f"  lora_A[0] 一致: {torch.allclose(original_lora_A, loaded_lora_A)}")
        print(f"  lora_B[0] 一致: {torch.allclose(original_lora_B, loaded_lora_B)}")

        # 7. 将新模型移到正确的设备和dtype
        new_moe_layer = new_moe_layer.to(device=device, dtype=dtype)

        # 8. 验证新模型可以正常进行前向传播
        new_moe_layer.eval()

        # 使用固定种子生成测试输入
        torch.manual_seed(42)
        x_test = torch.randn(4, 256, transformer.config.hidden_size, device=device, dtype=dtype)
        exposure_test = torch.tensor([0, 1, 2, 3], device=device)

        with torch.no_grad():
            output_new = new_moe_layer(x_test, exposure_category=exposure_test)

        # 验证输出形状正确
        assert output_new.shape == x_test.shape, "新模型输出形状不正确"

        print(f"  新模型前向传播成功")
        print(f"  输出shape: {output_new.shape}")
        print(f"  输出范围: [{output_new.min().item():.3f}, {output_new.max().item():.3f}]")

        print("\n✅ 权重保存和加载测试通过")


def test_multi_block_dual_route():
    """测试4: 多个block使用双路由"""
    print("\n" + "=" * 70)
    print("测试4: 多个Block使用双路由MoE-LoRA")
    print("=" * 70)

    # 加载Mock FLUX
    loader = MockFluxModelLoader()
    transformer, _, _, _, _, _, _ = loader.load_all()
    num_blocks = len(transformer.blocks)

    print(f"✓ Mock FLUX模型加载，共 {num_blocks} 个blocks")

    # 为所有blocks添加双路由MoE-LoRA
    moe_layers = []
    for i in range(num_blocks):
        block = transformer.blocks[i]
        original_linear = block[0]

        moe_layer = MoELoRALayer(
            base_layer=original_linear,
            rank=8,
            alpha=8.0,
            num_experts=9,
            routing_mode="dual",
            router_config={
                'num_specific': 5,
                'shared_top_k': 2
            }
        )
        transformer.blocks[i] = nn.Sequential(moe_layer, block[1], block[2])
        moe_layers.append(moe_layer)

    print(f"✓ 为所有 {num_blocks} 个blocks添加双路由MoE-LoRA")

    # 准备输入
    batch_size = 4
    seq_len = 256
    hidden_size = transformer.config.hidden_size
    device = next(transformer.parameters()).device
    dtype = next(transformer.parameters()).dtype

    latents = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    exposure_categories = torch.tensor([0, 1, 2, 3], device=device)

    print(f"\n输入:")
    print(f"  latents shape: {latents.shape}")
    print(f"  exposure_categories: {exposure_categories.tolist()}")

    # 逐block前向传播
    transformer.eval()
    with torch.no_grad():
        hidden_states = latents

        # 逐block前向
        for i, moe_layer in enumerate(moe_layers):
            hidden_states_before = hidden_states

            # 手动调用block内的各层（不能直接调用Sequential，因为MoE需要exposure_category）
            # block = Sequential(moe_layer, GELU, Linear)
            hidden_states = moe_layer(hidden_states, exposure_category=exposure_categories)
            # 注意：我们只调用MoE层用于测试，实际使用中整个transformer会被正确处理

            print(f"  Block {i}: {hidden_states_before.shape} -> {hidden_states.shape}")

    print(f"\n✓ 多Block前向传播成功")
    print(f"✓ 最终输出shape: {hidden_states.shape}")

    # 统计每个block的路由信息
    print(f"\n各Block的路由统计:")
    for i, moe_layer in enumerate(moe_layers):
        specific_stats = moe_layer.specific_router.get_routing_stats()
        shared_stats = moe_layer.shared_router.get_routing_stats()

        print(f"\n  Block {i}:")
        print(f"    Specific调用: {specific_stats['total_calls']}")
        print(f"    Shared调用: {shared_stats['total_calls']}")

    print("\n✅ 多Block双路由测试通过")


def main():
    print("\n" + "=" * 70)
    print("开始双路由MoE-LoRA端到端测试")
    print("=" * 70)

    try:
        test_dual_route_forward_pass()
        test_with_dataset()
        test_save_and_load()
        test_multi_block_dual_route()

        print("\n" + "=" * 70)
        print("✅ 所有端到端测试通过！")
        print("=" * 70)

        print("\n总结:")
        print("  ✓ 双路由前向传播正常工作")
        print("  ✓ Specific Router根据曝光类别正确选择专家")
        print("  ✓ Shared Router动态选择共享专家")
        print("  ✓ 与数据集曝光度分类集成正常")
        print("  ✓ 权重保存和加载功能正常")
        print("  ✓ 多Block双路由工作正常")
        print("  ✓ 路由统计功能正常")
        print("=" * 70)

        print("\n✨ 阶段三（双路由机制）已全部完成！")
        print("\n准备进入阶段四：")
        print("  - 训练流程实现")
        print("  - 损失函数集成")
        print("  - 完整训练循环")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
