"""
Flow Matching Loss测试

测试内容：
1. Loss本身功能
   - 初始化
   - velocity预测模式
   - noise预测模式
   - reduction参数
   - 错误处理

2. 数值正确性
   - 手动计算验证
   - 梯度检查
   - 不同形状输入

3. 便捷函数
   - compute_flow_matching_loss功能
"""

import sys
import os
import torch

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.loss.flow_matching import FlowMatchingLoss, compute_flow_matching_loss


# ============================================================================
# 第一部分：Loss本身功能测试
# ============================================================================

def test_loss_initialization():
    """测试1: Loss初始化"""
    # 默认初始化
    loss_fn = FlowMatchingLoss()
    assert loss_fn.prediction_type == "velocity", "默认应该是velocity模式"
    assert loss_fn.reduction == "mean", "默认reduction应该是mean"

    # 自定义初始化
    loss_fn_noise = FlowMatchingLoss(prediction_type="noise", reduction="sum")
    assert loss_fn_noise.prediction_type == "noise"
    assert loss_fn_noise.reduction == "sum"

    print("✓ 测试1: Loss初始化")


def test_loss_invalid_prediction_type():
    """测试2: 无效的prediction_type"""
    try:
        loss_fn = FlowMatchingLoss(prediction_type="invalid")
        assert False, "应该抛出ValueError"
    except ValueError as e:
        assert "prediction_type must be" in str(e)

    print("✓ 测试2: 无效的prediction_type")


def test_velocity_mode():
    """测试3: velocity预测模式"""
    loss_fn = FlowMatchingLoss(prediction_type="velocity")

    # 创建测试数据
    batch_size = 2
    channels = 4
    height = width = 32

    model_output = torch.randn(batch_size, channels, height, width)
    target = torch.randn(batch_size, channels, height, width)
    noise = torch.randn(batch_size, channels, height, width)

    # 计算loss
    loss = loss_fn(model_output, target, noise)

    # 检查loss是标量
    assert loss.dim() == 0, "loss应该是标量"
    assert loss.item() >= 0, "MSE loss应该非负"

    print("✓ 测试3: velocity预测模式")


def test_velocity_mode_missing_noise():
    """测试4: velocity模式缺少noise参数"""
    loss_fn = FlowMatchingLoss(prediction_type="velocity")

    model_output = torch.randn(2, 4, 32, 32)
    target = torch.randn(2, 4, 32, 32)

    try:
        loss = loss_fn(model_output, target, noise=None)
        assert False, "应该抛出ValueError"
    except ValueError as e:
        assert "noise is required" in str(e)

    print("✓ 测试4: velocity模式缺少noise参数")


def test_noise_mode():
    """测试5: noise预测模式"""
    loss_fn = FlowMatchingLoss(prediction_type="noise")

    # 创建测试数据
    model_output = torch.randn(2, 4, 32, 32)
    target = torch.randn(2, 4, 32, 32)
    noise = torch.randn(2, 4, 32, 32)

    # 计算loss
    loss = loss_fn(model_output, target, noise)

    # 检查loss是标量
    assert loss.dim() == 0, "loss应该是标量"
    assert loss.item() >= 0, "MSE loss应该非负"

    print("✓ 测试5: noise预测模式")


def test_reduction_mean():
    """测试6: reduction='mean'"""
    loss_fn = FlowMatchingLoss(reduction="mean")

    model_output = torch.randn(2, 4, 32, 32)
    target = torch.randn(2, 4, 32, 32)
    noise = torch.randn(2, 4, 32, 32)

    loss = loss_fn(model_output, target, noise)

    # mean reduction返回标量
    assert loss.dim() == 0, "mean reduction应该返回标量"

    print("✓ 测试6: reduction='mean'")


def test_reduction_sum():
    """测试7: reduction='sum'"""
    loss_fn = FlowMatchingLoss(reduction="sum")

    model_output = torch.randn(2, 4, 32, 32)
    target = torch.randn(2, 4, 32, 32)
    noise = torch.randn(2, 4, 32, 32)

    loss = loss_fn(model_output, target, noise)

    # sum reduction返回标量
    assert loss.dim() == 0, "sum reduction应该返回标量"

    # sum应该比mean大
    loss_fn_mean = FlowMatchingLoss(reduction="mean")
    loss_mean = loss_fn_mean(model_output, target, noise)
    assert loss.item() > loss_mean.item(), "sum应该比mean大"

    print("✓ 测试7: reduction='sum'")


def test_reduction_none():
    """测试8: reduction='none'"""
    loss_fn = FlowMatchingLoss(reduction="none")

    batch_size = 2
    channels = 4
    height = width = 32

    model_output = torch.randn(batch_size, channels, height, width)
    target = torch.randn(batch_size, channels, height, width)
    noise = torch.randn(batch_size, channels, height, width)

    loss = loss_fn(model_output, target, noise)

    # none reduction返回与输入相同形状
    assert loss.shape == (batch_size, channels, height, width), "none reduction应该保持形状"

    print("✓ 测试8: reduction='none'")


# ============================================================================
# 第二部分：数值正确性测试
# ============================================================================

def test_velocity_correctness():
    """测试9: velocity模式数值正确性"""
    loss_fn = FlowMatchingLoss(prediction_type="velocity", reduction="mean")

    # 创建简单的测试数据
    model_output = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # [1, 1, 2, 2]
    target = torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]])
    noise = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])

    # 手动计算
    # target_velocity = target - noise = [[[[4.0, 5.0], [6.0, 7.0]]]]
    # diff = model_output - target_velocity = [[[[1-4, 2-5], [3-6, 4-7]]]]
    #                                        = [[[[-3, -3], [-3, -3]]]]
    # squared_diff = [[[[9, 9], [9, 9]]]]
    # mean = 36 / 4 = 9.0

    loss = loss_fn(model_output, target, noise)

    expected_loss = 9.0
    assert abs(loss.item() - expected_loss) < 1e-5, f"loss应该约为{expected_loss}，实际为{loss.item()}"

    print("✓ 测试9: velocity模式数值正确性")


def test_noise_correctness():
    """测试10: noise模式数值正确性"""
    loss_fn = FlowMatchingLoss(prediction_type="noise", reduction="mean")

    # 创建简单的测试数据
    model_output = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]])  # 在noise模式下target不影响loss
    noise = torch.tensor([[[[2.0, 3.0], [4.0, 5.0]]]])

    # 手动计算
    # diff = model_output - noise = [[[[1-2, 2-3], [3-4, 4-5]]]]
    #                              = [[[[-1, -1], [-1, -1]]]]
    # squared_diff = [[[[1, 1], [1, 1]]]]
    # mean = 4 / 4 = 1.0

    loss = loss_fn(model_output, target, noise)

    expected_loss = 1.0
    assert abs(loss.item() - expected_loss) < 1e-5, f"loss应该约为{expected_loss}，实际为{loss.item()}"

    print("✓ 测试10: noise模式数值正确性")


def test_gradient_flow():
    """测试11: 梯度流动"""
    loss_fn = FlowMatchingLoss(prediction_type="velocity")

    # 创建需要梯度的张量
    model_output = torch.randn(2, 4, 32, 32, requires_grad=True)
    target = torch.randn(2, 4, 32, 32)
    noise = torch.randn(2, 4, 32, 32)

    # 计算loss并反向传播
    loss = loss_fn(model_output, target, noise)
    loss.backward()

    # 检查梯度
    assert model_output.grad is not None, "model_output应该有梯度"
    assert model_output.grad.shape == model_output.shape, "梯度形状应该与输入相同"
    assert not torch.isnan(model_output.grad).any(), "梯度不应该包含NaN"

    print("✓ 测试11: 梯度流动")


def test_different_shapes():
    """测试12: 不同形状的输入"""
    loss_fn = FlowMatchingLoss()

    # 测试不同的batch size
    for batch_size in [1, 2, 4]:
        model_output = torch.randn(batch_size, 4, 32, 32)
        target = torch.randn(batch_size, 4, 32, 32)
        noise = torch.randn(batch_size, 4, 32, 32)

        loss = loss_fn(model_output, target, noise)
        assert loss.dim() == 0, f"batch_size={batch_size}时loss应该是标量"

    # 测试不同的分辨率
    for resolution in [16, 32, 64]:
        model_output = torch.randn(2, 4, resolution, resolution)
        target = torch.randn(2, 4, resolution, resolution)
        noise = torch.randn(2, 4, resolution, resolution)

        loss = loss_fn(model_output, target, noise)
        assert loss.dim() == 0, f"resolution={resolution}时loss应该是标量"

    # 测试不同的通道数
    for channels in [3, 4, 8]:
        model_output = torch.randn(2, channels, 32, 32)
        target = torch.randn(2, channels, 32, 32)
        noise = torch.randn(2, channels, 32, 32)

        loss = loss_fn(model_output, target, noise)
        assert loss.dim() == 0, f"channels={channels}时loss应该是标量"

    print("✓ 测试12: 不同形状的输入")


def test_zero_loss():
    """测试13: 完美预测的loss应该为0"""
    loss_fn = FlowMatchingLoss(prediction_type="velocity")

    target = torch.randn(2, 4, 32, 32)
    noise = torch.randn(2, 4, 32, 32)

    # 完美预测：model_output = target_velocity
    target_velocity = target - noise
    model_output = target_velocity.clone()

    loss = loss_fn(model_output, target, noise)

    assert loss.item() < 1e-6, f"完美预测的loss应该接近0，实际为{loss.item()}"

    print("✓ 测试13: 完美预测的loss应该为0")


# ============================================================================
# 第三部分：便捷函数测试
# ============================================================================

def test_compute_function():
    """测试14: compute_flow_matching_loss便捷函数"""
    model_output = torch.randn(2, 4, 32, 32)
    target = torch.randn(2, 4, 32, 32)
    noise = torch.randn(2, 4, 32, 32)

    # 使用便捷函数
    loss = compute_flow_matching_loss(model_output, target, noise)

    # 使用类
    loss_fn = FlowMatchingLoss()
    loss_class = loss_fn(model_output, target, noise)

    # 应该得到相同的结果
    assert abs(loss.item() - loss_class.item()) < 1e-6, "便捷函数和类应该产生相同的结果"

    print("✓ 测试14: compute_flow_matching_loss便捷函数")


def test_compute_function_with_args():
    """测试15: compute_flow_matching_loss带参数"""
    model_output = torch.randn(2, 4, 32, 32)
    target = torch.randn(2, 4, 32, 32)
    noise = torch.randn(2, 4, 32, 32)

    # 测试不同参数
    loss_velocity = compute_flow_matching_loss(
        model_output, target, noise,
        prediction_type="velocity",
        reduction="mean"
    )

    loss_noise = compute_flow_matching_loss(
        model_output, target, noise,
        prediction_type="noise",
        reduction="sum"
    )

    # 两种模式应该产生不同的loss
    assert loss_velocity.item() != loss_noise.item(), "不同参数应该产生不同的loss"

    print("✓ 测试15: compute_flow_matching_loss带参数")


def test_repr():
    """测试16: __repr__方法"""
    loss_fn = FlowMatchingLoss(prediction_type="velocity", reduction="mean")

    repr_str = repr(loss_fn)

    assert "FlowMatchingLoss" in repr_str
    assert "velocity" in repr_str
    assert "mean" in repr_str

    print("✓ 测试16: __repr__方法")


# ============================================================================
# 运行所有测试
# ============================================================================

def main():
    """运行所有测试"""
    print("=" * 60)
    print("开始Flow Matching Loss测试")
    print("=" * 60)

    # 第一部分：Loss本身功能测试
    print("\n【第一部分：Loss本身功能测试】")
    test_loss_initialization()
    test_loss_invalid_prediction_type()
    test_velocity_mode()
    test_velocity_mode_missing_noise()
    test_noise_mode()
    test_reduction_mean()
    test_reduction_sum()
    test_reduction_none()

    # 第二部分：数值正确性测试
    print("\n【第二部分：数值正确性测试】")
    test_velocity_correctness()
    test_noise_correctness()
    test_gradient_flow()
    test_different_shapes()
    test_zero_loss()

    # 第三部分：便捷函数测试
    print("\n【第三部分：便捷函数测试】")
    test_compute_function()
    test_compute_function_with_args()
    test_repr()

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    main()
