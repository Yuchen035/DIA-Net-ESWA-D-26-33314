"""
配置系统手动测试脚本（不依赖pytest）

运行: python test/test_config_manual.py
"""

import sys
import tempfile
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import TrainingConfig, LoRAConfig, ModelConfig, DataConfig


def test_lora_config():
    """测试LoRA配置"""
    print("测试1: LoRA配置默认值...")
    config = LoRAConfig()
    assert config.rank == 16, f"Expected rank=16, got {config.rank}"
    assert config.alpha == 16.0, f"Expected alpha=16.0, got {config.alpha}"
    print("✓ LoRA配置默认值正确")

    print("测试2: LoRA配置从字典加载...")
    config = LoRAConfig.from_dict({'rank': 32, 'alpha': 32.0})
    assert config.rank == 32, f"Expected rank=32, got {config.rank}"
    assert config.alpha == 32.0, f"Expected alpha=32.0, got {config.alpha}"
    print("✓ LoRA配置从字典加载正确")


def test_model_config():
    """测试模型配置"""
    print("\n测试3: 模型配置默认值...")
    config = ModelConfig()
    assert config.base_model == "black-forest-labs/FLUX.1-dev"
    assert config.dtype == "bfloat16"
    print("✓ 模型配置默认值正确")


def test_data_config():
    """测试数据配置"""
    print("\n测试4: 数据配置缺少必需参数...")
    try:
        DataConfig.from_dict({'input_image_dir': '/tmp/input'})
        assert False, "应该抛出ValueError"
    except ValueError as e:
        assert "target_image_dir" in str(e)
        print(f"✓ 正确捕获缺少必需参数: {e}")

    print("测试5: 数据配置正常加载...")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = DataConfig.from_dict({
            'input_image_dir': tmpdir,
            'target_image_dir': tmpdir,
        })
        assert config.input_image_dir == tmpdir
        assert config.resolution == 512  # 默认值
        print("✓ 数据配置正常加载")


def test_training_config():
    """测试完整训练配置"""
    print("\n测试6: 完整配置从字典加载...")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        target_dir = Path(tmpdir) / "target"
        input_dir.mkdir()
        target_dir.mkdir()

        config_dict = {
            'job_name': 'test_job',
            'device': 'cuda',
            'seed': 123,
            'model': {
                'base_model': 'test/model',
                'dtype': 'float32',
            },
            'lora': {
                'rank': 32,
                'alpha': 32.0,
            },
            'data': {
                'input_image_dir': str(input_dir),
                'target_image_dir': str(target_dir),
                'resolution': 1024,
            },
            'train': {
                'num_epochs': 50,
                'learning_rate': 5e-5,
            },
            'save': {
                'output_dir': './test_output',
            },
        }

        config = TrainingConfig.from_dict(config_dict)
        assert config.job_name == 'test_job'
        assert config.lora.rank == 32
        assert config.train.num_epochs == 50
        assert config.data.resolution == 1024
        print("✓ 完整配置从字典加载正确")


def test_config_validation():
    """测试配置验证"""
    print("\n测试7: 配置验证 - 无效设备...")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        target_dir = Path(tmpdir) / "target"
        input_dir.mkdir()
        target_dir.mkdir()

        config_dict = {
            'device': 'invalid_device',
            'data': {
                'input_image_dir': str(input_dir),
                'target_image_dir': str(target_dir),
            },
        }

        config = TrainingConfig.from_dict(config_dict)
        try:
            config.validate()
            assert False, "应该抛出ValueError"
        except ValueError as e:
            assert "Invalid device" in str(e)
            print(f"✓ 正确捕获无效设备: {e}")

    print("测试8: 配置验证 - 负数rank...")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        target_dir = Path(tmpdir) / "target"
        input_dir.mkdir()
        target_dir.mkdir()

        config_dict = {
            'lora': {'rank': -1},
            'data': {
                'input_image_dir': str(input_dir),
                'target_image_dir': str(target_dir),
            },
        }

        config = TrainingConfig.from_dict(config_dict)
        try:
            config.validate()
            assert False, "应该抛出ValueError"
        except ValueError as e:
            assert "rank must be positive" in str(e)
            print(f"✓ 正确捕获负数rank: {e}")


def test_yaml_operations():
    """测试YAML操作"""
    print("\n测试9: 配置保存到YAML...")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        target_dir = Path(tmpdir) / "target"
        input_dir.mkdir()
        target_dir.mkdir()

        config_dict = {
            'job_name': 'test_yaml',
            'data': {
                'input_image_dir': str(input_dir),
                'target_image_dir': str(target_dir),
            },
        }

        config = TrainingConfig.from_dict(config_dict)
        yaml_path = Path(tmpdir) / "config.yaml"
        config.save_yaml(yaml_path)

        assert yaml_path.exists(), "YAML文件应该被创建"
        print("✓ 配置成功保存到YAML")

        print("测试10: 从YAML加载配置...")
        loaded_config = TrainingConfig.from_yaml(yaml_path)
        assert loaded_config.job_name == 'test_yaml'
        assert loaded_config.lora.rank == 16  # 默认值
        print("✓ 从YAML成功加载配置")


def test_to_dict():
    """测试配置转为字典"""
    print("\n测试11: 配置转为字典...")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        target_dir = Path(tmpdir) / "target"
        input_dir.mkdir()
        target_dir.mkdir()

        config_dict = {
            'job_name': 'test_to_dict',
            'lora': {'rank': 64},
            'data': {
                'input_image_dir': str(input_dir),
                'target_image_dir': str(target_dir),
            },
        }

        config = TrainingConfig.from_dict(config_dict)
        output_dict = config.to_dict()

        assert output_dict['job_name'] == 'test_to_dict'
        assert output_dict['lora']['rank'] == 64
        print("✓ 配置转为字典正确")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("开始配置系统手动测试")
    print("=" * 60)

    try:
        test_lora_config()
        test_model_config()
        test_data_config()
        test_training_config()
        test_config_validation()
        test_yaml_operations()
        test_to_dict()

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
