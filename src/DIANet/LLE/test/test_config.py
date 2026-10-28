"""
配置系统测试

测试覆盖：
1. 正常配置加载（YAML、字典）
2. 必需参数缺失
3. 配置验证（无效值）
4. 默认值处理
5. 配置保存
6. 边界情况
"""

import pytest
import yaml
import tempfile
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import (
    TrainingConfig,
    LoRAConfig,
    ModelConfig,
    DataConfig,
    TrainConfig,
    SaveConfig,
    ConfigDict,
)


class TestConfigDict:
    """测试ConfigDict类"""

    def test_dict_access(self):
        """测试字典访问"""
        config = ConfigDict({'key': 'value'})
        assert config['key'] == 'value'

    def test_dot_access(self):
        """测试点号访问"""
        config = ConfigDict({'key': 'value'})
        assert config.key == 'value'

    def test_nested_access(self):
        """测试嵌套访问"""
        config = ConfigDict({'lora': {'rank': 16}})
        assert config.lora.rank == 16

    def test_missing_key(self):
        """测试访问不存在的键"""
        config = ConfigDict({'key': 'value'})
        with pytest.raises(AttributeError):
            _ = config.missing_key

    def test_get_with_default(self):
        """测试get方法的默认值"""
        config = ConfigDict({'key': 'value'})
        assert config.get('key') == 'value'
        assert config.get('missing', 'default') == 'default'


class TestLoRAConfig:
    """测试LoRAConfig类"""

    def test_default_values(self):
        """测试默认值"""
        config = LoRAConfig()
        assert config.rank == 16
        assert config.alpha == 16.0
        assert config.dropout == 0.0
        assert len(config.target_modules) > 0

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'rank': 32,
            'alpha': 32.0,
            'dropout': 0.1,
            'target_modules': ['module1', 'module2'],
        }
        config = LoRAConfig.from_dict(data)
        assert config.rank == 32
        assert config.alpha == 32.0
        assert config.dropout == 0.1
        assert config.target_modules == ['module1', 'module2']

    def test_from_dict_partial(self):
        """测试从部分字典创建（使用默认值）"""
        data = {'rank': 64}
        config = LoRAConfig.from_dict(data)
        assert config.rank == 64
        assert config.alpha == 16.0  # 默认值


class TestModelConfig:
    """测试ModelConfig类"""

    def test_default_values(self):
        """测试默认值"""
        config = ModelConfig()
        assert config.base_model == "black-forest-labs/FLUX.1-dev"
        assert config.dtype == "bfloat16"

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'base_model': 'custom/model',
            'dtype': 'float32',
        }
        config = ModelConfig.from_dict(data)
        assert config.base_model == 'custom/model'
        assert config.dtype == 'float32'


class TestDataConfig:
    """测试DataConfig类"""

    def test_from_dict_valid(self, tmp_path):
        """测试从有效字典创建"""
        data = {
            'input_image_dir': str(tmp_path / 'input'),
            'target_image_dir': str(tmp_path / 'target'),
            'prompt': 'test prompt',
            'resolution': 1024,
        }
        config = DataConfig.from_dict(data)
        assert config.input_image_dir == str(tmp_path / 'input')
        assert config.target_image_dir == str(tmp_path / 'target')
        assert config.prompt == 'test prompt'
        assert config.resolution == 1024

    def test_missing_required_field(self):
        """测试缺少必需字段"""
        # 缺少 input_image_dir
        data = {'target_image_dir': '/path/to/target'}
        with pytest.raises(ValueError, match="Missing required config: data.input_image_dir"):
            DataConfig.from_dict(data)

        # 缺少 target_image_dir
        data = {'input_image_dir': '/path/to/input'}
        with pytest.raises(ValueError, match="Missing required config: data.target_image_dir"):
            DataConfig.from_dict(data)

    def test_default_values(self, tmp_path):
        """测试默认值"""
        data = {
            'input_image_dir': str(tmp_path / 'input'),
            'target_image_dir': str(tmp_path / 'target'),
        }
        config = DataConfig.from_dict(data)
        assert config.prompt == "enhance the low-light image"
        assert config.resolution == 512
        assert config.batch_size == 1
        assert config.num_workers == 4


class TestTrainConfig:
    """测试TrainConfig类"""

    def test_default_values(self):
        """测试默认值"""
        config = TrainConfig()
        assert config.num_epochs == 100
        assert config.learning_rate == 1e-4
        assert config.lr_scheduler == "constant"
        assert config.gradient_accumulation_steps == 1
        assert config.max_grad_norm == 1.0

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'num_epochs': 50,
            'learning_rate': 5e-5,
            'lr_scheduler': 'cosine',
        }
        config = TrainConfig.from_dict(data)
        assert config.num_epochs == 50
        assert config.learning_rate == 5e-5
        assert config.lr_scheduler == 'cosine'


class TestSaveConfig:
    """测试SaveConfig类"""

    def test_default_values(self):
        """测试默认值"""
        config = SaveConfig()
        assert config.output_dir == "./outputs/lora_training"
        assert config.save_every_n_epochs == 10
        assert config.save_dtype == "float16"

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'output_dir': './custom/output',
            'save_every_n_epochs': 5,
            'save_dtype': 'float32',
        }
        config = SaveConfig.from_dict(data)
        assert config.output_dir == './custom/output'
        assert config.save_every_n_epochs == 5
        assert config.save_dtype == 'float32'


class TestTrainingConfig:
    """测试TrainingConfig类"""

    @pytest.fixture
    def valid_config_dict(self, tmp_path):
        """创建有效的配置字典"""
        input_dir = tmp_path / "input"
        target_dir = tmp_path / "target"
        input_dir.mkdir()
        target_dir.mkdir()

        return {
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

    def test_from_dict_valid(self, valid_config_dict):
        """测试从有效字典创建配置"""
        config = TrainingConfig.from_dict(valid_config_dict)
        assert config.job_name == 'test_job'
        assert config.device == 'cuda'
        assert config.seed == 123
        assert config.model.base_model == 'test/model'
        assert config.lora.rank == 32
        assert config.train.num_epochs == 50

    def test_missing_data_section(self):
        """测试缺少data配置段"""
        data = {
            'job_name': 'test',
            'model': {},
            'lora': {},
            'train': {},
            'save': {},
        }
        with pytest.raises(ValueError, match="Missing required config: data.input_image_dir"):
            TrainingConfig.from_dict(data)

    def test_from_yaml_valid(self, tmp_path, valid_config_dict):
        """测试从YAML文件加载配置"""
        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(valid_config_dict, f)

        config = TrainingConfig.from_yaml(yaml_path)
        assert config.job_name == 'test_job'
        assert config.lora.rank == 32

    def test_from_yaml_not_found(self):
        """测试加载不存在的YAML文件"""
        with pytest.raises(FileNotFoundError):
            TrainingConfig.from_yaml('/nonexistent/config.yaml')

    def test_to_dict(self, valid_config_dict):
        """测试配置转为字典"""
        config = TrainingConfig.from_dict(valid_config_dict)
        config_dict = config.to_dict()

        assert config_dict['job_name'] == 'test_job'
        assert config_dict['lora']['rank'] == 32
        assert config_dict['train']['num_epochs'] == 50

    def test_save_yaml(self, tmp_path, valid_config_dict):
        """测试保存配置到YAML文件"""
        config = TrainingConfig.from_dict(valid_config_dict)
        yaml_path = tmp_path / "output" / "config.yaml"

        config.save_yaml(yaml_path)

        assert yaml_path.exists()

        # 重新加载并验证
        loaded_config = TrainingConfig.from_yaml(yaml_path)
        assert loaded_config.job_name == config.job_name
        assert loaded_config.lora.rank == config.lora.rank

    def test_validate_valid_config(self, tmp_path, valid_config_dict):
        """测试验证有效配置"""
        config = TrainingConfig.from_dict(valid_config_dict)
        config.validate()  # 应该不抛出异常

    def test_validate_invalid_device(self, valid_config_dict):
        """测试验证无效设备"""
        valid_config_dict['device'] = 'invalid_device'
        config = TrainingConfig.from_dict(valid_config_dict)
        with pytest.raises(ValueError, match="Invalid device"):
            config.validate()

    def test_validate_invalid_dtype(self, valid_config_dict):
        """测试验证无效dtype"""
        valid_config_dict['model']['dtype'] = 'invalid_dtype'
        config = TrainingConfig.from_dict(valid_config_dict)
        with pytest.raises(ValueError, match="Invalid dtype"):
            config.validate()

    def test_validate_negative_rank(self, valid_config_dict):
        """测试验证负数rank"""
        valid_config_dict['lora']['rank'] = -1
        config = TrainingConfig.from_dict(valid_config_dict)
        with pytest.raises(ValueError, match="LoRA rank must be positive"):
            config.validate()

    def test_validate_zero_rank(self, valid_config_dict):
        """测试验证零rank"""
        valid_config_dict['lora']['rank'] = 0
        config = TrainingConfig.from_dict(valid_config_dict)
        with pytest.raises(ValueError, match="LoRA rank must be positive"):
            config.validate()

    def test_validate_invalid_dropout(self, valid_config_dict):
        """测试验证无效dropout"""
        # dropout = 1.0 应该失败（必须 < 1）
        valid_config_dict['lora']['dropout'] = 1.0
        config = TrainingConfig.from_dict(valid_config_dict)
        with pytest.raises(ValueError, match="LoRA dropout must be in"):
            config.validate()

        # dropout = -0.1 应该失败（必须 >= 0）
        valid_config_dict['lora']['dropout'] = -0.1
        config = TrainingConfig.from_dict(valid_config_dict)
        with pytest.raises(ValueError, match="LoRA dropout must be in"):
            config.validate()

    def test_validate_negative_epochs(self, valid_config_dict):
        """测试验证负数epochs"""
        valid_config_dict['train']['num_epochs'] = -1
        config = TrainingConfig.from_dict(valid_config_dict)
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            config.validate()

    def test_validate_negative_learning_rate(self, valid_config_dict):
        """测试验证负数学习率"""
        valid_config_dict['train']['learning_rate'] = -0.001
        config = TrainingConfig.from_dict(valid_config_dict)
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            config.validate()

    def test_validate_zero_batch_size(self, valid_config_dict):
        """测试验证零batch_size"""
        valid_config_dict['data']['batch_size'] = 0
        config = TrainingConfig.from_dict(valid_config_dict)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate()

    def test_validate_nonexistent_input_dir(self, valid_config_dict):
        """测试验证不存在的输入目录"""
        valid_config_dict['data']['input_image_dir'] = '/nonexistent/input'
        config = TrainingConfig.from_dict(valid_config_dict)
        with pytest.raises(ValueError, match="Input image directory not found"):
            config.validate()

    def test_validate_nonexistent_target_dir(self, valid_config_dict):
        """测试验证不存在的目标目录"""
        valid_config_dict['data']['target_image_dir'] = '/nonexistent/target'
        config = TrainingConfig.from_dict(valid_config_dict)
        with pytest.raises(ValueError, match="Target image directory not found"):
            config.validate()

    def test_repr(self, valid_config_dict):
        """测试__repr__方法"""
        config = TrainingConfig.from_dict(valid_config_dict)
        repr_str = repr(config)
        assert 'TrainingConfig' in repr_str
        assert 'test_job' in repr_str
        assert '32' in repr_str  # rank

    def test_default_values(self, tmp_path):
        """测试使用默认值的最小配置"""
        input_dir = tmp_path / "input"
        target_dir = tmp_path / "target"
        input_dir.mkdir()
        target_dir.mkdir()

        minimal_config = {
            'data': {
                'input_image_dir': str(input_dir),
                'target_image_dir': str(target_dir),
            }
        }

        config = TrainingConfig.from_dict(minimal_config)
        assert config.job_name == "lora_training"  # 默认值
        assert config.device == "cuda"  # 默认值
        assert config.seed == 42  # 默认值
        assert config.model.dtype == "bfloat16"  # 默认值
        assert config.lora.rank == 16  # 默认值
        assert config.train.num_epochs == 100  # 默认值


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
