"""
配置系统 - 从YAML文件加载训练配置

支持：
- YAML文件加载
- 嵌套配置访问（如 config.lora.rank）
- 配置验证
- 默认值处理
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field


class ConfigDict(dict):
    """
    支持点号访问的配置字典

    Examples:
        config = ConfigDict({'lora': {'rank': 16}})
        print(config.lora.rank)  # 16
    """

    def __init__(self, data: Dict[str, Any]):
        super().__init__()
        for key, value in data.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)
            else:
                self[key] = value

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        self[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持默认值"""
        return super().get(key, default)


@dataclass
class LoRAConfig:
    """LoRA配置"""
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list = field(default_factory=lambda: [
        "single_transformer_blocks.*.attn.to_q",
        "single_transformer_blocks.*.attn.to_k",
        "single_transformer_blocks.*.attn.to_v",
        "single_transformer_blocks.*.attn.to_out.0",
    ])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoRAConfig":
        """从字典创建配置"""
        return cls(
            rank=data.get('rank', 16),
            alpha=data.get('alpha', 16.0),
            dropout=data.get('dropout', 0.0),
            target_modules=data.get('target_modules', cls().target_modules),
        )


@dataclass
class ModelConfig:
    """模型配置"""
    base_model: str = "black-forest-labs/FLUX.1-dev"
    model_path: Optional[str] = None  # 本地模型路径（优先于base_model）
    dtype: str = "bfloat16"
    device: str = "cuda"
    low_cpu_mem_usage: bool = True
    photodoodle: Optional[ConfigDict] = None  # PhotoDoodle配置

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """从字典创建配置"""
        # PhotoDoodle配置转换
        photodoodle = None
        if 'photodoodle' in data and data['photodoodle']:
            photodoodle = ConfigDict(data['photodoodle'])

        return cls(
            base_model=data.get('base_model', 'black-forest-labs/FLUX.1-dev'),
            model_path=data.get('model_path'),
            dtype=data.get('dtype', 'bfloat16'),
            device=data.get('device', 'cuda'),
            low_cpu_mem_usage=data.get('low_cpu_mem_usage', True),
            photodoodle=photodoodle,
        )


@dataclass
class DataConfig:
    """数据配置"""
    input_image_dir: str
    target_image_dir: str
    prompt: str = "enhance the low-light image"
    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = False
    num_workers: int = 4
    pin_memory: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConfig":
        """从字典创建配置"""
        # 必需参数检查
        if 'input_image_dir' not in data:
            raise ValueError("Missing required config: data.input_image_dir")
        if 'target_image_dir' not in data:
            raise ValueError("Missing required config: data.target_image_dir")

        return cls(
            input_image_dir=data['input_image_dir'],
            target_image_dir=data['target_image_dir'],
            prompt=data.get('prompt', 'enhance the low-light image'),
            resolution=data.get('resolution', 512),
            center_crop=data.get('center_crop', False),
            random_flip=data.get('random_flip', False),
            num_workers=data.get('num_workers', 4),
            pin_memory=data.get('pin_memory', False),
        )


@dataclass
class TrainConfig:
    """训练配置"""
    num_epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    max_sequence_length: int = 512  # T5最大序列长度
    save_every_n_epochs: int = 10  # 保存检查点频率
    cfg_scale: float = 3.5  # Classifier-free guidance scale

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        """从字典创建配置"""
        return cls(
            num_epochs=data.get('num_epochs', 100),
            batch_size=data.get('batch_size', 1),
            learning_rate=data.get('learning_rate', 1e-4),
            lr_scheduler=data.get('lr_scheduler', 'constant'),
            gradient_accumulation_steps=data.get('gradient_accumulation_steps', 1),
            max_grad_norm=data.get('max_grad_norm', 1.0),
            max_sequence_length=data.get('max_sequence_length', 512),
            save_every_n_epochs=data.get('save_every_n_epochs', 10),
            cfg_scale=data.get('cfg_scale', 3.5),
        )


@dataclass
class SaveConfig:
    """保存配置"""
    output_dir: str = "./outputs/lora_training"
    save_every_n_epochs: int = 10
    save_dtype: str = "float16"
    save_total_limit: Optional[int] = None  # 最多保留N个检查点

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SaveConfig":
        """从字典创建配置"""
        return cls(
            output_dir=data.get('output_dir', './outputs/lora_training'),
            save_every_n_epochs=data.get('save_every_n_epochs', 10),
            save_dtype=data.get('save_dtype', 'float16'),
            save_total_limit=data.get('save_total_limit'),
        )


class TrainingConfig:
    """
    训练配置类

    从YAML文件加载配置，支持验证和默认值

    Examples:
        # 从YAML文件加载
        config = TrainingConfig.from_yaml('config.yaml')

        # 访问配置
        print(config.lora.rank)
        print(config.train.learning_rate)

        # 从字典加载
        config = TrainingConfig.from_dict({...})
    """

    def __init__(
        self,
        job_name: str = "lora_training",
        device: str = "cuda",
        seed: int = 42,
        model: Optional[ModelConfig] = None,
        lora: Optional[LoRAConfig] = None,
        data: Optional[DataConfig] = None,
        train: Optional[TrainConfig] = None,
        save: Optional[SaveConfig] = None,
    ):
        self.job_name = job_name
        self.device = device
        self.seed = seed
        self.model = model or ModelConfig()
        self.lora = lora or LoRAConfig()
        self.data = data
        self.train = train or TrainConfig()
        self.save = save or SaveConfig()

        # 验证必需的配置
        if self.data is None:
            raise ValueError("Missing required config section: 'data'")

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "TrainingConfig":
        """
        从YAML文件加载配置

        Args:
            yaml_path: YAML配置文件路径

        Returns:
            TrainingConfig实例

        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置验证失败
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """
        从字典加载配置

        Args:
            data: 配置字典

        Returns:
            TrainingConfig实例

        Raises:
            ValueError: 配置验证失败
        """
        # 解析各个子配置
        model_config = ModelConfig.from_dict(data.get('model', {}))
        lora_config = LoRAConfig.from_dict(data.get('lora', {}))
        data_config = DataConfig.from_dict(data.get('data', {}))
        train_config = TrainConfig.from_dict(data.get('train', {}))
        save_config = SaveConfig.from_dict(data.get('save', {}))

        return cls(
            job_name=data.get('job_name', 'lora_training'),
            device=data.get('device', 'cuda'),
            seed=data.get('seed', 42),
            model=model_config,
            lora=lora_config,
            data=data_config,
            train=train_config,
            save=save_config,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典

        Returns:
            配置字典
        """
        return {
            'job_name': self.job_name,
            'device': self.device,
            'seed': self.seed,
            'model': {
                'base_model': self.model.base_model,
                'dtype': self.model.dtype,
            },
            'lora': {
                'rank': self.lora.rank,
                'alpha': self.lora.alpha,
                'dropout': self.lora.dropout,
                'target_modules': self.lora.target_modules,
            },
            'data': {
                'input_image_dir': self.data.input_image_dir,
                'target_image_dir': self.data.target_image_dir,
                'prompt': self.data.prompt,
                'resolution': self.data.resolution,
                'batch_size': self.data.batch_size,
                'num_workers': self.data.num_workers,
            },
            'train': {
                'num_epochs': self.train.num_epochs,
                'learning_rate': self.train.learning_rate,
                'lr_scheduler': self.train.lr_scheduler,
                'gradient_accumulation_steps': self.train.gradient_accumulation_steps,
                'max_grad_norm': self.train.max_grad_norm,
            },
            'save': {
                'output_dir': self.save.output_dir,
                'save_every_n_epochs': self.save.save_every_n_epochs,
                'save_dtype': self.save.save_dtype,
            },
        }

    def save_yaml(self, yaml_path: Union[str, Path]):
        """
        保存配置到YAML文件

        Args:
            yaml_path: 保存路径
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def get_dtype(self):
        """
        获取模型数据类型的torch.dtype对象

        Returns:
            torch.dtype: 对应的torch数据类型
        """
        import torch
        dtype_mapping = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        return dtype_mapping.get(self.model.dtype, torch.bfloat16)

    # 便捷属性：直接访问train子配置的常用属性
    @property
    def gradient_accumulation_steps(self):
        """梯度累积步数"""
        return self.train.gradient_accumulation_steps

    @property
    def num_epochs(self):
        """训练轮数"""
        return self.train.num_epochs

    @property
    def max_sequence_length(self):
        """最大序列长度"""
        return self.train.max_sequence_length

    @property
    def cfg_scale(self):
        """CFG scale"""
        return self.train.cfg_scale

    @property
    def max_grad_norm(self):
        """最大梯度范数（用于梯度裁剪）"""
        return self.train.max_grad_norm

    @property
    def save_every_n_epochs(self):
        """保存检查点的epoch间隔"""
        return self.save.save_every_n_epochs

    def validate(self):
        """
        验证配置的有效性

        Raises:
            ValueError: 配置无效
        """
        # 验证设备
        if self.device not in ['cuda', 'cpu', 'mps']:
            raise ValueError(f"Invalid device: {self.device}")

        # 验证dtype
        if self.model.dtype not in ['float32', 'float16', 'bfloat16']:
            raise ValueError(f"Invalid dtype: {self.model.dtype}")

        # 验证LoRA参数
        if self.lora.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.lora.rank}")
        if self.lora.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.lora.alpha}")
        if not 0 <= self.lora.dropout < 1:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {self.lora.dropout}")

        # 验证训练参数
        if self.train.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.train.num_epochs}")
        if self.train.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.train.learning_rate}")
        if self.train.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.train.batch_size}")

        # 验证数据路径
        input_dir = Path(self.data.input_image_dir)
        target_dir = Path(self.data.target_image_dir)
        if not input_dir.exists():
            raise ValueError(f"Input image directory not found: {input_dir}")
        if not target_dir.exists():
            raise ValueError(f"Target image directory not found: {target_dir}")

    def __repr__(self) -> str:
        return (
            f"TrainingConfig(\n"
            f"  job_name='{self.job_name}',\n"
            f"  device='{self.device}',\n"
            f"  lora.rank={self.lora.rank},\n"
            f"  train.learning_rate={self.train.learning_rate},\n"
            f"  train.num_epochs={self.train.num_epochs}\n"
            f")"
        )
