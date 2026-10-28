"""
Mock FLUX模型 - 用于快速测试训练流程

这是一个轻量级的FLUX模型替代品，保持与真实FLUX相同的接口，
但内部使用简单的线性层，可以在普通硬件上快速运行。

用途：
- 测试完整的训练流程
- 验证各模块的协作
- 开发和调试
- 不需要下载50GB的真实模型
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# Mock配置类
# ============================================================================

@dataclass
class MockFluxConfig:
    """Mock FLUX Transformer配置"""
    in_channels: int = 32  # 输入通道数（16*2，因为trainer会concat input和target）
    num_layers: int = 4  # Transformer层数（真实FLUX有19+38层）
    hidden_size: int = 256  # 隐藏层大小（真实FLUX是3072）
    num_attention_heads: int = 4  # 注意力头数（真实FLUX是24）
    guidance_embeds: bool = True  # 是否支持guidance


@dataclass
class MockVAEConfig:
    """Mock VAE配置"""
    latent_channels: int = 16  # latent通道数
    shift_factor: float = 0.0
    scaling_factor: float = 1.0


# ============================================================================
# Mock Transformer
# ============================================================================

class MockFluxTransformer2DModel(nn.Module):
    """
    Mock版本的FluxTransformer2DModel

    保持与真实FLUX相同的接口，但内部使用简单的线性层
    """

    def __init__(self, config: Optional[MockFluxConfig] = None):
        super().__init__()

        if config is None:
            config = MockFluxConfig()

        self.config = config

        # 简单的线性层组成的"transformer"
        self.input_proj = nn.Linear(config.in_channels, config.hidden_size)

        # 模拟多层transformer blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            for _ in range(config.num_layers)
        ])

        self.output_proj = nn.Linear(config.hidden_size, config.in_channels)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        # Text condition projection (简化版)
        self.text_proj = nn.Linear(768, config.hidden_size)  # CLIP/T5的hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        txt_ids: Optional[torch.Tensor] = None,
        img_ids: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[dict] = None,
        return_dict: bool = False,
    ):
        """
        前向传播（保持与真实FLUX相同的接口）

        Args:
            hidden_states: [batch, channels, height, width] 或 [batch, seq_len, dim]
            timestep: [batch] 时间步
            encoder_hidden_states: [batch, seq_len, 768] 文本编码
            pooled_projections: [batch, 768] pooled文本特征
            guidance: [batch] guidance scale
            txt_ids: 文本位置ID
            img_ids: 图像位置ID
            joint_attention_kwargs: 额外的attention参数
            return_dict: 是否返回字典
        """
        # 1. 处理输入形状
        if hidden_states.dim() == 4:
            # [batch, channels, H, W] -> [batch, H*W, channels]
            batch, channels, H, W = hidden_states.shape
            hidden_states = hidden_states.flatten(2).transpose(1, 2)
            is_image = True
        else:
            # [batch, seq_len, dim]
            batch, seq_len, dim = hidden_states.shape
            is_image = False
            H, W = None, None

        # 2. 投影到hidden_size
        x = self.input_proj(hidden_states)  # [batch, seq_len, hidden_size]

        # 3. 添加时间步嵌入
        if timestep is not None:
            if timestep.dim() == 1:
                timestep = timestep.unsqueeze(1).float()  # [batch, 1]
            # 转换到模型的dtype
            timestep = timestep.to(dtype=x.dtype)
            time_emb = self.time_embed(timestep)  # [batch, hidden_size]
            x = x + time_emb.unsqueeze(1)  # broadcast到seq_len

        # 4. 添加文本条件
        if encoder_hidden_states is not None:
            text_emb = self.text_proj(encoder_hidden_states)  # [batch, text_len, hidden_size]
            # 简化：只使用mean pooling
            text_emb = text_emb.mean(dim=1, keepdim=True)  # [batch, 1, hidden_size]
            x = x + text_emb

        # 5. 通过transformer blocks
        for block in self.blocks:
            x = block(x) + x  # 残差连接

        # 6. 输出投影
        output = self.output_proj(x)  # [batch, seq_len, in_channels]

        # 7. 恢复原始形状
        if is_image:
            # [batch, H*W, channels] -> [batch, channels, H, W]
            output = output.transpose(1, 2).reshape(batch, -1, H, W)

        if return_dict:
            return {"sample": output}
        else:
            return (output,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """模拟from_pretrained接口"""
        # 忽略真实的模型路径，直接创建mock模型
        config = MockFluxConfig()
        return cls(config)


# ============================================================================
# Mock VAE
# ============================================================================

class MockAutoencoder(nn.Module):
    """
    Mock版本的AutoencoderKL
    """

    def __init__(self, config: Optional[MockVAEConfig] = None):
        super().__init__()

        if config is None:
            config = MockVAEConfig()

        self.config = config

        # 简单的卷积编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # -> 64 x H/2 x W/2
            nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # -> 128 x H/4 x W/4
            nn.GELU(),
            nn.Conv2d(128, config.latent_channels * 2, 3, 1, 1),  # -> latent_channels*2 x H/4 x W/4
        )

        # 简单的卷积解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(config.latent_channels, 128, 3, 1, 1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # -> 64 x H/2 x W/2
            nn.GELU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # -> 3 x H x W
        )

    class LatentDistribution:
        """模拟latent分布"""
        def __init__(self, parameters):
            # parameters: [batch, latent_channels*2, H, W]
            # 分成mean和logvar
            self.mean = parameters[:, :parameters.shape[1]//2]
            self.logvar = parameters[:, parameters.shape[1]//2:]

        def sample(self):
            # 重参数化技巧
            std = torch.exp(0.5 * self.logvar)
            eps = torch.randn_like(std)
            return self.mean + eps * std

    def encode(self, x):
        """编码图像到latent"""
        parameters = self.encoder(x)
        return type('obj', (object,), {
            'latent_dist': self.LatentDistribution(parameters)
        })()

    def decode(self, z):
        """解码latent到图像"""
        return self.decoder(z)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """模拟from_pretrained接口"""
        config = MockVAEConfig()
        return cls(config)


# ============================================================================
# Mock Text Encoders
# ============================================================================

class MockCLIPTextModel(nn.Module):
    """Mock CLIP文本编码器"""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(50000, hidden_size)  # 词表大小
        self.pooler = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, output_hidden_states=False, **kwargs):
        # 简单的embedding
        hidden_states = self.embedding(input_ids)  # [batch, seq_len, hidden_size]

        # Pooled output (使用第一个token，类似CLIP的CLS token)
        pooler_output = self.pooler(hidden_states[:, 0])  # [batch, hidden_size]

        return type('obj', (object,), {
            'last_hidden_state': hidden_states,
            'pooler_output': pooler_output,
        })()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return cls()


class MockT5EncoderModel(nn.Module):
    """Mock T5文本编码器"""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(50000, hidden_size)

    def forward(self, input_ids, output_hidden_states=False, **kwargs):
        hidden_states = self.embedding(input_ids)

        return type('obj', (object,), {
            'last_hidden_state': hidden_states,
        })()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return cls()


# ============================================================================
# Mock Tokenizers
# ============================================================================

class MockTokenizer:
    """Mock Tokenizer"""

    def __init__(self, model_max_length=77):
        self.model_max_length = model_max_length

    def __call__(self, text, padding=None, max_length=None, truncation=True, return_tensors=None):
        if isinstance(text, str):
            text = [text]

        batch_size = len(text)
        max_len = max_length or self.model_max_length

        # 生成随机token IDs
        input_ids = torch.randint(0, 50000, (batch_size, max_len))

        return type('obj', (object,), {
            'input_ids': input_ids,
        })()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        max_length = kwargs.get('model_max_length', 77)
        return cls(model_max_length=max_length)


# ============================================================================
# Mock Scheduler
# ============================================================================

class MockFlowMatchScheduler:
    """Mock Flow Matching Scheduler"""

    def __init__(self, num_train_timesteps=1000):
        self.config = type('obj', (object,), {
            'num_train_timesteps': num_train_timesteps
        })()
        self.timesteps = torch.arange(0, num_train_timesteps)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return cls()


# ============================================================================
# Mock Model Loader（整合所有mock组件）
# ============================================================================

class MockFluxModelLoader:
    """
    Mock版本的ModelLoader

    返回所有必需的mock组件，用于测试训练流程
    """

    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

    def load_all(self) -> Tuple:
        """
        加载所有mock组件

        Returns:
            (transformer, vae, text_encoder_1, text_encoder_2,
             tokenizer_1, tokenizer_2, scheduler)
        """
        print("\n" + "="*60)
        print("🎭 加载Mock FLUX模型（轻量级测试版本）")
        print("="*60)

        # 创建mock组件
        transformer = MockFluxTransformer2DModel().to(self.device, dtype=self.dtype)
        vae = MockAutoencoder().to(self.device, dtype=self.dtype)
        text_encoder_1 = MockCLIPTextModel().to(self.device, dtype=self.dtype)
        text_encoder_2 = MockT5EncoderModel().to(self.device, dtype=self.dtype)
        tokenizer_1 = MockTokenizer(model_max_length=77)
        tokenizer_2 = MockTokenizer(model_max_length=512)
        scheduler = MockFlowMatchScheduler()

        # 统计参数
        transformer_params = sum(p.numel() for p in transformer.parameters())
        vae_params = sum(p.numel() for p in vae.parameters())

        print(f"✓ Transformer: {transformer_params:,} 参数")
        print(f"✓ VAE: {vae_params:,} 参数")
        print(f"✓ Text Encoders: Mock版本")
        print(f"✓ Scheduler: Mock Flow Matching")
        print(f"✓ Device: {self.device}")
        print(f"✓ Dtype: {self.dtype}")
        print("="*60 + "\n")

        return (
            transformer,
            vae,
            text_encoder_1,
            text_encoder_2,
            tokenizer_1,
            tokenizer_2,
            scheduler,
        )

    def load_transformer_only(self):
        """只加载transformer（用于测试）"""
        return MockFluxTransformer2DModel().to(self.device, dtype=self.dtype)


# ============================================================================
# 便捷函数
# ============================================================================

def create_mock_flux_components(device="cuda", dtype=torch.bfloat16):
    """
    创建所有mock FLUX组件的便捷函数

    Args:
        device: 设备
        dtype: 数据类型

    Returns:
        包含所有组件的元组
    """
    loader = MockFluxModelLoader(device=device, dtype=dtype)
    return loader.load_all()


if __name__ == "__main__":
    # 测试mock模型
    print("测试Mock FLUX模型")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建mock组件
    components = create_mock_flux_components(device=device, dtype=torch.float32)
    transformer, vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, scheduler = components

    # 测试VAE
    print("\n测试VAE:")
    test_image = torch.randn(2, 3, 256, 256).to(device)
    latent = vae.encode(test_image).latent_dist.sample()
    print(f"  输入图像: {test_image.shape}")
    print(f"  Latent: {latent.shape}")

    # 测试Text Encoders
    print("\n测试Text Encoders:")
    test_text = ["a cat", "a dog"]
    tokens_1 = tokenizer_1(test_text, max_length=77)
    output_1 = text_encoder_1(tokens_1.input_ids.to(device))
    print(f"  CLIP output: {output_1.pooler_output.shape}")

    tokens_2 = tokenizer_2(test_text, max_length=512)
    output_2 = text_encoder_2(tokens_2.input_ids.to(device))
    print(f"  T5 output: {output_2.last_hidden_state.shape}")

    # 测试Transformer
    print("\n测试Transformer:")
    test_latent = torch.randn(2, 64, 64, 64).to(device)
    test_timestep = torch.randint(0, 1000, (2,)).to(device)
    output = transformer(
        hidden_states=test_latent,
        timestep=test_timestep,
        encoder_hidden_states=output_2.last_hidden_state,
        pooled_projections=output_1.pooler_output,
    )
    print(f"  输入: {test_latent.shape}")
    print(f"  输出: {output[0].shape}")

    print("\n✅ Mock FLUX模型测试通过！")
