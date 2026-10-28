# -*- coding: utf-8 -*-
"""训练器模块 - 核心训练循环"""

# ============================================================================
# 第一部分：依赖与基础说明
#   - 引入训练所需的外部库与工具函数
#   - 这些导入在文件级别初始化，不涉及业务逻辑
# ============================================================================
import os                                        # 路径处理
from contextlib import nullcontext               # 统一 with 接口，占位用

import torch                                     # 核心张量/自动求导库
import torch.nn.functional as F                  # 常用函数式 API（loss 等）
from optimum.quanto import QTensor               # Quanto 量化张量类型
from einops import rearrange, repeat             # 张量形状重排与重复
from tqdm import tqdm                            # 进度条
from PIL import Image                            # 保存/处理采样图片
from diffusers.hooks import apply_group_offloading  # 混合 offload（block/leaf level）
from safetensors.torch import save_file             # 临时保存 LoRA
from peft import LoraConfig                        # 构造验证用 LoRA 配置
from peft.utils import (
    get_peft_model_state_dict,                     # 导出 LoRA/MoE 权重（含路由器）
    set_peft_model_state_dict,                     # 安全加载 LoRA 权重
)                                                  

# ============================================================================
# 第二部分：Trainer 类定义与初始化
#   - 绑定全局配置/日志
#   - 设定训练设备与默认精度
#   - 按精度选择是否启用 GradScaler，并兼容 torch 新旧接口
# ============================================================================
class Trainer:
    """训练器 - 负责执行训练循环"""
    
    def __init__(self, config, logger):
        self.config = config                      # 全局配置对象：包含模型/训练/数据等参数
        self.logger = logger                      # 日志记录器
        self.device = config.device               # 训练目标设备（"cuda" / "cpu"）
        self.dtype = config.get_dtype()           # 训练默认精度，保持与模型 dtype 一致
        self.global_step = 0                      # 记录已执行的 step 数
        self.scheduler = None                     # 训练时用到的调度器（flow-match）
        self.use_fp16 = self.dtype == torch.float16  # 仅 fp16 需要 GradScaler；bf16/float32 不缩放

        # GradScaler 作用：fp16 计算的动态范围窄，反向时梯度易下溢为 0。
        #   - GradScaler 会先用一个缩放因子 scale 放大 loss（等价放大所有梯度），再反向传播，
        #   - 之后在 optimizer.step() 之前自动把梯度除以 scale 恢复原值（unscale），并检测溢出。
        #   - 若检测到 inf/NaN，会跳过 step 并下调 scale，避免损坏权重。
        # 在 bf16/float32 下动态范围足够，不需要缩放。下面是选择与启用条件：
        # 1) 优先 torch.amp.GradScaler（新接口，需传设备字符串，如 "cuda"）
        # 2) 若当前 PyTorch 版本无该接口，则回退 torch.cuda.amp.GradScaler（旧接口）
        # 3) enabled 仅在 fp16 且 CUDA 可用时为 True，避免在 bf16/CPU 误用缩放
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            # 新接口：torch.amp.GradScaler(device='cuda', enabled=...)
            self.grad_scaler = torch.amp.GradScaler(
                "cuda",
                enabled=self.use_fp16 and torch.cuda.is_available(),
            )
        else:
            # 旧接口：torch.cuda.amp.GradScaler(enabled=...)
            self.grad_scaler = torch.cuda.amp.GradScaler(
                enabled=self.use_fp16 and torch.cuda.is_available()
            )
    
# ============================================================================
# 第三部分：文本编码辅助函数
#   - encode_prompt：批量编码提示词，获得三路输入：
#       prompt_embeds（T5 序列特征），pooled_prompt_embeds（CLIP 全局特征），text_ids（位置/角色占位）
#   - 目的：一次性在目标设备上完成双编码器推理，避免频繁搬运
# ============================================================================
    def encode_prompt(self, prompts, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2):
        """编码文本提示词（支持CPU和GPU text encoder）"""
        # 获取text encoder所在的设备
        te1_device = next(text_encoder_1.parameters()).device
        te2_device = next(text_encoder_2.parameters()).device

        # CLIP tokenizer
        text_inputs_1 = tokenizer_1(
            prompts,
            padding="max_length",
            max_length=tokenizer_1.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # 将input_ids移到text encoder所在的设备（可能是CPU或GPU）
        text_input_ids_1 = text_inputs_1.input_ids.to(te1_device, dtype=torch.long)
        with torch.no_grad():
            text_encoder_output_1 = text_encoder_1(
                text_input_ids_1,
                output_hidden_states=False,
            )
        pooled_prompt_embeds = text_encoder_output_1.pooler_output

        # 将结果移到训练设备（GPU）
        pooled_prompt_embeds = pooled_prompt_embeds.to(self.device)

        # T5 tokenizer
        text_inputs_2 = tokenizer_2(
            prompts,
            padding="max_length",
            max_length=self.config.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        # 将input_ids移到text encoder所在的设备（可能是CPU或GPU）
        text_input_ids_2 = text_inputs_2.input_ids.to(te2_device, dtype=torch.long)
        with torch.no_grad():
            text_encoder_output_2 = text_encoder_2(
                text_input_ids_2,
                output_hidden_states=False,
            )
        prompt_embeds = text_encoder_output_2.last_hidden_state

        # 将结果移到训练设备（GPU）
        prompt_embeds = prompt_embeds.to(self.device)

        # Text IDs 占位张量（在训练设备上创建）
        text_ids = torch.zeros(
            prompt_embeds.shape[0],
            prompt_embeds.shape[1],
            3,
            device=self.device,
            dtype=prompt_embeds.dtype,
        )

        return prompt_embeds, pooled_prompt_embeds, text_ids

    # ============================================================================
    # 第四部分：FlowMatch 噪声合成
    #   - 功能：按时间步 t_01 在 clean latent 与噪声之间线性插值，得到训练用 noisy 输入
    #   - 公式：noisy = (1 - t) * clean + t * noise，t∈[0,1] 表示从干净到全噪声的混合程度
    # ============================================================================
    @staticmethod
    def _flowmatch_add_noise(latents, noise, timesteps):
        """
        latents: 目标 clean latent，形状与 noise 相同
        noise: 高斯噪声
        timesteps: 整数时间步（0..1000），表示混合比例
        返回：noisy_model_input，与 latents 同形状
        """
        # 1) 将整数时间步映射到 [0,1]，并放到 latent 所在设备
        t_01 = (timesteps / 1000).to(latents.device)
        # 2) 按比例线性插值，t=0 时全 clean，t=1 时全 noise
        noisy_model_input = (1.0 - t_01) * latents + t_01 * noise
        return noisy_model_input
    
    # ============================================================================
    # 第五部分：训练主循环
    #   - 准备：冻结基础模型、统计可训练参数、设置混合精度上下文
    #   - 主循环：迭代 epoch 和 batch，完成文本/图像编码、噪声混合、前向、loss、反向与优化
    #   - 可选验证：后续 epoch 结束时可触发采样验证（由调用方传入 pipeline 控制）
    # ============================================================================
    # 入口与依赖
    def train(
        self,
        transformer,        # Flux Transformer（含 LoRA）
        vae,                # VAE 编解码器（只做 encode，不训练）
        text_encoder_1,     # CLIP 文本编码器
        text_encoder_2,     # T5 文本编码器
        tokenizer_1,        # CLIP tokenizer
        tokenizer_2,        # T5 tokenizer
        scheduler,          # FlowMatch 调度器（提供 timesteps）
        train_dataloader,   # 训练数据迭代器（含图像/latent/prompt）
        optimizer,          # 优化器
        lr_scheduler,       # 学习率调度器
        validation_pipeline=None,  # 可选，按 epoch 采样验证的 pipeline
    ):
        """执行训练"""
        # 基础日志与调度器准备
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"[训练开始] 开始 LoRA 微调训练")
        self.logger.info(f"{'='*60}")

        self.scheduler = scheduler                                # 保存调度器引用
        train_timesteps = scheduler.timesteps.to(self.device)     # 训练用时间步，移到目标设备
        num_train_timesteps = scheduler.config.num_train_timesteps  # 总时间步数量（未直接使用，保留一致性）

        # 冻结非训练部分
        # - VAE、CLIP、T5 仅做推理，不参与训练
        vae.requires_grad_(False)
        vae.eval()
        text_encoder_1.requires_grad_(False)
        text_encoder_1.eval()
        text_encoder_2.requires_grad_(False)
        text_encoder_2.eval()

        # - Transformer 仅训练 LoRA/MoE-LoRA 相关参数，其余参数冻结。
        #   直接基于命名约定放行 LoRA/MoE 参数（包含路由器/专家），其余冻结。
        for name, param in transformer.named_parameters():
            if any(
                tag in name
                for tag in (
                    "lora_",        # 标准/大部分 MoE 参数
                    "lora_A",       # 专家 A
                    "lora_B",       # 专家 B
                    "lora_route",   # 路由器
                )
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 统计并记录可训练参数规模（验证 LoRA 冻结是否生效）
        trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in transformer.parameters())
        self.logger.info(
            f"[训练准备] 可训练参数：{trainable_params:,} / {all_params:,} ({100*trainable_params/all_params:.2f}%)"
        )

        # 混合精度与梯度累积准备
        # 是否开启 autocast：仅当训练 dtype 是 fp16 或 bf16 时才自动混合精度，否则用全精度
        use_autocast = self.dtype in {torch.float16, torch.bfloat16}
        autocast_dtype = self.dtype if use_autocast else None         # autocast 目标 dtype
        autocast_device = "cuda" if self.device.startswith("cuda") else "cpu"  # autocast 设备

        grad_acc_steps = max(1, self.config.gradient_accumulation_steps)  # 梯度累积步数下限 1
        # zero_grad: set_to_none=True 将梯度置 None 而不是 0，能减少显存占用，且下一步会重新分配梯度张量
        optimizer.zero_grad(set_to_none=True)

        # 训练循环：外层 epoch，内层 iter
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            self.logger.info(f"{'='*60}")

            transformer.train()      # 确保 LoRA 等训练态
            epoch_loss = 0.0         # 累计本 epoch 的平均损失

            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                prompts = batch["prompt"]  # 当前 batch 的文本提示列表

                # 优先使用预缓存的 latents（节省算力/显存）
                if "input_latent" in batch:
                    input_latents = batch["input_latent"].to(self.device, dtype=self.dtype)    # 控制图 latent
                    target_latents = batch["target_latent"].to(self.device, dtype=self.dtype)  # 目标图 latent
                else:
                    # ============================================================================
                    # 实时编码图像（未缓存时）
                    #   - 目的：当未预缓存 latent 时，VAE 编码原图得到 latent
                    #   - 难点/原因：
                    #       1) VAE 可能被 offload 在 CPU，为省显存需按需搬到 GPU，再搬回。
                    #       2) 编码用 no_grad 避免存梯度，节约显存/算力。
                    #       3) latent 需按模型配置做 shift/scaling，匹配下游分布。
                    # ============================================================================
                    input_images = batch["input_image"].to(self.device, dtype=self.dtype)      # 控制输入图
                    target_images = batch["target_image"].to(self.device, dtype=self.dtype)    # 目标图

                    # 编码图像 -> latent
                    with torch.no_grad():  # 编码阶段不需要梯度，避免额外显存与反传开销
                        vae_device_original = next(vae.parameters()).device  # 记录 VAE 原始设备（可能在 CPU）
                        if vae_device_original != torch.device(self.device):
                            vae.to(self.device)  # 临时搬到训练设备，编码更快

                        # 采样控制图 latent：
                        #   - VAE encode 返回正态分布 N(mu, sigma)，latent_dist.sample() ~ N(mu, sigma)
                        #   - 采样而非直接用 mu，可注入轻微噪声，与训练时一致
                        input_latents = vae.encode(input_images).latent_dist.sample()
                        # 对 latent 做 shift/scaling：
                        #   - 公式：latent = (z - shift_factor) * scaling_factor
                        #   - FLUX VAE 预训练时的后处理也是这么做的，推理/再训练要复用相同变换，
                        #     以保证传给 Transformer 的 latent 分布与预训练一致（不然值域偏移会导致输出异常）。
                        input_latents = (input_latents - vae.config.shift_factor) * vae.config.scaling_factor

                        # 同理编码目标图，得到目标 latent（同样采样+shift/scale）
                        target_latents = vae.encode(target_images).latent_dist.sample()
                        target_latents = (target_latents - vae.config.shift_factor) * vae.config.scaling_factor

                        # 编码完成后将 VAE 放回原设备（如 CPU），继续 offload 节省显存
                        if vae_device_original != torch.device(self.device):
                            vae.to(vae_device_original)

                batch_size = input_latents.shape[0]  # 当前 batch 大小

                # 文本编码器保持在CPU进行编码（节省显存）
                # T5-XXL太大(4.7GB)，如果移到GPU会导致OOM
                # 因此直接在CPU上编码，只移动结果到GPU
                te1_device_original = next(text_encoder_1.parameters()).device
                te2_device_original = next(text_encoder_2.parameters()).device

                # 编码文本（text encoder保持在CPU）
                prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                    prompts, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2
                )

                # ============================================================================
                # 噪声注入与输入拼接（FlowMatch 训练核心）
                #   - 目标：构造 noisy target latent，公式 noisy = (1 - t) * clean + t * noise
                #   - 采样随机时间步 t，控制噪声混合程度，覆盖不同噪声强度的训练场景
                #   - 将 noisy target 与 control latent 拼接，供 transformer 同时看条件与目标
                # ============================================================================
                # 1) 采样与目标 latent 同形状的高斯噪声
                noise = torch.randn_like(target_latents)
                # 2) 随机抽取时间步索引（0..len-1），batch 维度独立
                timestep_indices = torch.randint(
                    0, train_timesteps.shape[0], (batch_size,), device=self.device
                )
                timesteps = train_timesteps[timestep_indices]  # 得到实际时间步值，形状 (batch,)
                # 3) 按公式 noisy = (1 - t) * clean + t * noise，生成带噪目标 latent
                noisy_latents = self._flowmatch_add_noise(target_latents, noise, timesteps)

                # 4) 拼接 control latent（input）和 noisy target latent：
                #    形状变为 (batch, 2 * C, H, W)，让模型同时看到控制图和目标图的带噪版本
                latent_model_input = torch.cat([noisy_latents, input_latents], dim=1)

                # ============================================================================
                # latent 打包为 transformer token（与 flux_kontext.get_noise_prediction 对齐）
                #   目标：
                #     1) 把 (B, C=2*8, H, W) 的 latent/控制拼接图重排为 token 序列
                #     2) 若有 control，先将 control 挪到 batch 维，便于后续按 token 再拼回
                #   示例：
                #     - 输入 latent_model_input 形状 (B=1, C=16, H=128, W=128)
                #     - 若有 control：分块后 (2B=2, C/2=8, H=128, W=128)
                #     - rearrange 后：形状 (2B, H*W=16384, C*2*2=32)
                # ============================================================================
                bs, _, h, w = latent_model_input.shape                     # 原始 batch/通道/高/宽
                has_control = latent_model_input.shape[1] != target_latents.shape[1]  # 判断是否包含 control 通道
                # 若有 control，将 control 放到 batch 维：latent_model_input -> (2B, C/2, H, W)
                if has_control:
                    latent, control = torch.chunk(latent_model_input, 2, dim=1)
                    latent_model_input = torch.cat([latent, control], dim=0)  # (2B, 8, H, W) 以 flux 通道数为例

                # 将 2x2 patch 打包为 token：
                #   - ph=pw=2，卷积式展开，每个 token 覆盖 2x2 空间块
                #   - 输出形状 (batch, h*w, c*ph*pw)，匹配 transformer 输入 (批, token数, 通道)
                latent_model_input = rearrange(
                    latent_model_input,
                    "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                    ph=2,
                    pw=2,
                )

                # ============================================================================
                # 位置编码（与 token 对齐）
                #   目标：为每个 token 构建行/列坐标，control token 加类型标记
                #   流程：
                #     a) 先构建单张的坐标网格 (H/2, W/2, 3)：因为前面用 2x2 patch 打包，每个 token 覆盖原图 2x2 区块，
                #        token 网格大小为 (H/2, W/2)。通道 1/2 存行列索引，0 通道预留类型标记。
                #     b) repeat 到 batch 维，得到 (B 或 2B, token, 3)
                #     c) 若有 control，复制一份 ctrl_ids 并在通道 0 标记为 1，再与 img_ids 拼在 token 维
                #     d) 将 control 再从 batch 维拼回 token 维，恢复 (B, 2*token, dim)
                # ============================================================================
                img_ids = torch.zeros(h // 2, w // 2, 3, device=self.device, dtype=latent_model_input.dtype)  # token 网格大小是原高宽/2
                # 行坐标：torch.arange(h//2) 生成 [0..h/2-1]，通过 [:, None] 扩展为列向量，广播到每一列
                img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=self.device)[:, None]
                # 列坐标：torch.arange(w//2) 生成 [0..w/2-1]，通过 [None, :] 扩展为行向量，广播到每一行
                img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=self.device)[None, :]
                img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)  # 展开到 (batch, token, 3)

                if has_control:
                    # 为 control token 复制一份位置编码，并在通道 0 置 1 作为类型标记
                    ctrl_ids = img_ids.clone()
                    ctrl_ids[..., 0] = 1  # 标记控制图像 token
                    img_ids = torch.cat([img_ids, ctrl_ids], dim=1)  # token 维度加倍
                    # 将 control 再从 batch 维拼回 token 维：latent_model_input 由 (2B, ...) 还原到 (B, 2*token, dim)
                    latent, control = torch.chunk(latent_model_input, 2, dim=0)
                    latent_model_input = torch.cat([latent, control], dim=1)
                    latent_size = latent.shape[1]  # 单张图的 token 数（h*w），后面用它从模型输出中截出“目标图”那一半
                else:
                    latent_size = latent_model_input.shape[1]

                # ============================================================================
                # ID 维度调整与 dtype 对齐
                #   目标：
                #     1) 将 txt_ids/img_ids 去掉 batch 维，符合 transformer 接口预期（(seq_len, dim)）
                #     2) 对输入张量做 dtype 统一：保持训练 dtype（bf16/fp16），避免误升到 fp32 占显存
                #   为什么只对这些张量做 to(dtype)：
                #     - latent_model_input/prompt_embeds/pooled_prompt_embeds：都是作为 transformer 输入的激活，
                #       用训练 dtype（bf16/fp16）能减少显存；LoRA 已是 fp32 参数，计算会自动提升为更高精度。
                #     - timestep_norm：是标量序列，保持 float（默认 fp32）即可，但需搬到 device。
                #     - 其他张量（noise/target_latents/input_latents/noisy_latents）在上游创建时就用 self.dtype，
                #       无需再转 fp32，否则会额外占显存且与混合精度策略相悖。
                # ============================================================================
                txt_ids = text_ids[0] if text_ids.ndim == 3 else text_ids
                img_ids_model = img_ids[0] if img_ids.ndim == 3 else img_ids

                timestep_norm = timesteps / float(num_train_timesteps)

                cast_dtype = self.dtype  # 训练 dtype（如 bf16），不强制升到 transformer.dtype
                latent_model_input = latent_model_input.to(self.device, dtype=cast_dtype)
                prompt_embeds = prompt_embeds.to(self.device, dtype=cast_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(self.device, dtype=cast_dtype)
                timestep_norm = timestep_norm.to(self.device)
                # 其余张量（noise/target_latents/input_latents/noisy_latents）已在上游用训练 dtype 创建/转换，
                # 不再强制转 fp32，避免显存占用增加

                # ============================================================================
                # guidance embedding scale（Flux 专用的时间/文本引导模块）
                #   背景：
                #     - Flux 使用 CombinedTimestepGuidanceTextProjEmbeddings，将 timestep、guidance scale、
                #       文本投影等信息组合成嵌入，输入 transformer。
                #     - guidance 类似“无分类器引导”的强度系数：值越大，模型越重视文本/条件的引导。
                #   逻辑：
                #     - 只有当 transformer.config.guidance_embeds=True 时才需要该向量；
                #     - 构造 shape=(batch,) 的常数向量，值来自配置 cfg_scale；
                #     - dtype 取训练 dtype（cast_dtype），放在训练设备上。
                #   例：cfg_scale=2.5 -> guidance = [2.5, 2.5, ...] (batch 长度)，供模块生成对应的引导嵌入。
                # ============================================================================
                guidance = torch.full(
                    (batch_size,),
                    float(self.config.cfg_scale),
                    device=self.device,
                    dtype=cast_dtype,
                ) if transformer.config.guidance_embeds else None

                # ============================================================================
                # 前向推理（带 autocast 与 quantized 输出处理）
                #   - autocast_ctx：根据 dtype/设备决定是否开启自动混合精度
                #   - transformer 前向：传入 latent/token IDs/timestep/guidance/text 嵌入
                #   - 若输出为 QTensor（quanto 量化张量），需先反量化再参与 loss
                # ============================================================================
                autocast_ctx = (
                    torch.autocast(device_type=autocast_device, dtype=autocast_dtype)
                    if use_autocast and autocast_dtype is not None
                    else nullcontext()
                )
                with autocast_ctx:
                    model_pred = transformer(
                        hidden_states=latent_model_input,          # (B, tokens, dim)
                        timestep=timestep_norm,                    # (B,) 归一化时间步
                        guidance=guidance,                         # (B,) guidance scale 或 None
                        pooled_projections=pooled_prompt_embeds,   # (B, hidden) pooled 文本向量
                        encoder_hidden_states=prompt_embeds,       # (B, seq_len, hidden) 文本序列
                        txt_ids=txt_ids,                           # (seq_len, 3) 文本位置/占位
                        img_ids=img_ids_model,                     # (token, 3) 图像位置/类型
                        joint_attention_kwargs=None,               # joint attention 额外参数：此处传 None，使用模型默认注意力；
                                                                   # 若未来需要自定义联合注意力（额外条件/掩码等），可传入字典
                        return_dict=False,                         # 返回 tuple，便于直接取 [0]
                    )[0]

                    # 如果 Transformer 使用 quanto 量化，输出可能是 QTensor；训练需用反量化的 float
                    if isinstance(model_pred, QTensor):
                        model_pred = model_pred.dequantize()

                    # ============================================================================
                    # 还原预测形状：只保留目标 token，并把 token 序列还原回 (B, C, H, W)
                    #   流程：
                    #     1) 截取前 latent_size 个 token（对应目标图），丢弃 control token 预测
                    #     2) rearrange 把 token 序列重排回图像形状，2x2 patch 复原到空间维度
                    #   示例：latent_size = H/2 * W/2，model_pred 形状 (B, 2*latent_size, dim) 有目标+控制，
                    #         截取后 (B, latent_size, dim) 只剩目标，再还原到 (B, C, H, W)
                    # ============================================================================
                    model_pred = model_pred[:, :latent_size]  # 只要目标图 token
                    model_pred = rearrange(
                        model_pred,
                        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                        h=h // 2,
                        w=w // 2,
                        ph=2,
                        pw=2,
                        c=target_latents.shape[1],
                    )

                    # ============================================================================
                    # Flow Matching 损失与反向
                    #   - 目标向量：v = noise - clean_latents（flow matching 的 target velocity）
                    #   - 损失：MSE(model_pred, target_velocity)
                    #   - 梯度累积：loss 除以 grad_acc_steps（等价于把总 loss 均分到每个累积 step），
                    #     这样累积 grad_acc_steps 次 backward 后的梯度之和与一次大 batch backward 等效。
                    # ============================================================================
                    target_velocity = (noise - target_latents)  # 目标速度向量
                    loss = F.mse_loss(model_pred, target_velocity, reduction="mean")
                    loss = loss / grad_acc_steps  # 将总 loss 均分到每个累积 step，累积次数后梯度等效大 batch

                # 按需使用 GradScaler 做混合精度反向
                if self.grad_scaler.is_enabled():
                    self.grad_scaler.scale(loss).backward()
                else:
                    loss.backward()

                # ============================================================================
                # 梯度累积步结束：梯度裁剪、优化器步进、学习率调度、清空梯度
                #   - 每 grad_acc_steps 次 backward 后执行一次优化步，等效于大 batch
                #   - GradScaler 场景下需先反缩放再裁剪，再 step/update
                # ============================================================================
                if (step + 1) % grad_acc_steps == 0:
                    if self.grad_scaler.is_enabled():
                        self.grad_scaler.unscale_(optimizer)  # 反缩放梯度，便于正确裁剪
                    torch.nn.utils.clip_grad_norm_(transformer.parameters(), self.config.max_grad_norm)  # 防梯度爆炸
                    if self.grad_scaler.is_enabled():
                        self.grad_scaler.step(optimizer)  # 用当前缩放后的梯度执行优化步（内部会处理溢出检测）
                        # update: 检测本次是否有溢出，若有则跳过 step 并降低 scale；否则可能提高 scale
                        self.grad_scaler.update()
                    else:
                        optimizer.step()
                    lr_scheduler.step()                   # 学习率调度
                    optimizer.zero_grad(set_to_none=True) # 清空梯度，置 None 省显存

                # 记录
                epoch_loss += loss.item() * grad_acc_steps
                self.global_step += 1

                progress_bar.set_postfix(
                    {
                        "loss": f"{(loss.item() * grad_acc_steps):.4f}",
                        "avg_loss": f"{epoch_loss / (step + 1):.4f}",
                    }
                )

            # 如果最后一个 mini-batch 没有凑满梯度累积，也要执行一次优化器更新
            if len(train_dataloader) % grad_acc_steps != 0:
                # ============================================================================
                # 收尾：当数据集大小不是 grad_acc_steps 的整数倍，最后不足一组的累积也要更新
                #   - 流程同上：反缩放->裁剪->step/update->lr_scheduler->zero_grad
                # ============================================================================
                if self.grad_scaler.is_enabled():
                    self.grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), self.config.max_grad_norm)
                if self.grad_scaler.is_enabled():
                    self.grad_scaler.step(optimizer)
                    self.grad_scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            avg_epoch_loss = epoch_loss / len(train_dataloader)
            self.logger.info(f"\n[Epoch {epoch + 1}] 平均损失：{avg_epoch_loss:.4f}")

            # 保存检查点：用 yield 通知外层（train_lora.py 的循环）触发保存逻辑
            # yield 会暂停生成器，将事件 ("save", epoch+1) 抛给调用方，外层可据此调用 saver.save_lora 等；
            # 处理完后外层继续迭代，训练流程不中断。
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                yield ("save", epoch + 1)

            # ============================================================================
            # 验证采样：按配置的周期触发，用验证 pipeline 生成样本图像
            #   条件：
            #     - validation_enabled=True
            #     - 提供了 validation_pipeline
            #     - (epoch+1) 能被 validation_every_n_epochs 整除
            #   行为：调用 _run_validation，用当前 LoRA/模型生成样本，便于监控效果
            # ============================================================================
            if (
                self.config.validation_enabled
                and validation_pipeline is not None
                and (epoch + 1) % self.config.validation_every_n_epochs == 0
            ):
                self._run_validation(
                    validation_pipeline,
                    transformer,
                    vae,
                    text_encoder_1,
                    text_encoder_2,
                    epoch + 1,
                )

        # 训练完成
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"[训练完成] 训练已完成！")
        self.logger.info(f"{'='*60}")
        yield ("save", None)

    # ============================================================================
    # 验证采样（按 epoch 触发）
    # - pipeline: FluxKontextPipeline，用于执行推理采样
    # - transformer: 当前训练的 Transformer（含 LoRA），在采样时切换 eval
    # - vae: VAE，用于解码；需暂时移动到推理设备
    # - text_encoder_1/2: CLIP 与 T5 文本编码器，用于生成提示词 embedding
    # - epoch_idx: 本次验证所属的 epoch 序号（用于文件命名、可选种子偏移）
    # ============================================================================
    def _run_validation(
        self,
        pipeline,          # FluxKontextPipeline 实例（此处不再使用，保持兼容签名）
        transformer,       # 训练中的 Transformer（含 LoRA），用于导出 LoRA 权重
        vae,               # VAE（未在 offload 验证中直接使用）
        text_encoder_1,    # CLIP（未在 offload 验证中直接使用）
        text_encoder_2,    # T5（未在 offload 验证中直接使用）
        epoch_idx: int,    # 当前验证对应的 epoch 序号，用于文件命名/种子偏移
    ):
        """按 epoch 采样验证"""
        # 若未提供验证提示词或验证图像，则直接跳过采样
        if not self.config.validation_prompts or not self.config.validation_image:
            self.logger.warning("[验证] 未提供 validation_prompts 或 validation_image，跳过验证采样")
            return

        # 准备输出目录与采样基础参数
        os.makedirs(self.config.validation_save_dir, exist_ok=True)  # 确保保存目录存在
        width = self.config.validation_width or self.config.resolution   # 采样宽度：优先配置，否则用训练分辨率
        height = self.config.validation_height or self.config.resolution # 采样高度：同上
        generator = torch.Generator(device=self.device)  # 固定设备的随机数发生器（保证可复现）
        if self.config.validation_seed is not None:
            generator.manual_seed(int(self.config.validation_seed) + epoch_idx)  # 可选种子，叠加 epoch 保证每轮不同

        # 为避免影响训练态，这里单独实例化一条推理管线，应用与 inference_lora.py 相同的混合 offload：
        # Transformer block_level；CLIP/T5 block_level；VAE leaf_level。
        tmp_lora_path = os.path.join(self.config.validation_save_dir, f"_tmp_lora_epoch_{epoch_idx}.safetensors")
        target_dtype = torch.float16

        # 记录训练状态，采样后恢复
        train_modes = {
            "transformer": transformer.training,
            "vae": vae.training if vae is not None else None,
            "text_encoder_1": text_encoder_1.training if text_encoder_1 is not None else None,
            "text_encoder_2": text_encoder_2.training if text_encoder_2 is not None else None,
        }
        # 记录原始设备，验证前将训练模型搬到 CPU 释放显存
        module_devices = {
            "transformer": next(transformer.parameters()).device,
            "vae": next(vae.parameters()).device if vae is not None else None,
            "text_encoder_1": next(text_encoder_1.parameters()).device if text_encoder_1 is not None else None,
            "text_encoder_2": next(text_encoder_2.parameters()).device if text_encoder_2 is not None else None,
        }

        try:
            # 导出当前 LoRA/MoE 权重到临时文件（fp16），使用 peft 提供的 state_dict，确保路由器/专家权重完整
            converted = get_peft_model_state_dict(transformer, adapter_name="default")
            converted = {k: v.detach().to("cpu", dtype=target_dtype) for k, v in converted.items()}
            save_file(converted, tmp_lora_path)

            # 暂时将训练态的核心模块搬到 CPU，释放 GPU 显存
            transformer.to("cpu")
            if vae is not None:
                vae.to("cpu")
            if text_encoder_1 is not None:
                text_encoder_1.to("cpu")
            if text_encoder_2 is not None:
                text_encoder_2.to("cpu")
            torch.cuda.empty_cache()

            # 构建独立 pipeline（fp16）并应用混合 offload
            from diffusers import FluxKontextPipeline

            pipeline_off = FluxKontextPipeline.from_pretrained(
                self.config.model_path,
                torch_dtype=target_dtype,
            )

            # 构造与训练一致的 LoRA/MoE 配置，先挂适配器再塞权重
            module_modes = self.config.lora_module_modes or {}
            variant_cfg = self.config.lora_variant_config or {}
            auto_variant = "moe" if any(m == "moe" for m in module_modes.values()) else "standard"
            variant = self.config.lora_variant or "auto"
            if variant == "auto":
                variant = auto_variant

            adapter_name = "default"  # 与训练阶段保持一致的适配器名，避免权重前缀不匹配

            num_experts = variant_cfg.get("num_experts", 6)
            expert_rank = variant_cfg.get("expert_rank", 32)
            expert_alpha = variant_cfg.get("expert_alpha", 32)
            top_k = variant_cfg.get("top_k", 1)
            blc_alpha = variant_cfg.get("blc_alpha", 0.0)
            blc_weight = variant_cfg.get("blc_weight", 0.0)

            # 与训练时默认一致的 target_modules
            if variant == "moe":
                target_modules = [
                    "attn.to_k",
                    "attn.to_q",
                    "attn.to_v",
                    "attn.to_out.0",
                    "ff.net.0.proj",
                    "ff.net.2",
                    "proj_out",
                ]
            else:
                target_modules = [
                    "attn.to_k",
                    "attn.to_q",
                    "attn.to_v",
                    "attn.to_out.0",
                ]

            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                init_lora_weights="gaussian",
                target_modules=target_modules,
                module_modes=module_modes,
                num_experts=num_experts,
                expert_rank=expert_rank,
                expert_alpha=expert_alpha,
                top_k=top_k,
                blc_alpha=blc_alpha,
                blc_weight=blc_weight,
            )

            pipeline_off.transformer.add_adapter(lora_config, adapter_name=adapter_name)

            onload_device = torch.device(self.device)
            offload_device = torch.device("cpu")
            num_blocks_per_group = 1  # 与 inference 示例保持一致

            # Transformer block-level
            apply_group_offloading(
                pipeline_off.transformer,
                onload_device=onload_device,
                offload_device=offload_device,
                offload_type="block_level",
                num_blocks_per_group=num_blocks_per_group,
                use_stream=True,
                record_stream=False,
            )
            # Text encoders block-level
            apply_group_offloading(
                pipeline_off.text_encoder,
                onload_device=onload_device,
                offload_device=offload_device,
                offload_type="block_level",
                num_blocks_per_group=num_blocks_per_group,
                use_stream=True,
            )
            apply_group_offloading(
                pipeline_off.text_encoder_2,
                onload_device=onload_device,
                offload_device=offload_device,
                offload_type="block_level",
                num_blocks_per_group=num_blocks_per_group,
                use_stream=True,
            )
            # VAE leaf-level
            apply_group_offloading(
                pipeline_off.vae,
                onload_device=onload_device,
                offload_device=offload_device,
                offload_type="leaf_level",
                use_stream=True,
            )

            # 加载临时 LoRA：直接用 load_state_dict 覆盖 lora_* 参数
            # 使用 peft 的加载逻辑，确保 MoE / adapter 名匹配
            set_peft_model_state_dict(
                pipeline_off.transformer,
                converted,
                adapter_name=adapter_name,
                ignore_mismatched_sizes=False,
                low_cpu_mem_usage=False,
            )
            # MoE-LoRA 需要动态路由，不能 fuse；标准 LoRA 可选择 fuse 以节省开销
            if variant == "moe":
                pipeline_off.transformer.set_adapter(adapter_name)
            else:
                pipeline_off.fuse_lora(adapter_names=[adapter_name])
            pipeline_off.set_progress_bar_config(disable=True)

            # 控制图
            control_image = Image.open(self.config.validation_image).convert("RGB")
            control_image = control_image.resize((width, height))

            with torch.no_grad():
                for idx, prompt in enumerate(self.config.validation_prompts):
                    result = pipeline_off(
                        prompt=prompt,
                        image=control_image,
                        height=height,
                        width=width,
                        num_inference_steps=self.config.num_inference_steps,
                        guidance_scale=self.config.guidance_scale,
                        generator=generator,
                        output_type="pil",
                    ).images[0]

                    save_path = os.path.join(
                        self.config.validation_save_dir,
                        f"epoch_{epoch_idx}_sample_{idx + 1}.png",
                    )
                    result.save(save_path)
                    self.logger.info(f"[验证] Epoch {epoch_idx} 样本已保存：{save_path}")
        except Exception as e:
            self.logger.error(f"[验证] 采样失败：{e}")
        finally:
            # 清理临时 LoRA 与管线，释放显存
            try:
                os.remove(tmp_lora_path)
            except Exception:
                pass
            try:
                del pipeline_off
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            # 恢复训练/评估模式
            if train_modes["transformer"]:
                transformer.train()
            else:
                transformer.eval()
            if train_modes["vae"] is not None:
                vae.train(train_modes["vae"])
            if train_modes["text_encoder_1"] is not None:
                text_encoder_1.train(train_modes["text_encoder_1"])
            if train_modes["text_encoder_2"] is not None:
                text_encoder_2.train(train_modes["text_encoder_2"])

            # 将训练模型搬回原设备
            transformer.to(module_devices["transformer"])
            if vae is not None and module_devices["vae"] is not None:
                vae.to(module_devices["vae"])
            if text_encoder_1 is not None and module_devices["text_encoder_1"] is not None:
                text_encoder_1.to(module_devices["text_encoder_1"])
            if text_encoder_2 is not None and module_devices["text_encoder_2"] is not None:
                text_encoder_2.to(module_devices["text_encoder_2"])
