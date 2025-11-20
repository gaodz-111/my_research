import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import ConcatDataset
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers.schedulers import PNDMScheduler, DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.models.modeling_utils import ModelMixin
import sys
import shutil
import warnings
import random
warnings.filterwarnings("ignore")

# 环境配置

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


parent_dir = os.path.abspath("./llava")  # 当前 llava 所在路径
sys.path.append(parent_dir)

from constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from utils import disable_torch_init
from mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from train_Hypernetwork import HypernetworkForCodebook
# --------------------------
# 1. 配置参数（关键：增加扩散模型训练相关配置）
# --------------------------


#现在有两条路径1.显式路径：直接改 latent（加残差），影响生成风格。2.隐式路径：通过 cross-attention（加风格调制层），让风格调制改变“文本-图像对齐的方式”。可能存在冗余，下一步需要去掉其中某一个


CONFIG = {
    # 模型路径
    "sd_model_path": "/data2/gaodz/stable-diffusion-2-1-base",
    "llava_model_path": "/data2/gaodz/llava-v1.6-vicuna-7b",
    "llava_code_path": "./llava",
    "data_json_path": "/data2/gaodz/Re-Align/hypernet_train_data_short_core.json",
    "image_root": "/data2/gaodz/OmniConsistency",
    "codebook_path": "/data2/gaodz/VQDiffusionVAE/epoch_10.pth",
    "output_dir": "/data2/gaodz/HypernetworkVQ",

    # 训练参数
    "batch_size": 2,  # 扩散模型训练对显存要求高，减小batch_size
    "epochs": 20,
    "learning_rate": 1e-5,  # 扩散模型微调需更低学习率
    "unet_freezing_ratio": 0.5,  # 冻结80%的UNet层，只微调顶层
    "log_interval": 5,
    "save_image_interval": 50,  # 每100个batch保存生成图像

    # 模型维度
    "clip_dim": 1024,
    "llava_dim": 4096,
    "num_heads": 8,
    "max_clip_length": 77,
    "codebook_size": 8192,  # Codebook的码本大小


    "datasets": [
        {
            "data_json_path": "/data2/gaodz/Re-Align/hypernet_train_data_short_core.json",
            "image_root": "/data2/gaodz/OmniConsistency"  # 数据集1的图片根目录
        },
        # {
        #     "data_json_path": "/data2/gaodz/Re-Align/COCO_short_core_1.json",
        #     "image_root": "/data2/gaodz/train2014"  # 数据集2的图片根目录
        # }
    ]

}
os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(os.path.join(CONFIG["output_dir"], "samples"), exist_ok=True)


# --------------------------
# 2. 数据集定义（保持不变，确保输入正确）
# --------------------------
class LLM_CoreTextDataset(Dataset):
    # 关键修改：新增 image_root 参数，代替从CONFIG读取
    def __init__(self, json_path, image_root, clip_tokenizer):
        self.samples = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                required = ["image", "core_text", "long_text", "target_image"]
                if not all(k in sample for k in required):
                    raise ValueError(f"样本缺少字段：{sample.keys()}，需包含{required}")
                self.samples.append(sample)

        self.image_root = image_root  # 每个数据集独立的image_root
        self.clip_tokenizer = clip_tokenizer
        self.image_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载目标图像（期望生成的图像）
        target_img_path = os.path.join(self.image_root, sample["target_image"])
        target_image = Image.open(target_img_path).convert("RGB")
        target_image = self.image_transform(target_image)  # [3,512,512]
        image_path = sample["image"]
        # CLIP核心语义编码输入
        clip_inputs = self.clip_tokenizer(
            sample["core_text"],
            max_length=CONFIG["max_clip_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "image_path": image_path,
            "target_image": target_image,  # 用于计算生成损失
            "core_text": sample["core_text"],
            "long_text": sample["long_text"],
            "clip_input_ids": clip_inputs["input_ids"].squeeze(0),
            "clip_attention_mask": clip_inputs["attention_mask"].squeeze(0),
            "sample_image_root": self.image_root
        }


class VQDiffusionVAE(ModelMixin, nn.Module):
    def __init__(self,
                 vae,  # 传入完整的AutoencoderKL实例
                 codebook_K=8192,
                 beta=0.25,
                 unfreeze_vae_layers=2):
        super().__init__()
        self.vae_latent_channels = 4

        # 保存完整的VAE模型（自带encode()和decode()方法）
        self.vae = vae  # AutoencoderKL实例，有encode()和decode()
        self.config = self.vae.config  # 直接复用VAE的完整配置（包含block_out_channels等）

        # 解冻VAE的关键层（针对内部的encoder和decoder子模块）
        self.unfreeze_vae_layers = unfreeze_vae_layers
        self._unfreeze_vae_part(self.vae.encoder, is_encoder=True)  # 解冻编码器子模块
        self._unfreeze_vae_part(self.vae.decoder, is_encoder=False)  # 解冻解码器子模块
        print(f"✅ 解冻VAE编码器层: {self.unfreeze_vae_layers}, 解码器层: {self.unfreeze_vae_layers}")

        # 初始化Codebook
        self.codebook_K = codebook_K
        self.codebook_D = self.vae_latent_channels
        self.codebook = nn.Embedding(self.codebook_K, self.codebook_D).to(torch.float32)
        nn.init.normal_(self.codebook.weight, mean=0, std=0.1)
        print(f"✅ 初始化Codebook: K={self.codebook_K}, D={self.codebook_D}")
        self.beta = beta
        self.scale_factor = 0.18215
        # 统一移动到设备
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def _unfreeze_vae_part(self, vae_part, is_encoder):
        weight_layers = [l for l in vae_part.modules() if hasattr(l, 'weight')]
        if is_encoder:
            target_layers = weight_layers[-self.unfreeze_vae_layers:] if len(
                weight_layers) >= self.unfreeze_vae_layers else weight_layers
        else:
            target_layers = weight_layers[:self.unfreeze_vae_layers] if len(
                weight_layers) >= self.unfreeze_vae_layers else weight_layers
        for layer in target_layers:
            layer.requires_grad_(True)
            if layer.weight.data.dtype != torch.float32:
                layer.weight.data = layer.weight.data.to(torch.float32)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data = layer.bias.data.to(torch.float32)

    def load_pretrained_codebook(self, codebook_path):
        try:
            checkpoint = torch.load(codebook_path, map_location=self.device)
            if "codebook_state_dict" in checkpoint:
                self.codebook.load_state_dict(checkpoint["codebook_state_dict"])
                print(f"✅ 加载codebook权重: {codebook_path}")
        except Exception as e:
            print(f"❌ 加载权重失败: {str(e)}")

    def quantize(self, z_e):
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, self.codebook_D)
        dist = torch.cdist(z_e_flat, self.codebook.weight, p=2)
        indices = dist.argmin(dim=1)
        indices = torch.clamp(indices, 0, self.codebook_K - 1)

        z_q_flat = self.codebook(indices)
        mask = torch.isnan(z_q_flat) | torch.isinf(z_q_flat)
        if mask.any():
            z_q_flat = torch.where(mask, z_e_flat, z_q_flat)

        commitment_loss = F.mse_loss(z_e_flat, z_q_flat.detach())
        codebook_loss = F.mse_loss(z_q_flat, z_e_flat.detach())
        vq_loss = commitment_loss + self.beta * codebook_loss

        z_q = z_q_flat.view(z_e.shape[0], z_e.shape[2], z_e.shape[3], self.codebook_D).permute(0, 3, 1, 2)
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, indices, z_q, vq_loss

    def encode_and_quantize(self, x):
        """使用AutoencoderKL自带的encode()方法"""
        # 调用完整VAE的encode()方法，返回包含latent_dist的对象
        encoder_output = self.vae.encode(x)  # 正确调用！AutoencoderKL有encode()
        posterior = encoder_output.latent_dist  # 正常获取分布
        z_e = posterior.sample()  # [B,4,H,W]
        z_e = z_e * self.vae.config.scaling_factor  # 使用VAE自身的缩放因子

        z_q_st, indices, z_q, vq_loss = self.quantize(z_e)
        return {
            "z_e": z_e,
            "z_q_st": z_q_st,
            "z_q": z_q,
            "indices": indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3]),
            "vq_loss": vq_loss
        }

    def decode(self, z_q, train_mode=True):
        """使用AutoencoderKL自带的decode()方法"""
        z_q = z_q / self.vae.config.scaling_factor  # 用VAE自身的缩放因子还原
        if train_mode:
            # 调用VAE的decode()方法，return_dict=False时返回(recon_image,)
            x_recon = self.vae.decode(z_q, return_dict=False)[0]
        else:
            with torch.no_grad():
                x_recon = self.vae.decode(z_q, return_dict=False)[0]
        return x_recon

    def forward(self, x):
        quant_results = self.encode_and_quantize(x)
        z_q = quant_results["z_q"]
        vq_loss = quant_results["vq_loss"]

        x_recon = self.decode(z_q, train_mode=True)
        recon_loss = F.mse_loss(x_recon, x)

        return {
            **quant_results,
            "x_recon": x_recon,
            "recon_loss": recon_loss,
            "total_vae_vq_loss": recon_loss + vq_loss
        }


def grad_norm(named_params):
    total = 0.0
    cnt = 0
    for n, p in named_params:
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
            cnt += 1
    return (total ** 0.5) if cnt > 0 else 0.0

# --------------------------
# 3. 双向注意力+Codebook调制器（增强调制能力）
# --------------------------
# class BidirectionalAttention(nn.Module):
#     def __init__(self, clip_dim, llava_dim, num_heads, codebook_size):
#         super().__init__()
#         self.clip_dim = clip_dim
#         self.codebook_size = codebook_size
#
#         # 1. 投影层：将LLaVA特征投影到CLIP维度，增加归一化稳定训练
#         self.llava_proj = nn.Sequential(
#             nn.Linear(llava_dim, clip_dim, dtype=dtype),
#             nn.LayerNorm(clip_dim, dtype=dtype),  # 归一化投影后的特征
#             nn.GELU()  # 引入非线性
#         )
#
#         # 2. 多头注意力层（共享权重用于双向交互）
#         self.multihead_attn = nn.MultiheadAttention(
#             embed_dim=clip_dim,
#             num_heads=num_heads,
#             batch_first=True,
#             dtype=dtype
#         )
#
#         # 3. 特征融合后的归一化层（稳定训练）
#         self.fusion_norm = nn.LayerNorm(clip_dim, dtype=dtype)
#
#         # 4. 增强版Hypernetwork：生成更鲁棒的Codebook调制参数
#         self.codebook_hypernet = nn.Sequential(
#             nn.Linear(clip_dim, clip_dim, dtype=dtype),
#             nn.LayerNorm(clip_dim, dtype=dtype),
#             nn.GELU(),
#             nn.Linear(clip_dim, codebook_size * 2, dtype=dtype)  # 输出mean和logvar
#         )
#
#     def forward(self, clip_emb, llava_emb, clip_attention_mask=None):
#         # 确保输入维度匹配
#         assert clip_emb.shape[-1] == self.clip_dim, f"CLIP特征维度应为{self.clip_dim}，实际为{clip_emb.shape[-1]}"
#
#         # 步骤1：LLaVA特征投影与预处理
#         llava_emb_proj = self.llava_proj(llava_emb)  # [batch, llava_seq, clip_dim]
#
#         # 步骤2：双向注意力融合（带掩码安全处理）
#         # 2.1 CLIP引导LLaVA特征
#         clip_guided_llava, _ = self.multihead_attn(
#             query=llava_emb_proj,
#             key=clip_emb,
#             value=clip_emb,
#             key_padding_mask=~clip_attention_mask.bool() if (clip_attention_mask is not None) else None
#         )  # [batch, llava_seq, clip_dim]
#
#         # 2.2 LLaVA补充CLIP特征（带序列长度截断保护）
#         max_clip_len = CONFIG.get("max_clip_length", clip_emb.shape[1])
#         llava_supplemented_clip, _ = self.multihead_attn(
#             query=clip_emb,
#             key=llava_emb_proj,
#             value=llava_emb_proj
#         )  # [batch, clip_seq, clip_dim]
#
#         # 步骤3：残差融合与归一化（增强特征表达）
#         fused_clip = self.fusion_norm(
#             clip_emb + llava_supplemented_clip[:, :max_clip_len, :]  # 截断到指定长度
#         )  # [batch, max_clip_len, clip_dim]
#
#         # 步骤4：生成Codebook调制参数（mean + logvar）
#         core_feature = fused_clip.mean(dim=1)  # 全局池化：[batch, clip_dim]
#         code_params = self.codebook_hypernet(core_feature)  # [batch, codebook_size*2]
#         code_mean, code_logvar = torch.chunk(code_params, 2, dim=1)  # 拆分mean和logvar
#
#         return {
#             "fused_clip": fused_clip,
#             "code_mean": code_mean,  # [batch, codebook_size]
#             "code_logvar": code_logvar  # [batch, codebook_size]
#         }


class BidirectionalAttention(nn.Module):
    def __init__(self, clip_dim, llava_dim, num_heads, codebook_size):
        super().__init__()
        self.clip_dim = clip_dim
        self.codebook_size = codebook_size

        self.llava_proj = nn.Sequential(
            nn.Linear(llava_dim, clip_dim, dtype=dtype),
            nn.LayerNorm(clip_dim, dtype=dtype),
            nn.GELU()
        )

        # 使用两个独立 attention 层（llava->clip, clip->llava）
        self.attn_llava2clip = nn.MultiheadAttention(embed_dim=clip_dim, num_heads=num_heads, batch_first=True, dtype=dtype)
        self.attn_clip2llava = nn.MultiheadAttention(embed_dim=clip_dim, num_heads=num_heads, batch_first=True, dtype=dtype)

        self.fusion_norm = nn.LayerNorm(clip_dim, dtype=dtype)

        self.codebook_hypernet = nn.Sequential(
            nn.Linear(clip_dim, clip_dim, dtype=dtype),
            nn.LayerNorm(clip_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(clip_dim, codebook_size * 2, dtype=dtype)
        )

    def forward(self, clip_emb, llava_emb, clip_attention_mask=None):
        """
        Inputs:
          clip_emb: [B, Lc, D]
          llava_emb: [B, Ll, D']  (we'll project to D)
        Returns:
          {
            "c1": fused_clip (用于 encoder_hidden_states) [B, Lc, D],
            "c2": clip-guided llava features [B, Ll, D],
            "code_mean": [B, codebook_size],
            "code_logvar": [B, codebook_size]
          }
        """
        assert clip_emb.shape[-1] == self.clip_dim

        # 投影 LLaVA 到 clip 维度
        llava_proj = self.llava_proj(llava_emb)  # [B, Ll, D]

        # 1) llava -> clip: 用 llava 来补充/增强 clip 表征 -> 得到 c1
        llava_to_clip, _ = self.attn_llava2clip(query=clip_emb, key=llava_proj, value=llava_proj)
        # 残差融合
        c1 = self.fusion_norm(clip_emb + llava_to_clip[:, :clip_emb.shape[1], :])  # [B, Lc, D]

        # 2) clip -> llava: 用 clip 来引导 llava -> 得到 c2
        key_pad_mask = None
        if clip_attention_mask is not None:
            # multihead expects key_padding_mask where True indicates positions to be ignored
            key_pad_mask = ~clip_attention_mask.bool()
        clip_to_llava, _ = self.attn_clip2llava(query=llava_proj, key=clip_emb, value=clip_emb, key_padding_mask=key_pad_mask)
        c2 = clip_to_llava  # [B, Ll, D]

        # 3) 从 c2 的全局语义（mean pooling）生成 codebook 参数
        core_feature = c2.mean(dim=1)  # [B, D]
        code_params = self.codebook_hypernet(core_feature)  # [B, codebook_size*2]
        #code_mean, code_logvar跟codebook无关，codebook_hypernet是在该类中初始化的Sequential
        code_mean, code_logvar = torch.chunk(code_params, 2, dim=1)

        return {
            "c1": c1,
            "c2": c2,
            "code_mean": code_mean,
            "code_logvar": code_logvar
        }



# --------------------------
# 4. 扩散模型UNet注入调制信号（核心修改）
# --------------------------
# 替换 ModulatedUNet 类为 FiLM 注入 + bottleneck
class ModulatedUNet(UNet2DConditionModel):
    """
    改造说明：
    - 使用 FiLM (alpha, beta) 注入到 q_proj 的输入（Q构造前）： Q' = (1 + delta_alpha) * Q + beta
    - 先将 code_modulation 投影到 bottleneck_dim，再由 per-layer small linear 映射为 2*in_dim（delta_alpha, beta）
    - 默认 bottleneck_dim=1024（可根据显存调整）
    """
    def __init__(self, unet_config, codebook_size, bottleneck_dim=1024, latent_channels=4):
        super().__init__(**(unet_config if isinstance(unet_config, dict) else unet_config.to_dict()))
        self.codebook_size = codebook_size
        self.bottleneck_dim = bottleneck_dim
        self.latent_channels = latent_channels

        # 全局 bottleneck projector: codebook_size -> bottleneck_dim
        # dtype/device 将在首次注册 q_proj 时用 ref 权重替代（如果没有 q_proj 则使用 float32/cpu）
        self.global_bottleneck = None

        # 原有结构：保存目标 q_proj 列表与 per-layer modulation layers
        self._target_q_projs = []
        self._q_proj_names = []
        self.modulation_layers = nn.ModuleList()  # each: Linear(bottleneck_dim -> 2*in_dim)

        # 扫描 mid_block
        found_any = False
        if hasattr(self, "mid_block") and hasattr(self.mid_block, "attentions"):
            for a_idx, attn_module in enumerate(self.mid_block.attentions):
                tblocks = getattr(attn_module, "transformer_blocks", []) or []
                for j, tb in enumerate(tblocks):
                    attn2 = getattr(tb, "attn2", None)
                    if attn2 is None:
                        continue
                    q_linear = getattr(attn2, "to_q", None) or getattr(attn2, "q_proj", None)
                    if q_linear is None:
                        continue
                    in_dim = q_linear.in_features
                    # We'll create mod_layer (bottleneck_dim -> 2*in_dim) later after knowing dtype/device
                    self._target_q_projs.append(q_linear)
                    self._q_proj_names.append(f"mid_block.attentions[{a_idx}].transformer_blocks[{j}].attn2")
                    found_any = True

        # 扫描最后两个 up_blocks
        n_up = len(self.up_blocks)
        start_idx = max(0, n_up - 2)
        for rel_i, up_block in enumerate(self.up_blocks[start_idx:]):
            i = start_idx + rel_i
            if not hasattr(up_block, "attentions"):
                continue
            for a_idx, attn_module in enumerate(up_block.attentions):
                tblocks = getattr(attn_module, "transformer_blocks", []) or []
                for j, tb in enumerate(tblocks):
                    attn2 = getattr(tb, "attn2", None)
                    if attn2 is None:
                        continue
                    q_linear = getattr(attn2, "to_q", None) or getattr(attn2, "q_proj", None)
                    if q_linear is None:
                        continue
                    in_dim = q_linear.in_features
                    self._target_q_projs.append(q_linear)
                    self._q_proj_names.append(f"up_blocks[{i}].attentions[{a_idx}].transformer_blocks[{j}].attn2")
                    found_any = True

        if not found_any:
            up_block_types = [type(block).__name__ for block in self.up_blocks]
            raise ValueError(
                "未在 mid_block 或最后两个 up_blocks 中找到可调制的 attn2。\n"
                f"当前 up_blocks 类型: {up_block_types}\n"
                "请确认 diffusers 版本和结构，或调整扫描范围。"
            )

        # 使用首个 q_proj 的 weight dtype/device 初始化 global_bottleneck 与 modulation_layers
        ref_w = self._target_q_projs[0].weight if len(self._target_q_projs) > 0 else None
        if ref_w is not None:
            ref_dtype = ref_w.dtype
            ref_device = ref_w.device
        else:
            ref_dtype = torch.float32
            ref_device = torch.device('cpu')

        # global bottleneck projector
        self.global_bottleneck = nn.Linear(self.codebook_size, self.bottleneck_dim, dtype=ref_dtype, device=ref_device)
        # 初始化权重小一些
        nn.init.normal_(self.global_bottleneck.weight, mean=0.0, std=0.01)
        if hasattr(self.global_bottleneck, "bias") and self.global_bottleneck.bias is not None:
            nn.init.zeros_(self.global_bottleneck.bias)

        # 为每个 q_proj 创建 modulation layer: bottleneck_dim -> 2 * in_dim
        for q_proj in self._target_q_projs:
            in_dim = q_proj.in_features
            mod_layer = nn.Linear(self.bottleneck_dim, 2 * in_dim, dtype=ref_dtype, device=ref_device)
            # 权重小幅初始化，bias 初始化为 0（使 delta_alpha 初始为 0, beta 初始为 0）
            nn.init.normal_(mod_layer.weight, mean=0.0, std=0.01)
            if hasattr(mod_layer, "bias") and mod_layer.bias is not None:
                nn.init.zeros_(mod_layer.bias)
            self.modulation_layers.append(mod_layer)

        # 可学习的 per-layer scale（初始小，例如 0.1），帮助控制注入强度
        self.layer_scales = nn.Parameter(torch.ones(len(self.modulation_layers), dtype=ref_dtype, device=ref_device) * 0.1)

    def forward(self, sample, timestep, encoder_hidden_states, code_modulation=None, patch_q_proj=True, **kwargs):
        """
        code_modulation: [B, codebook_size]
        如果 code_modulation 不为 None，会首先投到 bottleneck，再由每层 mod_layer 生成 (delta_alpha, beta)
        最终在 q_proj 前把输入按 FiLM 变换： x' = (1 + delta_alpha) * x + beta
        """
        # 1) 若提供 code_modulation，先计算 bottleneck 表示
        mod_list = None
        if code_modulation is not None:
            # project to bottleneck
            bott = self.global_bottleneck(code_modulation)  # [B, bottleneck_dim]
            # 为每层计算 modulation (list of [B, 2*in_dim])
            mod_list = [m(bott) for m in self.modulation_layers]
            # apply per-layer scale (广播)
            if self.layer_scales is not None:
                # self.layer_scales: [L], mod_list len = L
                mod_list = [mod * self.layer_scales[i].view(1,1) for i, mod in enumerate(mod_list)]

        # 2) 若需要保留 attention-level modulation（通过临时 patch q_proj.forward）
        backups = []
        if code_modulation is not None and patch_q_proj:
            for q_proj_module, mod in zip(self._target_q_projs, mod_list):
                orig_forward = q_proj_module.forward

                # closure binds orig_forward and mod
                def make_film_forward(orig_fwd, modulation_tensor):
                    # modulation_tensor: [B, 2*in_dim]
                    # produce delta_alpha, beta per batch
                    def _film_forward(x):
                        # x: [B, L, in_dim]  (usually L is seq len)
                        # ensure dtype/device match
                        mt = modulation_tensor.to(dtype=x.dtype, device=x.device)
                        B = mt.shape[0]
                        in_dim = mt.shape[1] // 2
                        delta_alpha = mt[:, :in_dim]   # [B, in_dim]
                        beta = mt[:, in_dim:]         # [B, in_dim]
                        # construct alpha = 1 + delta_alpha
                        alpha = (1.0 + delta_alpha).to(dtype=x.dtype, device=x.device)  # [B, in_dim]
                        # apply FiLM: x' = alpha * x + beta
                        # expand to seq dim: alpha.unsqueeze(1) -> [B,1,in_dim], broadcast
                        x_film = alpha.unsqueeze(1) * x + beta.unsqueeze(1)
                        if random.random() < 0.01:
                            print(f"[Diag] q_proj_in.mean={x.mean().item():.4f}, alpha.mean={alpha.mean().item():.4f}, "
                                  f"beta.mean={beta.mean().item():.4f}")
                            print(f"[Diag] q_proj_in_film.mean={x_film.mean().item():.4f}")
                        return orig_fwd(x_film)
                    return _film_forward

                film_fwd = make_film_forward(orig_forward, mod)
                q_proj_module.forward = film_fwd
                backups.append((q_proj_module, orig_forward))

        try:
            out = super().forward(sample, timestep, encoder_hidden_states, **kwargs)
        finally:
            # restore forwards
            for q_proj_module, orig in backups:
                q_proj_module.forward = orig

        return out

def custom_collate_fn(batch):
    """
    batch: list of dict
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence

    # 1. target_image 直接stack（已统一尺寸512x512）
    target_images = torch.stack([item['target_image'] for item in batch], dim=0)

    # 2. image_path / long_text / core_text 保留原始list
    image_paths = [item['image_path'] for item in batch]
    long_texts = [item['long_text'] for item in batch]
    core_texts = [item['core_text'] for item in batch]
    sample_image_roots = [item['sample_image_root'] for item in batch]
    # 3. clip_input_ids / attention_mask pad到batch内最长
    clip_input_ids = pad_sequence(
        [item['clip_input_ids'] for item in batch],
        batch_first=True,
        padding_value=0
    )
    clip_attention_mask = pad_sequence(
        [item['clip_attention_mask'] for item in batch],
        batch_first=True,
        padding_value=0
    )

    return {
        "target_image": target_images,
        "image_path": image_paths,
        "long_text": long_texts,
        "core_text": core_texts,
        "clip_input_ids": clip_input_ids,
        "clip_attention_mask": clip_attention_mask,
        "sample_image_roots": sample_image_roots,
    }


class PerceptualLoss(nn.Module):
    def __init__(self, device, local_vgg_path):
        super().__init__()
        # 1. 构建VGG16模型结构（与预训练模型一致）
        from torchvision.models import vgg16, VGG16_Weights
        self.vgg = vgg16(weights=None).to(device)  # 不加载预训练权重，仅创建结构

        # 2. 加载本地权重文件（确保路径正确）
        # 本地权重文件通常是 `.pth` 格式，例如：./pretrained_models/vgg16/vgg16-397923af.pth
        try:
            state_dict = torch.load(local_vgg_path, map_location=device)
            self.vgg.load_state_dict(state_dict)
            print(f"✅ 成功加载本地VGG16模型：{local_vgg_path}")
        except FileNotFoundError:
            raise ValueError(f"❌ 本地模型文件不存在：{local_vgg_path}")
        except Exception as e:
            raise RuntimeError(f"❌ 加载模型失败：{str(e)}")

        # 3. 取前8层用于特征提取（与之前逻辑一致）
        self.feature_extractor = nn.Sequential(*list(self.vgg.features.children())[:8]).eval()

        # 4. 冻结参数（仅用于特征提取，不训练）
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # 5. VGG输入归一化（保持一致）
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, gen_image, target_image):
        gen_image = (gen_image + 1) / 2  # [-1,1] → [0,1]
        target_image = (target_image + 1) / 2
        gen_feat = self.feature_extractor(self.normalize(gen_image))
        target_feat = self.feature_extractor(self.normalize(target_image))
        return nn.functional.l1_loss(gen_feat, target_feat)

# --------------------------
# 5. 加载模型组件（含扩散模型微调配置）
# --------------------------
# def load_models():
#     # 加载Stable Diffusion核心组件
#     vae = AutoencoderKL.from_pretrained(
#         CONFIG["sd_model_path"], subfolder="vae", torch_dtype=dtype
#     ).to(device)
#     unet = UNet2DConditionModel.from_pretrained(
#         CONFIG["sd_model_path"], subfolder="unet", torch_dtype=dtype
#     ).to(device)
#     # 包装UNet以支持Code调制
#     modulated_unet = ModulatedUNet(unet.config, CONFIG["codebook_size"]).to(device, dtype=dtype)
#     modulated_unet.load_state_dict(unet.state_dict(), strict=False)
#
#     # 加载CLIP（从SD中提取）
#     pipe = StableDiffusionPipeline.from_pretrained(
#         CONFIG["sd_model_path"], vae=vae, unet=unet, torch_dtype=dtype, safety_checker=None
#     ).to(device)
#     clip_encoder = pipe.text_encoder
#     clip_tokenizer = pipe.tokenizer
#     for param in clip_encoder.parameters():
#         param.requires_grad = False
#
#     # 加载LLaVA
#     sys.path.append(os.path.abspath(CONFIG["llava_code_path"]))
#     model_name = get_model_name_from_path(CONFIG["llava_model_path"])
#     llava_tokenizer, llava_model, image_processor, _  = load_pretrained_model(
#         CONFIG["llava_model_path"], None, model_name, device=device, torch_dtype=torch.float32
#     )
#     llava_model = llava_model.to(device, dtype=torch.float32)
#     for param in llava_model.parameters():
#         param.requires_grad = False
#
#     # --------------------------
#     # 关键修改1：优化UNet冻结策略（仅冻结底层，保留细节层）
#     # --------------------------
#     freeze_count = 0
#     trainable_count = 0
#     for name, param in modulated_unet.named_parameters():
#         # 仅冻结down_blocks的前2层（底层：提取通用边缘/纹理，与细节生成关联弱）
#         if "down_blocks.0" in name or "down_blocks.1" in name:
#             param.requires_grad = False
#             freeze_count += 1
#         else:
#             # 保留：up_blocks（细节生成）、mid_block（核心结构）、交叉注意力（文本-细节对齐）、调制层
#             param.requires_grad = True
#             trainable_count += 1
#     print(f"UNet冻结层数量：{freeze_count}（仅down_blocks.0/1）")
#     print(f"UNet可训练层数量：{trainable_count}（含细节生成与文本对齐层）")
#
#     return {
#         "vae": vae,
#         "unet": modulated_unet,
#         "clip_encoder": clip_encoder,
#         "clip_tokenizer": clip_tokenizer,
#         "llava_tokenizer": llava_tokenizer,
#         "image_processor": image_processor,
#         "llava_model": llava_model,
#         "scheduler": DDIMScheduler.from_pretrained(
#             CONFIG["sd_model_path"],
#             subfolder="scheduler",
#             beta_start=0.00085,  # SD 2.1 默认参数，确保与模型匹配
#             beta_end=0.012,
#             beta_schedule="scaled_linear"
#         )
#     }
class ResidualGenerator(nn.Module):
    def __init__(self, clip_dim, target_channels=256, target_size=(64, 64)):
        super().__init__()
        self.target_size = target_size
        self.target_channels = target_channels
        # c2全局特征提取→映射到小尺寸特征图
        self.global_feat = nn.Sequential(
            nn.Linear(clip_dim, clip_dim // 2, dtype=dtype),
            nn.LayerNorm(clip_dim // 2, dtype=dtype),
            nn.GELU(),
            nn.Linear(clip_dim // 2, target_channels * 4 * 4, dtype=dtype)  # [B, C*4*4]
        )
        # 上采样到目标尺寸（64x64）
        self.upsample = nn.Sequential(
            #
            nn.ConvTranspose2d(target_channels, target_channels, kernel_size=4, stride=2, padding=1, dtype=dtype),
            nn.GroupNorm(num_groups=4, num_channels=target_channels, dtype=dtype),  # 分组数=4
            nn.GELU(),
            #
            nn.ConvTranspose2d(target_channels, target_channels, kernel_size=4, stride=2, padding=1, dtype=dtype),
            nn.GroupNorm(num_groups=4, num_channels=target_channels, dtype=dtype),  # 分组数=4
            nn.GELU(),


        )

    def forward(self, c2):
        # c2: [B, Ll, clip_dim] → 全局平均池化
        c2_global = c2.mean(dim=1)  # [B, clip_dim]
        # 映射为特征图
        feat_flat = self.global_feat(c2_global)  # [B, C*4*4]
        feat_map = feat_flat.view(-1, self.target_channels, 4, 4)  # [B, C,4,4]
        # 上采样
        residual = self.upsample(feat_map)  # [B, C,64,64]
        # 适配目标尺寸
        if residual.shape[-2:] != self.target_size:
            residual = F.interpolate(residual, self.target_size, mode="bilinear", align_corners=False)

        return residual

def load_models():
    # --------------------------
    # 新增：加载VQ-VAE与Codebook（替换原SD VAE编码）
    # --------------------------
    # 初始化VQDiffusionVAE
    original_vae = AutoencoderKL.from_pretrained(
        CONFIG["sd_model_path"],
        subfolder="vae",
        torch_dtype=torch.float32
    )


    # 初始化修改后的VQDiffusionVAE
    vq_vae = VQDiffusionVAE(
        vae=original_vae,
        codebook_K=8192,
        beta=0.25,
        unfreeze_vae_layers=2
    ).to(device, dtype=dtype)


    # 加载预训练codebook权重（使用你的load_trained_model核心逻辑）
    vq_vae.load_pretrained_codebook(CONFIG["codebook_path"])
    print(f"✅ 成功加载Codebook权重：{CONFIG['codebook_path']}")

    # 冻结VQ-VAE编码器和Codebook（仅训练残差生成器和UNet，若需微调可设为True）
    # for param in vq_vae.diffusion_vae.encoder.parameters():
    #     param.requires_grad = False
    # for param in vq_vae.codebook.parameters():
    #     param.requires_grad = False
    #
    # # --------------------------
    # # 保留SD的VAE解码器（用于最终图像生成）
    # # --------------------------
    # sd_vae_decoder = AutoencoderKL.from_pretrained(
    #     CONFIG["sd_model_path"], subfolder="vae", torch_dtype=dtype
    # ).to(device)
    # 仅保留解码器，冻结所有参数
    # for param in sd_vae_decoder.parameters():
    #     param.requires_grad = False
    # vq_vae.diffusion_vae = sd_vae_decoder  # 将SD解码器接入VQ-VAE

    # --------------------------
    # 加载UNet（原逻辑不变，包装为ModulatedUNet）
    # --------------------------
    unet = UNet2DConditionModel.from_pretrained(
        CONFIG["sd_model_path"], subfolder="unet", torch_dtype=dtype
    ).to(device)
    modulated_unet = ModulatedUNet(unet.config, CONFIG["codebook_size"]).to(device, dtype=dtype)
    modulated_unet.load_state_dict(unet.state_dict(), strict=False)

    # --------------------------
    # 加载CLIP、LLaVA（原逻辑不变）
    # --------------------------
    # 加载CLIP
    pipe = StableDiffusionPipeline.from_pretrained(
        CONFIG["sd_model_path"], vae=vq_vae, unet=unet, torch_dtype=dtype, safety_checker=None
    ).to(device)
    clip_encoder = pipe.text_encoder
    clip_tokenizer = pipe.tokenizer
    for param in clip_encoder.parameters():
        param.requires_grad = False

    # 加载LLaVA
    sys.path.append(os.path.abspath(CONFIG["llava_code_path"]))
    model_name = get_model_name_from_path(CONFIG["llava_model_path"])
    llava_tokenizer, llava_model, image_processor, _ = load_pretrained_model(
        CONFIG["llava_model_path"], None, model_name, device=device, torch_dtype=torch.float32
    )
    llava_model = llava_model.to(device, dtype=torch.float32)
    for param in llava_model.parameters():
        param.requires_grad = False

    # --------------------------
    # UNet冻结策略（原逻辑不变）
    # --------------------------
    freeze_count = 0
    trainable_count = 0
    for name, param in modulated_unet.named_parameters():
        if "down_blocks.0" in name or "down_blocks.1" in name:
            param.requires_grad = False
            freeze_count += 1
        else:
            param.requires_grad = True
            trainable_count += 1
    print(f"UNet冻结层数量：{freeze_count}（仅down_blocks.0/1）")
    print(f"UNet可训练层数量：{trainable_count}（含细节生成与文本对齐层）")

    # --------------------------
    # 新增：初始化ResidualGenerator（c2生成残差叠加到量化latent）
    # --------------------------


    # 初始化残差生成器（可训练）
    # residual_generator = ResidualGenerator(
    #     clip_dim=CONFIG["clip_dim"],
    #     target_channels=vq_vae.codebook_D,  # 关键：使用codebook_D=4，与VAE潜变量通道一致
    #     target_size=(64, 64)  # 关键：VQ-VAE输出分辨率是32x32（256/8=32）
    # ).to(device, dtype=dtype)

    return {
        "vq_vae": vq_vae,  # 新增：VQ-VAE（含codebook）

        "unet": modulated_unet,
        "clip_encoder": clip_encoder,
        "clip_tokenizer": clip_tokenizer,
        "llava_tokenizer": llava_tokenizer,
        "image_processor": image_processor,
        "llava_model": llava_model,
        "scheduler": DDIMScheduler.from_pretrained(
            CONFIG["sd_model_path"],
            subfolder="scheduler",
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear"
        ),
        # "residual_generator": residual_generator  # 新增：残差生成器
    }



def save_generated_samples(
        unet, vae, scheduler, fusion_outputs, code_modulation, texts,
        batch_idx, epoch, output_dir, device, dtype,
        restore_train_mode=True  # 新增参数：是否恢复train模式（默认True，兼容训练）
):
    """
    texts: list of str, 每个文本生成一张图
    restore_train_mode: bool, 推理时设为False，避免切回train模式
    """
    # 保存UNet原始模式（防止函数外的模式被污染）
    original_mode = unet.training  # True=train模式，False=eval模式
    unet.eval()  # 强制进入推理模式，关闭随机层

    with torch.no_grad():
        scheduler.set_timesteps(num_inference_steps=50)

        for i, text_emb in enumerate(fusion_outputs["fused_clip"]):
            # latent 初始化（推理时无需缩放，纯随机噪声）
            latents = torch.randn((1, 4, 64, 64), device=device, dtype=dtype)

            # code_modulation 对应每个文本，确保dtype/device一致
            cm = code_modulation[i:i + 1].to(dtype=latents.dtype, device=latents.device)
            encoder_hidden_states = text_emb.unsqueeze(0).to(dtype=latents.dtype, device=latents.device)

            # 逐步去噪（与扩散模型推理逻辑一致）
            for t in scheduler.timesteps:
                model_pred = unet(
                    latents, t,
                    encoder_hidden_states=encoder_hidden_states,
                    code_modulation=cm
                ).sample
                latents = scheduler.step(model_output=model_pred, timestep=t, sample=latents).prev_sample

            # VAE解码（SD标准流程：除以0.18215，再归一化到0-1）
            image = vae.decode(latents, train_mode=False)# 关键修改
            image = (image / 2 + 0.5).clamp(0, 1).squeeze().permute(1, 2, 0).cpu().numpy()

            # 优化文件名：推理时避免显示无意义的epoch/batch（可选）
            if restore_train_mode:
                # 训练时：保留epoch/batch信息
                save_name = f"epoch{epoch + 1}_text{i + 1}.png"
            else:
                # 推理时：用文本索引+时间命名，更清晰
                import time
                timestamp = int(time.time())
                save_name = f"inference_text{i + 1}_{timestamp}.png"

            # 保存图片（确保目录存在）
            save_dir = os.path.join(output_dir, "samples")
            os.makedirs(save_dir, exist_ok=True)
            Image.fromarray((image * 255).astype("uint8")).save(
                os.path.join(save_dir, save_name)
            )

    # 仅在训练场景下恢复原始模式（推理时跳过）
    if restore_train_mode:
        unet.train(mode=original_mode)


# --------------------------
# 6. 训练主函数（关联扩散模型生成损失）
# --------------------------
# def main():
#     # 加载所有模型组件
#     models = load_models()
#     vae = models["vae"]
#     unet = models["unet"]
#     clip_encoder = models["clip_encoder"]
#     clip_tokenizer = models["clip_tokenizer"]
#     llava_tokenizer = models["llava_tokenizer"]
#     image_processor = models["image_processor"]
#     llava_model = models["llava_model"]
#     scheduler = models["scheduler"]
#
#     # 加载数据集
#     all_datasets = []  # 存储所有数据集实例
#
#     for dataset_info in CONFIG["datasets"]:
#         # 为每个数据集创建独立的加载器
#         dataset = LLM_CoreTextDataset(
#             json_path=dataset_info["data_json_path"],
#             image_root=dataset_info["image_root"],  # 传入当前数据集的image_root
#             clip_tokenizer=clip_tokenizer
#         )
#         all_datasets.append(dataset)
#         print(f"加载数据集：{dataset_info['data_json_path']}，样本数：{len(dataset)}")
#
#     # 合并所有数据集为一个整体
#     combined_dataset = ConcatDataset(all_datasets)
#     print(f"所有数据集合并完成，总样本数：{len(combined_dataset)}")
#     dataloader = DataLoader(
#         combined_dataset,
#         batch_size=CONFIG["batch_size"],
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True,
#         collate_fn=custom_collate_fn
#     )
#     print(f"DataLoader创建完成，总batch数：{len(dataloader)}")
#
#     # 初始化双向注意力模型
#     bidirectional_attn = BidirectionalAttention(
#         clip_dim=CONFIG["clip_dim"],
#         llava_dim=CONFIG["llava_dim"],
#         num_heads=CONFIG["num_heads"],
#         codebook_size=CONFIG["codebook_size"]
#     ).to(device, dtype=dtype)
#
#     # --------------------------
#     # 关键修改2：初始化感知损失
#     # --------------------------
#     local_vgg_path = "/data2/gaodz/vgg16/vgg16-397923af.pth"
#     perceptual_loss_fn = PerceptualLoss(device=device, local_vgg_path=local_vgg_path).to(dtype=dtype)
#
#     # 优化器（保持不变）
#     params_to_train = list(bidirectional_attn.parameters()) + list(unet.parameters())
#     optimizer = optim.AdamW(params_to_train, lr=CONFIG["learning_rate"], weight_decay=1e-5)
#
#     # 学习率调度器（保持不变）
#     lr_scheduler = get_scheduler(
#         "cosine",
#         optimizer=optimizer,
#         num_warmup_steps=500,
#         num_training_steps=len(dataloader) * CONFIG["epochs"]
#     )
#
#     # 训练循环
#     for epoch in range(CONFIG["epochs"]):
#         unet.train()
#         bidirectional_attn.train()
#         total_loss = 0.0
#         scheduler.set_timesteps(num_inference_steps=1000, device=device)
#         for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
#             # 解析输入（保持不变）
#             target_images = batch["target_image"].to(device, dtype=dtype)
#             img_paths = batch["image_path"]
#             clip_input_ids = batch["clip_input_ids"].to(device)
#             clip_attention_mask = batch["clip_attention_mask"].to(device)
#             long_texts = batch["long_text"]
#             core_texts = batch["core_text"]
#             sample_image_roots = batch["sample_image_roots"]
#             model_dtype = llava_model.dtype
#
#             # 1. 生成CLIP和LLaVA编码（保持不变）
#             with torch.no_grad():
#                 clip_emb = clip_encoder(input_ids=clip_input_ids, attention_mask=clip_attention_mask).last_hidden_state
#
#                 llava_prompts = [f"<image>\n{t}" for t in long_texts]
#                 processed_imgs = []
#                 for img_path, img_root in zip(img_paths, sample_image_roots):
#
#                     img = Image.open(os.path.join(img_root, img_path)).convert("RGB")
#                     processed_img = image_processor(img, return_tensors="pt")["pixel_values"]
#                     processed_imgs.append(processed_img.to(dtype=model_dtype))
#                 processed_imgs = torch.stack(processed_imgs).to(device)
#
#                 input_text = []
#                 for long_txt in long_texts:
#                     temp_inputs = tokenizer_image_token(long_txt, llava_tokenizer, IMAGE_TOKEN_INDEX,
#                                                         return_tensors='pt').squeeze(0)
#                     input_text.append(temp_inputs)
#                 llava_inputs = pad_sequence(
#                     input_text,
#                     batch_first=True,
#                     padding_value=llava_tokenizer.pad_token_id
#                 ).to(device)
#
#                 llava_outputs = llava_model(llava_inputs, images=processed_imgs, output_hidden_states=True)
#                 llava_emb = llava_outputs.hidden_states[-1].mean(dim=1, keepdim=True)
#             llava_emb = llava_emb.to(dtype=dtype)
#
#             # 2. 双向注意力融合（保持不变）
#             fusion_outputs = bidirectional_attn(
#                 clip_emb=clip_emb,
#                 llava_emb=llava_emb,
#                 clip_attention_mask=clip_attention_mask
#             )
#             code_mean = fusion_outputs["code_mean"]
#             code_logvar = fusion_outputs["code_logvar"]
#             code_modulation = code_mean + torch.exp(0.5 * code_logvar) * torch.randn_like(code_logvar)
#
#             # 3. 扩散模型生成过程（保持不变）
#             with torch.no_grad():
#                 target_latents = vae.encode(target_images).latent_dist.sample() * 0.18215  # SD VAE 缩放因子
#
#             # 3.1 加噪声（DDIM 支持批量时间步，无需额外初始化）
#             noise = torch.randn_like(target_latents)
#             bsz = target_latents.shape[0]
#             # 从 DDIM 调度器的训练时间步范围中随机选择（直接用 scheduler.num_train_timesteps，无需额外 set_timesteps）
#             T = scheduler.config.num_train_timesteps
#             # beta 分布采样 [0,1]
#             u = torch.distributions.Beta(2.0, 2.0).sample((bsz,)).to(device)
#             timesteps = (u * (T - 1)).long()
#             noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)
#
#             # 3.2 类型对齐（保持不变）
#             noisy_latents = noisy_latents.to(dtype=unet.dtype)  # 显式赋值（原代码未赋值，可能导致 dtype 不匹配）
#             code_modulation = code_modulation.to(dtype=unet.dtype)
#
#             # 3.3 UNet 预测噪声（保持不变）
#             model_pred = unet(
#                 noisy_latents,
#                 timesteps,
#                 encoder_hidden_states=fusion_outputs["fused_clip"],
#                 code_modulation=code_modulation
#             ).sample
#
#             # --------------------------
#             # 关键修改：适配 DDIM 的感知损失计算（支持批量时间步）
#             # --------------------------
#             # 1. 扩散噪声损失（L1，保持不变）
#             diffusion_loss = nn.functional.mse_loss(model_pred, noise)
#
#             # 2. 感知损失（DDIM 无需提前 set_timesteps，直接用批量 timesteps）
#             # with torch.no_grad():
#             #     pred_latents = []  # 存储每个样本的去噪结果
#             #     # 循环处理批量中的每个样本
#             #     for i in range(bsz):
#             #         # 对单个样本调用scheduler.step（timestep为标量）
#             #         pred_latent = scheduler.step(
#             #             model_output=model_pred[i:i + 1],  # 取第i个样本的预测噪声（保持batch维度）
#             #             timestep=timesteps[i].item(),  # 取标量时间步（关键修改）
#             #             sample=noisy_latents[i:i + 1]  # 取第i个样本的噪声潜变量
#             #         ).prev_sample
#             #         pred_latents.append(pred_latent)
#             #     # 拼接所有样本的结果，恢复批量维度 [batch_size, 4, 64, 64]
#             #     pred_latents = torch.cat(pred_latents, dim=0)
#             #
#             # # 解码生成图像（保持不变）
#             # gen_image = vae.decode(pred_latents / 0.18215).sample  # [batch, 3, 512, 512]
#             # perceptual_loss = perceptual_loss_fn(gen_image, target_images) * 0.1  # 权重保持 0.1
#
#             alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
#             if alphas_cumprod is None:
#                 # 保险回退：尝试从 scheduler.config 中构造（大多数情况下 alphas_cumprod 可用）
#                 raise RuntimeError("scheduler 缺少 alphas_cumprod，无法计算 pred_x0，请确认使用的是 diffusers 的调度器。")
#
#             # 确保 alphas_cumprod 在 GPU 上并且 dtype 与 unet 相同
#             alphas_cumprod = alphas_cumprod.to(device=device, dtype=unet.dtype)
#
#             # timesteps 是你之前随机采样的索引（0..len-1），直接索引 alphas_cumprod
#             alpha_t = alphas_cumprod[timesteps].view(bsz, 1, 1, 1)  # [B,1,1,1]
#             sqrt_alpha_t = alpha_t.sqrt()
#             sqrt_one_minus_alpha = (1.0 - alpha_t).sqrt()
#
#             # model_pred 是预测 noise（eps）
#             # noisy_latents 是 x_t
#             # pred_x0 = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
#             pred_x0 = (noisy_latents - sqrt_one_minus_alpha * model_pred) / (sqrt_alpha_t + 1e-8)
#
#             # decode pred_x0（注意 VAE 缩放因子）
#             gen_image = vae.decode(pred_x0 / 0.18215).sample  # -> [B,3,512,512]
#             # perceptual 权重提高到 0.5（你可以再试 0.5 ~ 1.0）
#             T = scheduler.config.num_train_timesteps
#             # timesteps 可能是索引 0..T-1
#             t_norm = timesteps.to(dtype=unet.dtype) / float(T - 1)  # [0,1]
#             # 你可以调 base 权重（目前你用了 0.5）
#             base_perceptual_weight = 0.3
#             perceptual_weights = base_perceptual_weight * (1.0 - t_norm).view(bsz, 1, 1, 1)  # broadcast
#             # 如果你在前面按-batch算出的 scalar 感知损失 per sample，可以乘以对应 scalar
#             # 因为 perceptual_loss_fn 返回 batch-averaged scalar，目前为简便，使用 batch mean * mean(weights)
#             perceptual_loss = perceptual_loss_fn(gen_image, target_images) * perceptual_weights.mean()
#
#             # 3. KL 损失（保持不变）
#             kl_loss = -0.5 * torch.mean(1 + code_logvar - code_mean ** 2 - code_logvar.exp())
#             kl_weight = 0.5
#
#             # 总损失（保持不变）
#             total_batch_loss = diffusion_loss + perceptual_loss + kl_weight * kl_loss
#
#             # --------------------------
#             # 关键修改4：添加梯度裁剪（稳定训练，避免细节学习混乱）
#             # --------------------------
#             optimizer.zero_grad()
#             total_batch_loss.backward()
#             torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)  # 限制梯度范数≤1.0
#             optimizer.step()
#             lr_scheduler.step()
#
#             # 6. 日志与样本保存（保持不变）
#             total_loss += total_batch_loss.item()
#             avg_loss = total_loss / (batch_idx + 1)
#             if (batch_idx + 1) % CONFIG["log_interval"] == 0:
#                 print(
#                     f"Batch {batch_idx + 1} | avg_loss: {avg_loss:.6f} "
#                     f"(diffusion: {diffusion_loss.item():.6f}, perceptual: {perceptual_loss.item():.6f}, KL: {kl_loss.item():.6f})"
#                 )
#
#             if (batch_idx + 1) % CONFIG["save_image_interval"] == 0:
#                 samples_dir = os.path.join(CONFIG["output_dir"], "samples")
#                 if os.path.exists(samples_dir):
#                     shutil.rmtree(samples_dir)
#                 os.makedirs(samples_dir, exist_ok=True)
#                 save_generated_samples(
#                     unet, vae, scheduler,
#                     fusion_outputs=fusion_outputs,
#                     code_modulation=code_modulation,
#                     texts=core_texts,
#                     batch_idx=batch_idx,
#                     epoch=epoch,
#                     output_dir=CONFIG["output_dir"],
#                     device=device,
#                     dtype=dtype,
#                     restore_train_mode=True
#                 )
#
#
#
#
#
#         if (epoch + 1) % 2 == 0:
#             save_path = os.path.join(CONFIG["output_dir"], f"model_epoch_{epoch + 1}.pth")
#             torch.save({
#                 "unet_state_dict": unet.state_dict(),
#                 "attn_state_dict": bidirectional_attn.state_dict(),
#             }, save_path)
#             print(f"Epoch {epoch + 1} 模型保存至：{save_path}")
#
#     print("训练完成！")


def main():
    # 加载所有模型组件（含VQ-VAE和ResidualGenerator）
    models = load_models()
    vq_vae = models["vq_vae"]  # 已修改：允许VAE协同训练
    # 关键：移除单独的sd_vae_decoder，复用vq_vae.diffusion_vae（避免冗余）
    unet = models["unet"]
    clip_encoder = models["clip_encoder"]
    clip_tokenizer = models["clip_tokenizer"]
    llava_tokenizer = models["llava_tokenizer"]
    image_processor = models["image_processor"]
    llava_model = models["llava_model"]
    scheduler = models["scheduler"]
    # residual_generator = models["residual_generator"]  # 新增

    # 加载数据集（原逻辑不变）
    all_datasets = []
    for dataset_info in CONFIG["datasets"]:
        dataset = LLM_CoreTextDataset(
            json_path=dataset_info["data_json_path"],
            image_root=dataset_info["image_root"],
            clip_tokenizer=clip_tokenizer
        )
        all_datasets.append(dataset)
        print(f"加载数据集：{dataset_info['data_json_path']}，样本数：{len(dataset)}")
    combined_dataset = ConcatDataset(all_datasets)
    print(f"所有数据集合并完成，总样本数：{len(combined_dataset)}")
    dataloader = DataLoader(
        combined_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    print(f"DataLoader创建完成，总batch数：{len(dataloader)}")

    # 初始化双向注意力模型（原逻辑不变）
    bidirectional_attn = BidirectionalAttention(
        clip_dim=CONFIG["clip_dim"],
        llava_dim=CONFIG["llava_dim"],
        num_heads=CONFIG["num_heads"],
        codebook_size=CONFIG["codebook_size"]
    ).to(device, dtype=dtype)

    # 初始化感知损失（原逻辑不变）
    local_vgg_path = "/data2/gaodz/vgg16/vgg16-397923af.pth"
    perceptual_loss_fn = PerceptualLoss(device=device, local_vgg_path=local_vgg_path).to(dtype=dtype)

    # --------------------------
    # 关键修改1：优化器加入VAE可训练参数（解冻层+codebook）
    # --------------------------
    # 收集所有可训练参数：双向注意力 + UNet + 残差生成器 + VAE可训练层 + codebook
    params_to_train = []
    # 1. 双向注意力
    params_to_train.extend(list(bidirectional_attn.parameters()))
    # 2. UNet
    params_to_train.extend(list(unet.parameters()))
    # 3. 残差生成器
    # params_to_train.extend(list(residual_generator.parameters()))
    # 4. VAE可训练参数（解冻的编码器/解码器层 + codebook）
    params_to_train.extend([p for p in vq_vae.parameters() if p.requires_grad])

    params_to_train.extend(list(vq_vae.codebook.parameters()))
    # 去重（避免重复添加）
    params_to_train = list(dict.fromkeys(params_to_train))

    # 初始化优化器（建议降低学习率，避免多参数训练不稳定）
    optimizer = optim.AdamW(
        params_to_train,
        lr=CONFIG["learning_rate"],  # 关键：学习率减半（原1e-5→5e-6）
        weight_decay=1e-5
    )

    # 学习率调度器（原逻辑不变）
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * CONFIG["epochs"]
    )

    # 训练循环（核心修改：VAE协同训练）
    for epoch in range(CONFIG["epochs"]):
        # 关键：所有可训练模块设为train模式（含VAE）
        unet.train()
        bidirectional_attn.train()
        # residual_generator.train()
        vq_vae.train()  # VAE设为训练模式（解冻层生效）
        total_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
            # 解析输入（原逻辑不变）
            target_images = batch["target_image"].to(device, dtype=dtype)  # [B,3,512,512]（[-1,1]）
            img_paths = batch["image_path"]
            clip_input_ids = batch["clip_input_ids"].to(device)
            clip_attention_mask = batch["clip_attention_mask"].to(device)
            long_texts = batch["long_text"]
            core_texts = batch["core_text"]
            sample_image_roots = batch["sample_image_roots"]
            model_dtype = llava_model.dtype

            # 1. 生成CLIP和LLaVA编码（原逻辑不变）
            with torch.no_grad():
                clip_emb = clip_encoder(input_ids=clip_input_ids, attention_mask=clip_attention_mask).last_hidden_state
                # LLaVA编码（原逻辑不变）
                llava_prompts = [f"<image>\n{t}" for t in long_texts]
                processed_imgs = []
                for img_path, img_root in zip(img_paths, sample_image_roots):
                    img = Image.open(os.path.join(img_root, img_path)).convert("RGB")
                    processed_img = image_processor(img, return_tensors="pt")["pixel_values"]
                    processed_imgs.append(processed_img.to(dtype=model_dtype))
                processed_imgs = torch.stack(processed_imgs).to(device)
                # LLaVA输入处理
                input_text = []
                for long_txt in long_texts:
                    temp_inputs = tokenizer_image_token(long_txt, llava_tokenizer, IMAGE_TOKEN_INDEX,
                                                        return_tensors='pt').squeeze(0)
                    input_text.append(temp_inputs)
                llava_inputs = pad_sequence(
                    input_text, batch_first=True, padding_value=llava_tokenizer.pad_token_id
                ).to(device)
                llava_outputs = llava_model(llava_inputs, images=processed_imgs, output_hidden_states=True)
                llava_emb = llava_outputs.hidden_states[-1].mean(dim=1, keepdim=True)  # [B,1,llava_dim]
            llava_emb = llava_emb.to(dtype=dtype)

            # 2. 双向注意力融合（原逻辑不变，新增c2输出）
            fusion_outputs = bidirectional_attn(
                clip_emb=clip_emb,
                llava_emb=llava_emb,
                clip_attention_mask=clip_attention_mask
            )
            c1 = fusion_outputs["c1"]  # UNet文本条件
            c2 = fusion_outputs["c2"]  # 生成残差用
            code_mean = fusion_outputs["code_mean"]
            code_logvar = fusion_outputs["code_logvar"]
            code_modulation = code_mean + torch.exp(0.5 * code_logvar) * torch.randn_like(code_logvar)

            # --------------------------
            # 核心修改2：VAE协同训练（移除no_grad()，保留梯度）
            # --------------------------
            # 不使用with torch.no_grad()，让VAE解冻层计算梯度
            quant_results = vq_vae(target_images)
            quantized_latent = quant_results["z_q_st"]  # [B,4,64,64]（带梯度）
            vq_loss = quant_results["vq_loss"]  # VQ量化损失
            vae_recon_loss = quant_results["recon_loss"]  # 新增：VAE重建损失（图像重建误差）

            # 缩放量化latent（匹配UNet输入范围）
            quantized_latent = quantized_latent * vq_vae.scale_factor

            # c2生成残差并叠加（尺寸需匹配，此处假设64x64）
            # residual = residual_generator(c2)
            # # 尺寸验证（避免叠加错误）
            #
            # residual = residual * vq_vae.scale_factor
            # augmented_latent = quantized_latent + residual
            augmented_latent = quantized_latent
            # 4. 扩散模型生成过程（原逻辑不变）
            noise = torch.randn_like(augmented_latent)
            bsz = augmented_latent.shape[0]
            T = scheduler.config.num_train_timesteps
            u = torch.distributions.Beta(2.0, 2.0).sample((bsz,)).to(device)
            timesteps = (u * (T - 1)).long()
            noisy_latents = scheduler.add_noise(augmented_latent, noise, timesteps)

            # 类型对齐（原逻辑不变）
            noisy_latents = noisy_latents.to(dtype=unet.dtype)
            code_modulation = code_modulation.to(dtype=unet.dtype)

            # 5. UNet预测噪声（原逻辑不变）
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=c1,
                code_modulation=code_modulation
            ).sample

            # 6. 损失计算（核心修改3：加入VAE重建损失）
            # 6.1 扩散噪声损失
            diffusion_loss = nn.functional.mse_loss(model_pred, noise)

            # 6.2 感知损失（复用VQ-VAE的diffusion_vae解码，避免冗余）
            alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
            if alphas_cumprod is None:
                raise RuntimeError("scheduler 缺少 alphas_cumprod，无法计算 pred_x0")
            alphas_cumprod = alphas_cumprod.to(device=device, dtype=unet.dtype)
            alpha_t = alphas_cumprod[timesteps].view(bsz, 1, 1, 1)
            sqrt_alpha_t = alpha_t.sqrt()
            sqrt_one_minus_alpha = (1.0 - alpha_t).sqrt()
            pred_x0 = (noisy_latents - sqrt_one_minus_alpha * model_pred) / (sqrt_alpha_t + 1e-8)

            # 关键：用VQ-VAE内置的diffusion_vae解码（替代单独的sd_vae_decoder）
            gen_image = vq_vae.decode(pred_x0 / vq_vae.scale_factor, train_mode=False)
            # 感知损失计算
            T = scheduler.config.num_train_timesteps
            t_norm = timesteps.to(dtype=unet.dtype) / float(T - 1)
            base_perceptual_weight = 0.3
            perceptual_weights = base_perceptual_weight * (1.0 - t_norm).view(bsz, 1, 1, 1)
            perceptual_loss = perceptual_loss_fn(gen_image, target_images) * perceptual_weights.mean()

            # 6.3 KL损失（原逻辑不变）
            kl_loss = -0.5 * torch.mean(1 + code_logvar - code_mean ** 2 - code_logvar.exp())
            kl_weight = 0.5

            # 6.4 VAE相关损失（量化损失 + 重建损失）
            vq_weight = 0.1  # VQ损失权重
            vae_recon_weight = 0.2  # 关键：VAE重建损失权重（需根据训练效果调整）

            # 总损失（多损失协同优化）
            total_batch_loss = (
                    diffusion_loss +  # 扩散核心损失
                    perceptual_loss +  # 感知质量损失
                    kl_weight * kl_loss +  # KL正则化损失
                    vq_weight * vq_loss +  # VQ量化损失
                    vae_recon_weight * vae_recon_loss  # VAE重建损失（新增）
            )

            # 7. 反向传播与优化（原逻辑不变，但梯度包含VAE）
            optimizer.zero_grad()
            total_batch_loss.backward()
            # 梯度裁剪（包含所有可训练参数，避免爆炸）
            torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            # 8. 日志与样本保存（新增VAE重建损失日志）
            total_loss += total_batch_loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            if (batch_idx + 1) % CONFIG["log_interval"] == 0:
                print(
                    f"Batch {batch_idx + 1} | avg_loss: {avg_loss:.6f} "
                    f"(diffusion: {diffusion_loss.item():.6f}, perceptual: {perceptual_loss.item():.6f}, "
                    f"KL: {kl_loss.item():.6f}, VQ: {vq_loss.item():.6f}, VAE_recon: {vae_recon_loss.item():.6f})"
                    # 新增VAE重建损失
                )
                norm_val = code_modulation.norm(dim=1).mean().item()
                print(f"[Diag] code_modulation.norm = {norm_val:.4f}")

                # 访问 ModulatedUNet 内部的 modulation layers
                with torch.no_grad():
                    bott = unet.global_bottleneck(code_modulation)  # [B, bottleneck_dim]
                    for i, mod_layer in enumerate(unet.modulation_layers):
                        mod_out = mod_layer(bott)  # [B, 2*in_dim]
                        mean_val = mod_out.mean().item()
                        std_val = mod_out.std().item()
                        print(f"[Diag] Layer {i}: mean={mean_val:.4f}, std={std_val:.4f}")
            # if (batch_idx + 1) % CONFIG["save_image_interval"] == 0:
            #     samples_dir = os.path.join(CONFIG["output_dir"], "samples")
            #     if os.path.exists(samples_dir):
            #         shutil.rmtree(samples_dir)
            #     os.makedirs(samples_dir, exist_ok=True)
            #     # 样本保存：用VQ-VAE的diffusion_vae解码
            #     save_generated_samples(
            #         unet, vq_vae.diffusion_vae, scheduler,  # 替换为vq_vae.diffusion_vae
            #         fusion_outputs={"fused_clip": c1},
            #         code_modulation=code_modulation,
            #         texts=core_texts,
            #         batch_idx=batch_idx,
            #         epoch=epoch,
            #         output_dir=CONFIG["output_dir"],
            #         device=device,
            #         dtype=dtype,
            #         restore_train_mode=True
            #     )

        # 模型保存（确保包含VAE训练后的状态）
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(CONFIG["output_dir"], f"model_epoch_{epoch + 1}.pth")
            torch.save({
                "unet_state_dict": unet.state_dict(),
                "attn_state_dict": bidirectional_attn.state_dict(),
                # "residual_generator_state_dict": residual_generator.state_dict(),
                "vq_vae_state_dict": vq_vae.state_dict(),  # 关键：保存VAE训练后的权重（含解冻层+codebook）
                "optimizer_state_dict": optimizer.state_dict(),  # 新增：保存优化器状态（支持续训）
                "epoch": epoch + 1
            }, save_path)
            print(f"Epoch {epoch + 1} 模型保存至：{save_path}")
        save_generated_samples(
            unet, vq_vae, scheduler,  # 替换为vq_vae.diffusion_vae
            fusion_outputs={"fused_clip": c1},
            code_modulation=code_modulation,
            texts=core_texts,
            batch_idx=batch_idx,
            epoch=epoch,
            output_dir=CONFIG["output_dir"],
            device=device,
            dtype=dtype,
            restore_train_mode=True
        )

if __name__ == "__main__":
    main()




