from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import math
from .cross_net import CrossSparseAggrNet_v2
from typing import List, Tuple, Optional
from argparse import Namespace
from .alibi_attention import TransformerAli
import clip

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True




@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# class ResidualAttentionBlock(nn.Module):
#     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
#         super().__init__()
#
#         self.attn = nn.MultiheadAttention(d_model, n_head)
#         self.ln_1 = LayerNorm(d_model)
#         self.mlp = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.ln_2 = LayerNorm(d_model)
#         self.attn_mask = attn_mask
#
#     def attention(self, x: torch.Tensor):
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
#
#     def forward(self, x: torch.Tensor):
#         x = x + self.attention(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
#         return x
#
#
# class Transformer(nn.Module):
#     def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
#         super().__init__()
#         self.width = width
#         self.layers = layers
#         self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
#
#     def forward(self, x: torch.Tensor):
#         return self.resblocks(x)
#
#
# class VisionTransformer(nn.Module):
#     def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.output_dim = output_dim
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
#
#         scale = width ** -0.5
#         self.class_embedding = nn.Parameter(scale * torch.randn(width))
#         self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
#         self.ln_pre = LayerNorm(width)
#
#         self.transformer = Transformer(width, layers, heads)
#
#         self.ln_post = LayerNorm(width)
#         self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
#
#     def forward_full(self, x: torch.Tensor):
#         x = self.conv1(x) # shape = [*, width, grid, grid]
#         x = x.reshape(x.shape[0], x.shape[1], -1) # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1) # shape = [*, grid ** 2, width]
#         x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) # shape = [*, grid ** 2 + 1, width]
#         x = x + self.positional_embedding.to(x.dtype)
#         x = self.ln_pre(x)
#
#         x = x.permute(1, 0, 2) # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2) # LND -> NLD
#
#         x = self.ln_post(x)
#
#         if self.proj is not None:
#             x = x @ self.proj
#
#         return x
#
#     def forward(self, x: torch.Tensor):
#         x = self.conv1(x)  # shape = [*, width, grid, grid]
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         x = x + self.positional_embedding.to(x.dtype)
#         x = self.ln_pre(x)
#
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#
#         x = self.ln_post(x[:, 0, :])
#
#         if self.proj is not None:
#             x = x @ self.proj
#
#         return x



class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width  # 每层特征维度d
        self.layers = layers  # 总层数
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)
        ])
        # 保存每一层的完整序列特征（B, L, width），支持梯度回传
        self.intermediate_seq_feats = []

    def reset_intermediate(self):
        """重置中间层特征（每个batch前调用）"""
        self.intermediate_seq_feats.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.reset_intermediate()
        for block in self.resblocks:
            x = block(x)  # x: (L, B, width) → LND格式
            # 保存当前层完整序列特征（转换为NLD格式：B, L, width），不detach保留梯度
            self.intermediate_seq_feats.append(x.permute(1, 0, 2).clone())
        return x


# -------------------------- VisionTransformer（核心修改） --------------------------
class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,  # ViT原始特征维度（如1024）
        layers: int,  # ViT总层数
        heads: int,
        output_dim: int,  # 原始CLIP输出维度（兼容旧逻辑）
        n_low: int = 3,  # 前N层（低层特征）
        n_high: int = 3,  # 倒数M层（高层特征）
        dropout_rate: float = 0.1,  # Dropout概率
        hidden_dim: int = 768  # 映射后统一维度（与文本特征一致）
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.width = width  # 原始特征维度（如1024）
        self.hidden_dim = hidden_dim  # 映射后维度（768）
        self.layers = layers
        self.n_low = n_low
        self.n_high = n_high
        self.dropout_rate = dropout_rate

        # 原始ViT基础组件（不变）
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)  # 带中间层序列保存的Transformer
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))  # 兼容旧输出维度

        # -------------------------- 低层特征映射层（分全局+序列） --------------------------
        # 全局特征：前N层CLS拼接（N×width）→ 映射到hidden_dim（768）
        self.low_global_linear = nn.Linear(n_low * width, hidden_dim)
        # 序列特征：前N层序列拼接（N×width）→ 映射到hidden_dim（768）
        self.low_seq_linear = nn.Linear(n_low * width, hidden_dim)
        self.low_ln = LayerNorm(hidden_dim)
        self.low_dropout = nn.Dropout(dropout_rate)

        # -------------------------- 高层特征映射层（分全局+序列） --------------------------
        # 全局特征：倒数M层CLS拼接（M×width）→ 映射到hidden_dim（768）
        self.high_global_linear = nn.Linear(n_high * width, hidden_dim)
        # 序列特征：倒数M层序列拼接（M×width）→ 映射到hidden_dim（768）
        self.high_seq_linear = nn.Linear(n_high * width, hidden_dim)
        self.high_ln = LayerNorm(hidden_dim)
        self.high_dropout = nn.Dropout(dropout_rate)

        # 初始化映射层参数
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """初始化所有参数"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

    # 原始forward（兼容旧逻辑，返回最终CLS编码）
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # (b, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (b, width, grid²)
        x = x.permute(0, 2, 1)  # (b, grid², width)
        cls_tokens = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_tokens, x], dim=1)  # (b, grid²+1, width)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD → LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND → NLD

        x = self.ln_post(x[:, 0, :])  # 提取最终CLS
        if self.proj is not None:
            x = x @ self.proj  # 映射到output_dim
        return x

    # forward_full（返回完整token序列，用于基线推理）
    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # (b, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (b, width, grid²)
        x = x.permute(0, 2, 1)  # (b, grid², width)
        cls_tokens = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_tokens, x], dim=1)  # (b, grid²+1, width)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD → LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND → NLD

        if self.proj is not None:
            x = x @ self.proj  # 映射到output_dim
        return x  # (b, seq_len, output_dim)

    # -------------------------- 核心方法：提取多尺度特征（拼接+映射） --------------------------
    def extract_multi_scale_feats(self, x: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        输出：4个768维特征（全局特征用于文本精炼，序列特征用于双分支推理）
        - F_low_global: (B, 768) → 低层全局特征
        - F_high_global: (B, 768) → 高层全局特征
        - F_low_seq: (B, L_img, 768) → 低层序列特征
        - F_high_seq: (B, L_img, 768) → 高层序列特征
        """
        # 1. ViT基础前处理（生成初始序列特征，width维如1024）
        x = self.conv1(x)  # (B, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, width, grid²)
        x = x.permute(0, 2, 1)  # (B, grid², width)
        # 添加CLS token + 位置编码
        cls_tokens = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_tokens, x], dim=1)  # (B, L_img, width)，L_img=grid²+1
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # (L_img, B, width) → LND格式

        # 2. Transformer前向传播，获取中间层序列特征
        self.transformer(x)
        intermediate_seq = self.transformer.intermediate_seq_feats  # list of (B, L_img, width)
        num_layers = len(intermediate_seq)

        # 3. 层数合法性校验
        assert self.n_low <= num_layers, f"前N层({self.n_low})超过总层数({num_layers})"
        assert self.n_high <= num_layers, f"倒数M层({self.n_high})超过总层数({num_layers})"

        # -------------------------- 低层特征：拼接+映射（width→768） --------------------------
        # 1. 低层全局特征（前N层CLS拼接）
        low_cls_list = [feat[:, 0, :] for feat in intermediate_seq[:self.n_low]]  # list of (B, width)
        low_global_concat = torch.cat(low_cls_list, dim=-1)  # (B, n_low×width)
        F_low_global = self.low_global_linear(low_global_concat)  # (B, 768)
        F_low_global = self.low_ln(F_low_global)
        F_low_global = F.relu(F_low_global)
        F_low_global = self.low_dropout(F_low_global)
        F_low_global = F_low_global / F_low_global.norm(dim=-1, keepdim=True)

        # 2. 低层序列特征（前N层序列拼接）
        low_seq_list = intermediate_seq[:self.n_low]  # list of (B, L_img, width)
        low_seq_concat = torch.cat(low_seq_list, dim=-1)  # (B, L_img, n_low×width)
        F_low_seq = self.low_seq_linear(low_seq_concat)  # (B, L_img, 768)
        F_low_seq = self.low_ln(F_low_seq)
        F_low_seq = F.relu(F_low_seq)
        F_low_seq = self.low_dropout(F_low_seq)
        F_low_seq = F_low_seq / F_low_seq.norm(dim=-1, keepdim=True)

        # -------------------------- 高层特征：拼接+映射（width→768） --------------------------
        # 1. 高层全局特征（倒数M层CLS拼接）
        high_cls_list = [feat[:, 0, :] for feat in intermediate_seq[-self.n_high:]]  # list of (B, width)
        high_global_concat = torch.cat(high_cls_list, dim=-1)  # (B, n_high×width)
        F_high_global = self.high_global_linear(high_global_concat)  # (B, 768)
        F_high_global = self.high_ln(F_high_global)
        F_high_global = F.relu(F_high_global)
        F_high_global = self.high_dropout(F_high_global)
        F_high_global = F_high_global / F_high_global.norm(dim=-1, keepdim=True)

        # 2. 高层序列特征（倒数M层序列拼接）
        high_seq_list = intermediate_seq[-self.n_high:]  # list of (B, L_img, width)
        high_seq_concat = torch.cat(high_seq_list, dim=-1)  # (B, L_img, n_high×width)
        F_high_seq = self.high_seq_linear(high_seq_concat)  # (B, L_img, 768)
        F_high_seq = self.high_ln(F_high_seq)
        F_high_seq = F.relu(F_high_seq)
        F_high_seq = self.high_dropout(F_high_seq)
        F_high_seq = F_high_seq / F_high_seq.norm(dim=-1, keepdim=True)

        return F_low_global, F_high_global, F_low_seq, F_high_seq





# class CLIP(nn.Module):
#     def __init__(self,
#                  embed_dim: int,
#                  # vision
#                  image_resolution: int,
#                  vision_layers: Union[Tuple[int, int, int, int], int],
#                  vision_width: int,
#                  vision_patch_size: int,
#                  # text
#                  context_length: int,
#                  vocab_size: int,
#                  transformer_width: int,
#                  transformer_heads: int,
#                  transformer_layers: int,
#                  load_from_clip: bool,
#                  run_finelip: bool
#                  ):
#         super().__init__()
#
#         self.context_length = 248
#         self.run_finelip = run_finelip
#
#         if isinstance(vision_layers, (tuple, list)):
#             vision_heads = vision_width * 32 // 64
#             self.visual = ModifiedResNet(
#                 layers=vision_layers,
#                 output_dim=embed_dim,
#                 heads=vision_heads,
#                 input_resolution=image_resolution,
#                 width=vision_width
#             )
#         else:
#             vision_heads = vision_width // 64
#             self.visual = VisionTransformer(
#                 input_resolution=image_resolution,
#                 patch_size=vision_patch_size,
#                 width=vision_width,
#                 layers=vision_layers,
#                 heads=vision_heads,
#                 output_dim=embed_dim
#             )
#
#         self.transformer = Transformer(
#             width=transformer_width,
#             layers=transformer_layers,
#             heads=transformer_heads,
#             attn_mask=self.build_attention_mask()
#         )
#
#         self.vocab_size = vocab_size
#         self.token_embedding = nn.Embedding(vocab_size, transformer_width)
#
#         if load_from_clip == False:
#             self.positional_embedding = nn.Parameter(torch.empty(248, transformer_width))
#             self.positional_embedding_res = nn.Parameter(torch.empty(248, transformer_width))
#
#         else:
#             self.positional_embedding = nn.Parameter(torch.empty(77, transformer_width))
#
#         self.ln_final = LayerNorm(transformer_width)
#
#         self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#
#         self.initialize_parameters()
#         self.mask1 = torch.zeros([248, 1])
#         self.mask1[:20, :] = 1
#         self.mask2 = torch.zeros([248, 1])
#         self.mask2[20:, :] = 1
#
#         if self.run_finelip:
#             self.cross_net = CrossSparseAggrNet_v2()
#             self.criterion = None
#
#     def initialize_parameters(self):
#         nn.init.normal_(self.token_embedding.weight, std=0.02)
#         nn.init.normal_(self.positional_embedding, std=0.01)
#
#         if isinstance(self.visual, ModifiedResNet):
#             if self.visual.attnpool is not None:
#                 std = self.visual.attnpool.c_proj.in_features ** -0.5
#                 nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
#
#             for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
#                 for name, param in resnet_block.named_parameters():
#                     if name.endswith("bn3.weight"):
#                         nn.init.zeros_(param)
#
#         proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
#         attn_std = self.transformer.width ** -0.5
#         fc_std = (2 * self.transformer.width) ** -0.5
#         for block in self.transformer.resblocks:
#             nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
#             nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
#             nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
#             nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
#
#         if self.text_projection is not None:
#             nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
#
#     def build_attention_mask(self):
#         # lazily create causal attention mask, with full attention between the vision tokens
#         # pytorch uses additive attention mask; fill with -inf
#         mask = torch.empty(self.context_length, self.context_length)
#         mask.fill_(float("-inf"))
#         mask.triu_(1)  # zero out the lower diagonal
#         return mask
#
#     @property
#     def dtype(self):
#         return self.visual.conv1.weight.dtype
#
#     def encode_image(self, image):
#         return self.visual(image.type(self.dtype))
#
#     def encode_image_full(self, image):
#         return self.visual.forward_full(image.type(self.dtype))
#
#     def encode_text(self, text):
#         x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
#
#         x = x + (self.positional_embedding.to(x.device) * self.mask1.to(x.device)).type(self.dtype).to(x.device) + (
#                     self.positional_embedding_res.to(x.device) * self.mask2.to(x.device)).type(self.dtype).to(x.device)
#
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#
#         x = self.ln_final(x).type(self.dtype)
#
#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
#
#         return x
#
#     def encode_text_full(self, text):
#         x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
#
#         x = x + (self.positional_embedding.to(x.device) * self.mask1.to(x.device)).type(self.dtype).to(x.device) + (
#                     self.positional_embedding_res.to(x.device) * self.mask2.to(x.device)).type(self.dtype).to(x.device)
#
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)
#
#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
#
#         return x
#
#     def inference_lvl1(self, image_feats, text_feats, rank=0):
#         if is_dist_avail_and_initialized():
#             image_feats_all = torch.cat(dist_nn.all_gather(image_feats), dim=0)
#             text_feats_all = torch.cat(dist_nn.all_gather(text_feats), dim=0)
#         else:
#             image_feats_all = image_feats.clone()
#             text_feats_all = text_feats.clone()
#
#         sim_i2t = torch.matmul(image_feats, text_feats_all.T)
#         sim_t2i = torch.matmul(image_feats_all, text_feats.T)
#         sim_t2i = sim_t2i.T
#
#         sim_i2t = self.logit_scale.exp() * sim_i2t
#         sim_t2i = self.logit_scale.exp() * sim_t2i
#
#         bs = image_feats.size(0)
#         targets = torch.linspace(
#             rank * bs, (rank + 1) * bs - 1, bs, dtype=torch.long, device=image_feats.device
#         )
#
#         loss_itc = (
#                            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1) +
#                            F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
#                    ) / 2
#
#         return loss_itc
#
#     def inference_lvl3(self, img_emb, cap_emb, lengths, img_ids=None,
#                        warmup_alpha=1.0, sparse_ratio=0.5, ratio_weight=2.0):
#         if is_dist_avail_and_initialized():
#             # lengths = concat_all_gather(lengths)
#             img_ids = concat_all_gather(img_ids)
#
#             max_len = int(lengths.max())
#
#             if max_len > cap_emb.shape[1]:
#                 # (B, L_max -L, C)
#                 pad_emb = torch.zeros(cap_emb.shape[0], max_len - cap_emb.shape[1], cap_emb.shape[2]).to(cap_emb.device)
#                 # (B, L, C) + (B, L_max -L, C) = (B, L_max, C)
#                 cap_emb = torch.cat([cap_emb, pad_emb], dim=1)
#
#             img_emb = torch.cat(dist_nn.all_gather(img_emb), dim=0)
#             # cap_emb = torch.cat(dist_nn.all_gather(cap_emb), dim=0)
#
#         # compute similarity matrix
#         improved_sims, score_mask_all = self.cross_net.forward_dual_aggr(img_emb, cap_emb, lengths)
#
#         if is_dist_avail_and_initialized():
#             improved_sims = torch.cat(dist_nn.all_gather(improved_sims), dim=1)
#
#         # basic alignment loss
#         align_loss = self.criterion(img_emb, cap_emb, img_ids, improved_sims)
#         align_loss *= warmup_alpha
#
#         # ratio_loss
#         # ratio_loss = (score_mask_all.mean() - sparse_ratio) ** 2
#
#         # loss = align_loss + ratio_weight * ratio_loss
#         return align_loss,improved_sims
#
#     def forward(self, image, text, img_ids, warmup_alpha, rank=0):
#         self.rank = rank
#         lengths = torch.tensor([torch.nonzero(text[i]).size(0) for i in range(text.shape[0])], device=text.device)
#         image_features_full = self.encode_image_full(image)
#
#         text_features_full = self.encode_text_full(text) @ self.text_projection
#
#         image_features = image_features_full[:, 0, :]
#         image_features = image_features / image_features.norm(dim=1, keepdim=True)
#
#         text_features = text_features_full[torch.arange(text_features_full.shape[0]), text.argmax(dim=-1)]
#         text_features = text_features / text_features.norm(dim=1, keepdim=True)
#
#         loss_1 = 0.0
#         loss_3 = torch.tensor([0]).to(text.device)
#         if self.run_finelip:
#             loss_3,improved_sims = self.inference_lvl3(image_features_full, text_features_full, lengths, img_ids, warmup_alpha)
#         else:  # run baseline
#             loss_1 = self.inference_lvl1(image_features, text_features, rank)
#
#         return loss_1, loss_3, improved_sims


# -------------------------- 跨模态精炼组件（放在CLIP类文件中） --------------------------
class CrossModalAttention(nn.Module):
    """跨模态注意力：文本Q，图像K/V，独立为两个分支设计"""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, text_feat: torch.Tensor, visual_feat: torch.Tensor) -> torch.Tensor:
        B = text_feat.shape[0]
        q = self.w_q(text_feat).reshape(B * self.n_head, -1, self.head_dim)
        k = self.w_k(visual_feat).reshape(B * self.n_head, -1, self.head_dim)
        v = self.w_v(visual_feat).reshape(B * self.n_head, -1, self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v).reshape(B, -1, self.d_model)
        attn_out = self.w_o(attn_out)
        attn_out = self.dropout(attn_out)
        return self.layer_norm(text_feat + attn_out)


class FFN(nn.Module):
    """前馈网络：独立为两个分支设计"""

    def __init__(self, d_model: int, hidden_dim: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x + self.mlp(x))


# -------------------------- 完整CLIP类（核心修改） --------------------------
class CLIP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            # vision
            image_resolution: int,
            vision_layers: Union[Tuple[int, int, int, int], int],
            vision_width: int,
            vision_patch_size: int,
            # text
            context_length: int,
            vocab_size: int,
            transformer_width: int,
            transformer_heads: int,
            transformer_layers: int,
            # 关键参数
            load_from_clip: bool,
            # 多层聚合参数
            n_low: int = 3,
            n_high: int = 3,
            agg_dropout_rate: float = 0.1,
            # 聚合模块参数
            aggr_n_head: int = 8,
            aggr_hidden_dim: int = 4096,
            aggr_dropout: float = 0.1,
            # 其他参数
            adapter_bottleneck: int = 64,
            adapter_dropout: float = 0.1,
            run_finelip: bool = True
    ):
        super().__init__()

        self.context_length = 248
        self.run_finelip = run_finelip
        self.embed_dim = embed_dim
        self.transformer_width = transformer_width  # 768维
        self.n_low = n_low
        self.n_high = n_high

        # -------------------------- 视觉模块（支持提取非CLS多尺度特征） --------------------------
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
            self.support_multi_scale = False
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                n_low=n_low,
                n_high=n_high,
                dropout_rate=agg_dropout_rate,
                hidden_dim=transformer_width  # 映射到768
            )
            self.support_multi_scale = True

        # -------------------------- 文本模块（输出完整token特征，无CLS截取） --------------------------
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        if not load_from_clip:
            self.positional_embedding = nn.Parameter(torch.empty(248, transformer_width))
            self.positional_embedding_res = nn.Parameter(torch.empty(248, transformer_width))
        else:
            self.positional_embedding = nn.Parameter(torch.empty(77, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # -------------------------- 核心：文本融合权重（可学习，非CLS文本token） --------------------------
        self.text_weights = nn.Parameter(torch.tensor([ 0.15, 0.15], dtype=torch.float32))  # 原始:粗:细
        self.softmax = nn.Softmax(dim=0)

        # -------------------------- 单聚合实例 + 文本融合模块（处理非CLS特征） --------------------------
        self.aggr_net = None
        self.coarse_fusion = None  # 粗文本：原始非CLS文本 + 低层非CLS图像
        self.fine_fusion = None    # 细文本：粗文本 + 高层非CLS图像

        if self.run_finelip and self.support_multi_scale:
            # 聚合模块参数
            class Opt:
                def __init__(self, clip_self):
                    self.embed_size = clip_self.embed_dim
                    self.num_patches = (clip_self.visual.input_resolution // clip_self.visual.patch_size) ** 2 if hasattr(
                        clip_self.visual, 'patch_size') else 196
            opt = Opt(self)
            self.cross_net = CrossSparseAggrNet_v2(opt=opt)  # 唯一聚合实例

            # 粗/细文本融合（处理非CLS文本token）
            self.coarse_fusion = nn.Sequential(
                CrossModalAttention(transformer_width, aggr_n_head, aggr_dropout),
                FFN(transformer_width, aggr_hidden_dim, aggr_dropout)
            )
            self.fine_fusion = nn.Sequential(
                CrossModalAttention(transformer_width, aggr_n_head, aggr_dropout),
                FFN(transformer_width, aggr_hidden_dim, aggr_dropout)
            )

        # -------------------------- 损失函数（基于非CLS特征） --------------------------
        self.criterion = self._build_criterion()

        # -------------------------- 初始化 --------------------------
        self.initialize_parameters()
        self.mask1 = torch.zeros([248, 1])
        self.mask1[:20, :] = 1
        self.mask2 = torch.zeros([248, 1])
        self.mask2[20:, :] = 1

    def _build_opt(self):
        """为CrossSparseAggrNet_v2构建参数"""
        class Opt:
            def __init__(self, clip_self):
                self.embed_size = clip_self.embed_dim
                self.num_patches = (clip_self.visual.input_resolution // clip_self.visual.patch_size) ** 2 if hasattr(
                    clip_self.visual, 'patch_size') else 196
                self.sparse_ratio = 0.5
                self.aggr_ratio = 0.4
                self.attention_weight = 0.8
                self.ratio_weight = 2.0
        return Opt(self)

    def _build_criterion(self):
        """占位：用户需替换为实际损失函数"""
        def criterion(im, s, img_ids, scores):
            return torch.mean(scores)  # 示例返回，需修改
        return criterion

    # -------------------------- 初始化方法（原有逻辑不变） --------------------------
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        if hasattr(self, 'positional_embedding'):
            nn.init.normal_(self.positional_embedding, std=0.01)
        if hasattr(self, 'positional_embedding_res'):
            nn.init.normal_(self.positional_embedding_res, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)
        else:
            # 初始化ViT的映射层
            for m in [self.visual.low_global_linear, self.visual.low_seq_linear,
                      self.visual.high_global_linear, self.visual.high_seq_linear]:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

        # 初始化双分支参数
        if self.run_finelip and self.support_multi_scale:
            for module in [self.coarse_fusion, self.fine_fusion]:
                for name, param in module.named_parameters():
                    if isinstance(param, nn.Linear):
                        nn.init.xavier_uniform_(param.weight)
                        if param.bias is not None:
                            nn.init.zeros_(param.bias)
                    elif isinstance(param, nn.LayerNorm):
                        nn.init.ones_(param.weight)
                        nn.init.zeros_(param.bias)
            for name, param in self.cross_net.named_parameters():
                if isinstance(param, nn.Linear):
                    nn.init.xavier_uniform_(param.weight)
                    if param.bias is not None:
                        nn.init.zeros_(param.bias)

    # -------------------------- 辅助方法（原有逻辑不变） --------------------------
    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_image_full(self, image):
        return self.visual.forward_full(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + (self.positional_embedding.to(x.device) * self.mask1.to(x.device)).type(self.dtype) + (
                self.positional_embedding_res.to(x.device) * self.mask2.to(x.device)).type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_text_full(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + (self.positional_embedding.to(x.device) * self.mask1.to(x.device)).type(self.dtype) + (
                self.positional_embedding_res.to(x.device) * self.mask2.to(x.device)).type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        return x

    def inference_lvl1(self, image_feats, text_feats, rank=0):
        if is_dist_avail_and_initialized():
            image_feats_all = torch.cat(dist_nn.all_gather(image_feats), dim=0)
            text_feats_all = torch.cat(dist_nn.all_gather(text_feats), dim=0)
        else:
            image_feats_all = image_feats.clone()
            text_feats_all = text_feats.clone()

        sim_i2t = torch.matmul(image_feats, text_feats_all.T)
        sim_t2i = torch.matmul(image_feats_all, text_feats.T).T

        sim_i2t = self.logit_scale.exp() * sim_i2t
        sim_t2i = self.logit_scale.exp() * sim_t2i

        bs = image_feats.size(0)
        targets = torch.linspace(
            rank * bs, (rank + 1) * bs - 1, bs, dtype=torch.long, device=image_feats.device
        )

        loss_itc = (F.cross_entropy(sim_i2t, targets, label_smoothing=0.1) + F.cross_entropy(sim_t2i, targets,
                                                                                             label_smoothing=0.1)) / 2
        return loss_itc

    # -------------------------- 文本分支拆分（原有逻辑不变） --------------------------

    def fuse_text_features(self, text_raw: torch.Tensor, text_coarse: torch.Tensor,
                           text_fine: torch.Tensor) -> torch.Tensor:
        """
        融合文本特征：text_raw权重固定为1，text_coarse/text_fine权重严格小于0.5（避免影响过大）
        - text_raw：原始文本特征（无权重，固定贡献）
        - text_coarse：粗文本特征（权重∈(0, 0.5)）
        - text_fine：细文本特征（权重∈(0, 0.5)）
        """
        # 取可学习参数的后两个维度，用于coarse和fine的权重
        w_coarse, w_fine = self.text_weights[0], self.text_weights[1]

        # 用sigmoid映射到(0,1)，再乘以0.5确保权重严格小于0.5
        w_coarse_clamped = torch.sigmoid(w_coarse) * 0.5  # ∈(0, 0.5)
        w_fine_clamped = torch.sigmoid(w_fine) * 0.5  # ∈(0, 0.5)

        # 融合公式：text_raw（固定1） + 加权coarse + 加权fine
        fused = text_raw + w_coarse_clamped * text_coarse + w_fine_clamped * text_fine

        # 层归一化保证数值稳定性
        return F.layer_norm(fused, normalized_shape=[self.transformer_width])

    def _pad_text_to_match(self, text_feat: torch.Tensor, target_len: int) -> torch.Tensor:
        B, L, C = text_feat.shape
        if L >= target_len:
            return text_feat[:, :target_len, :]
        else:
            pad = torch.zeros(B, target_len - L, C, dtype=text_feat.dtype, device=text_feat.device)
            return torch.cat([text_feat, pad], dim=1)

    # -------------------------- 最终forward（原有逻辑不变） --------------------------
    def inference_lvl3(self, img_emb, cap_emb, lengths, img_ids=None,
                       warmup_alpha=1.0, sparse_ratio=0.5, ratio_weight=2.0, image=None):
        """
        最终运行版：去掉所有打印逻辑，保留核心功能和关键修复
        修复点：分布式同步lengths、防御性判断image、正确返回双值、文本统一长度248
        """
        # 防御性判断：确保image非空且数据类型正确
        if image is None:
            raise ValueError("inference_lvl3 必须传入原始图像 image 参数！")
        image = image.type(self.dtype)

        # 分布式处理：同步img_ids、lengths、cap_emb、img_emb
        if is_dist_avail_and_initialized():
            if img_ids is not None and img_ids.dim() > 0:
                img_ids = concat_all_gather(img_ids)
            # 同步lengths（关键修复：避免不同GPU长度不一致）
            lengths = concat_all_gather(lengths)
            # 处理cap_emb padding
            max_len = int(lengths.max().item()) if lengths.dim() > 0 else cap_emb.shape[1]
            if max_len > cap_emb.shape[1]:
                pad_emb = torch.zeros(cap_emb.shape[0], max_len - cap_emb.shape[1], cap_emb.shape[2]).to(cap_emb.device)
                cap_emb = torch.cat([cap_emb, pad_emb], dim=1)
            # 同步img_emb
            img_emb = torch.cat(dist_nn.all_gather(img_emb), dim=0)

        # 加固lengths维度（避免0维）
        if lengths.dim() == 0:
            lengths = lengths.unsqueeze(0)

        # 文本处理：保留CLS，统一padding到248
        max_valid_len = 248
        text_cls_included_list = []
        for i in range(cap_emb.shape[0]):
            valid_len = lengths[i].item() if (i < len(lengths) and lengths.dim() > 0) else cap_emb.shape[1]
            # 截断到有效长度（保留CLS）
            text_i = cap_emb[i:i + 1, :valid_len, :]
            # Padding到248
            if text_i.shape[1] < max_valid_len:
                pad = torch.zeros(1, max_valid_len - text_i.shape[1], text_i.shape[2],
                                  device=text_i.device, dtype=text_i.dtype)
                text_i = torch.cat([text_i, pad], dim=1)
            text_cls_included_list.append(text_i)
        text_cls_included = torch.cat(text_cls_included_list, dim=0)

        # 多尺度特征提取
        img_low_global, img_high_global, _, _ = self.visual.extract_multi_scale_feats(image)
        img_low_seq = img_low_global.unsqueeze(1)
        img_high_seq = img_high_global.unsqueeze(1)

        # 文本融合：粗特征→细特征→最终融合
        text_coarse = self.coarse_fusion[0](text_cls_included, img_low_seq)
        text_coarse = self.coarse_fusion[1](text_coarse)
        text_coarse = F.normalize(text_coarse, dim=-1)

        text_fine = self.fine_fusion[0](text_coarse, img_high_seq)
        text_fine = self.fine_fusion[1](text_fine)
        text_fine = F.normalize(text_fine, dim=-1)

        text_fused = self.fuse_text_features(text_cls_included, text_coarse, text_fine)

        # 分布式最终padding（确保文本长度一致）
        if is_dist_avail_and_initialized():
            max_len = int(lengths.max().item()) if lengths.dim() > 0 else cap_emb.shape[1]
            if max_len > text_fused.shape[1]:
                pad_emb = torch.zeros(text_fused.shape[0], max_len - text_fused.shape[1], text_fused.shape[2]).to(
                    text_fused.device)
                text_fused = torch.cat([text_fused, pad_emb], dim=1)

        # 调用cross_net（3维特征输入，参数顺序不变）
        improved_sims, score_mask_all = self.cross_net.forward_dual_aggr(
            img_emb, text_fused, lengths
        )

        # 分布式同步improved_sims
        if is_dist_avail_and_initialized() and isinstance(improved_sims, torch.Tensor):
            improved_sims = torch.cat(dist_nn.all_gather(improved_sims), dim=1)

        # 损失计算
        align_loss = self.criterion(img_emb, text_fused, img_ids, improved_sims)
        align_loss *= warmup_alpha

        # 正确返回双值（匹配forward解包需求）
        return align_loss, improved_sims

    # -------------------------- 严格匹配用户给定的forward结构 --------------------------
    def forward(self, image, text, img_ids, warmup_alpha, rank=0):
        self.rank = rank
        lengths = torch.tensor([torch.nonzero(text[i]).size(0) for i in range(text.shape[0])], device=text.device)

        image_features_full = self.encode_image_full(image)



        text_features_full = self.encode_text_full(text) @ self.text_projection

        image_features = image_features_full[:, 0, :]
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        text_features = text_features_full[torch.arange(text_features_full.shape[0]), text.argmax(dim=-1)]
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        loss_1 = 0.0
        loss_3 = torch.tensor([0.0], dtype=torch.float32).to(text.device)
        improved_sims = None
        if self.run_finelip:
            loss_3, improved_sims = self.inference_lvl3(image_features_full, text_features_full, lengths, img_ids, warmup_alpha,image=image)
        else:  # run baseline
            loss_1 = self.inference_lvl1(image_features, text_features, rank)

        return loss_1, loss_3




















def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)



def convert_weights_ali(model: nn.Module):
    """
    将模型参数转换为 FP16 精度（仅对线性层等权重参数，跳过层归一化等需要 FP32 的参数）
    """

    def _convert_weights_to_fp32(l):
        # 1. 处理卷积层和全连接层（与原逻辑一致）
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        # 2. 处理PyTorch内置多头注意力（原逻辑保留，你的模型暂不使用，留作兼容）
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr, None)  # 加None默认值，避免属性不存在报错
                if tensor is not None:
                    tensor.data = tensor.data.float()

        # 3. 处理自定义模块的关键投影层（适配你的模型）
        # 3.1 处理文本全局投影层（CLIPWithFusionAndOT的text_projection）
        if hasattr(l, "text_projection"):
            attr = getattr(l, "text_projection")
            if attr is not None and isinstance(attr, nn.Parameter):
                attr.data = attr.data.float()

        # 3.2 处理自定义注意力的输出投影层（ALiBiAttention的proj）
        if hasattr(l, "proj"):
            attr = getattr(l, "proj")
            # 确保proj是Linear层（你的ALiBiAttention中proj是nn.Linear）
            if isinstance(attr, nn.Linear):
                attr.weight.data = attr.weight.data.float()
                if attr.bias is not None:
                    attr.bias.data = attr.bias.data.float()

        # 3.3 处理自定义注意力的QKV投影层（ALiBiAttention的qkv）
        if hasattr(l, "qkv"):
            attr = getattr(l, "qkv")
            # 确保qkv是Linear层（你的ALiBiAttention中qkv是nn.Linear）
            if isinstance(attr, nn.Linear):
                attr.weight.data = attr.weight.data.float()
                if attr.bias is not None:
                    attr.bias.data = attr.bias.data.float()

        # 4. 明确排除对精度敏感的模块（避免nan，关键补充）
        if isinstance(l, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            return  # 这些模块保留FP32，不做转换

    # 对模型所有模块应用转换逻辑（自动递归处理子模块）
    model.apply(_convert_weights_to_fp32)



def build_model(state_dict: dict, load_from_clip: bool, run_finelip: bool):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, load_from_clip, run_finelip
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, strict=False) # partial loading
    return model.eval()


def build_model_2(
        state_dict: dict,
        load_from_clip: bool,
        run_fusion: bool = True,
        fusion_num_heads: int = 8,
        ot_epsilon: float = 1e-2,
        alibi_start_alpha: float = 0.1,
        alibi_K: int = 77,
        alibi_delta: int = 2,
        alibi_gamma: float = 1.0,
        device: str = "cpu"
):
    """构建CLIPWithFusionAndOT模型（位置编码逻辑完全对齐原始build_model）"""
    # ---------------------- 1. 提取视觉分支参数（与原始build_model一致） ----------------------
    vit = "visual.proj" in state_dict
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([
            k for k in state_dict.keys()
            if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
        ])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}")))
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0], \
            "ResNet视觉位置嵌入长度不匹配"
        image_resolution = output_width * 32

    # ---------------------- 2. 提取文本分支参数（完全对齐原始build_model：context_length从权重获取） ----------------------
    required_text_keys = ["text_projection", "token_embedding.weight", "ln_final.weight"]
    for key in required_text_keys:
        assert key in state_dict, f"预训练权重缺少关键文本层：{key}"

    embed_dim = state_dict["text_projection"].shape[1]
    # 【核心对齐】context_length始终从加载的权重中提取（原始CLIP逻辑）
    context_length = state_dict["positional_embedding"].shape[0]  # 训练/测试均由权重决定（77或248）
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    print(f"[位置编码逻辑] 从权重提取context_length：{context_length}（与原始build_model一致）")

    # ---------------------- 3. 初始化模型（传递从权重提取的context_length） ----------------------
    model = CLIPWithFusionAndOT(
        # 原始CLIP核心参数（context_length由权重决定）
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        context_length=context_length,  # 关键：使用从权重提取的长度
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        load_from_clip=load_from_clip,  # 仅控制是否复用预训练权重，不直接决定长度
        run_fusion=run_fusion,
        fusion_num_heads=fusion_num_heads,

        # ALiBi/Finelip新增参数
        alibi_start_alpha=alibi_start_alpha,
        alibi_K=alibi_K,
        alibi_delta=alibi_delta,
        alibi_gamma=alibi_gamma,
        keep_len=20
    )

    # ---------------------- 4. 权重预处理与映射（保留完整映射逻辑） ----------------------
    new_state_dict = state_dict.copy()
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in new_state_dict:
            del new_state_dict[key]

    # ALiBi注意力权重映射（完整保留所有层的映射）
    for layer_idx in range(transformer_layers):
        # 1. 注意力QKV输入投影
        clip_in_proj_weight = f"transformer.resblocks.{layer_idx}.attn.in_proj_weight"
        clip_in_proj_bias = f"transformer.resblocks.{layer_idx}.attn.in_proj_bias"
        model_in_proj_weight = f"transformer.resblocks.{layer_idx}.in_proj.weight"
        model_in_proj_bias = f"transformer.resblocks.{layer_idx}.in_proj.bias"
        if clip_in_proj_weight in new_state_dict:
            new_state_dict[model_in_proj_weight] = new_state_dict.pop(clip_in_proj_weight)
        if clip_in_proj_bias in new_state_dict:
            new_state_dict[model_in_proj_bias] = new_state_dict.pop(clip_in_proj_bias)


        # 2. 注意力输出投影
        clip_out_proj_weight = f"transformer.resblocks.{layer_idx}.attn.out_proj.weight"
        clip_out_proj_bias = f"transformer.resblocks.{layer_idx}.attn.out_proj.bias"
        model_out_proj_weight = f"transformer.resblocks.{layer_idx}.attn.out_proj.weight"
        model_out_proj_bias = f"transformer.resblocks.{layer_idx}.attn.out_proj.bias"
        if clip_out_proj_weight in new_state_dict:
            new_state_dict[model_out_proj_weight] = new_state_dict.pop(clip_out_proj_weight)
        if clip_out_proj_bias in new_state_dict:
            new_state_dict[model_out_proj_bias] = new_state_dict.pop(clip_out_proj_bias)

        # 3. LayerNorm层
        clip_ln1_weight = f"transformer.resblocks.{layer_idx}.ln_1.weight"
        clip_ln1_bias = f"transformer.resblocks.{layer_idx}.ln_1.bias"
        model_ln1_weight = f"transformer.resblocks.{layer_idx}.ln_1.weight"
        model_ln1_bias = f"transformer.resblocks.{layer_idx}.ln_1.bias"
        if clip_ln1_weight in new_state_dict:
            new_state_dict[model_ln1_weight] = new_state_dict.pop(clip_ln1_weight)
        if clip_ln1_bias in new_state_dict:
            new_state_dict[model_ln1_bias] = new_state_dict.pop(clip_ln1_bias)

        clip_ln2_weight = f"transformer.resblocks.{layer_idx}.ln_2.weight"
        clip_ln2_bias = f"transformer.resblocks.{layer_idx}.ln_2.bias"
        model_ln2_weight = f"transformer.resblocks.{layer_idx}.ln_2.weight"
        model_ln2_bias = f"transformer.resblocks.{layer_idx}.ln_2.bias"
        if clip_ln2_weight in new_state_dict:
            new_state_dict[model_ln2_weight] = new_state_dict.pop(clip_ln2_weight)
        if clip_ln2_bias in new_state_dict:
            new_state_dict[model_ln2_bias] = new_state_dict.pop(clip_ln2_bias)

        # 4. MLP层
        clip_mlp_fc_weight = f"transformer.resblocks.{layer_idx}.mlp.c_fc.weight"
        clip_mlp_fc_bias = f"transformer.resblocks.{layer_idx}.mlp.c_fc.bias"
        model_mlp_fc_weight = f"transformer.resblocks.{layer_idx}.mlp.c_fc.weight"
        model_mlp_fc_bias = f"transformer.resblocks.{layer_idx}.mlp.c_fc.bias"
        if clip_mlp_fc_weight in new_state_dict:
            new_state_dict[model_mlp_fc_weight] = new_state_dict.pop(clip_mlp_fc_weight)
        if clip_mlp_fc_bias in new_state_dict:
            new_state_dict[model_mlp_fc_bias] = new_state_dict.pop(clip_mlp_fc_bias)

        clip_mlp_proj_weight = f"transformer.resblocks.{layer_idx}.mlp.c_proj.weight"
        clip_mlp_proj_bias = f"transformer.resblocks.{layer_idx}.mlp.c_proj.bias"
        model_mlp_proj_weight = f"transformer.resblocks.{layer_idx}.mlp.c_proj.weight"
        model_mlp_proj_bias = f"transformer.resblocks.{layer_idx}.mlp.c_proj.bias"
        if clip_mlp_proj_weight in new_state_dict:
            new_state_dict[model_mlp_proj_weight] = new_state_dict.pop(clip_mlp_proj_weight)
        if clip_mlp_proj_bias in new_state_dict:
            new_state_dict[model_mlp_proj_bias] = new_state_dict.pop(clip_mlp_proj_bias)

    # 权重类型转换（与原始build_model一致）
    convert_weights_ali(model)

    # ---------------------- 5. 权重加载与验证（预期缺失键调整） ----------------------
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    # 预期缺失键：仅包含新增参数（与原始逻辑一致，不依赖load_from_clip状态）
    alibi_expected_missing = [
        f"transformer.resblocks.{layer_idx}.attn.alpha_scale"
        for layer_idx in range(transformer_layers)
    ]
    # positional_embedding_res是新增参数，原始权重中始终缺失（无论训练/测试）
    finelip_expected_missing = ["positional_embedding_res"]
    # 融合模块：仅run_fusion=False时预期缺失
    fusion_expected_missing = [k for k in missing_keys if "cross_fusion" in k] if not run_fusion else []

    expected_missing = alibi_expected_missing + finelip_expected_missing + fusion_expected_missing
    unexpected_missing = [k for k in missing_keys if k not in expected_missing]


    print(f"[权重加载成功] 预期缺失键共{len(expected_missing)}个："
          f"ALiBi({len(alibi_expected_missing)}) + Finelip({len(finelip_expected_missing)}) + 融合({len(fusion_expected_missing)})")

    # ---------------------- 6. 返回模型（与原始build_model一致） ----------------------
    return model.to(device).eval()