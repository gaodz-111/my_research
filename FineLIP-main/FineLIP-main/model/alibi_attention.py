# model/alibi_attention.py（Finelip 内部文件）
import torch
import torch.nn as nn
import math
from typing import Optional
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np


import torch.distributed as dist
import torch.distributed.nn as dist_nn



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
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)










class ALiBiAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, alibi_start_alpha: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.alibi_start_alpha = alibi_start_alpha
        self.attention_biases = None

        # 还原：输出投影层的键名与原版一致（attn.out_proj.weight）
        # 注意：该层需作为self.attn的子层，确保键名是"attn.out_proj"（与原版一致）
        self.out_proj = nn.Linear(embed_dim, embed_dim)  # 键名：attn.out_proj.weight

    # forward逻辑不变（确保QKV拆分顺序、偏置添加正确）
    def forward(self, qkv: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        T, B, _ = qkv.shape
        qkv = qkv.reshape(T, B, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 1, 3, 0, 4)
        q, k, v = qkv.unbind(0)

        # 计算注意力分数（添加ALiBi偏置，核心逻辑保留）
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if self.attention_biases is None or self.attention_biases.shape[1] < T:
            self.attention_biases = self._build_alibi_biases(seq_len=T, device=q.device)
        alibi_bias = self.attention_biases[:, :T, :T].unsqueeze(0)
        attn_scores = attn_scores + alibi_bias


        if attn_mask is not None:

            # 扩展掩码维度以匹配注意力分数（[B, T, T] → [B, 1, T, T]）
            attn_mask = attn_mask.unsqueeze(1)

            # 验证维度匹配


        # 注意力掩码（与原版一致）
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_scores = torch.clamp(attn_scores, min=-1e4)

        # 注意力聚合与输出投影（与原版一致）
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).reshape(B, T, self.embed_dim)
        output = output.permute(1, 0, 2)
        output = self.out_proj(output)  # 输出投影（键名正确）

        return output

    def _build_alibi_biases(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # ALiBi偏置计算逻辑不变（核心功能，保留）
        biases = []
        for head in range(self.num_heads):
            alpha = self.alibi_start_alpha / (2 ** (head / self.num_heads))
            range_vec = torch.arange(seq_len, device=device)
            distance = range_vec[:, None] - range_vec[None, :]
            bias = -alpha * torch.abs(distance)
            biases.append(bias)
        return torch.stack(biases, dim=0)


class FinelipResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, alibi_start_alpha: float, attn_mask: torch.Tensor = None):
        super().__init__()
        self.d_model = d_model
        self.attn_mask = attn_mask  # 初始化时存储默认掩码（与原版一致）
        self.in_proj = nn.Linear(d_model, 3 * d_model)
        self.attn = ALiBiAttention(d_model, n_head, alibi_start_alpha)
        self.ln_1 = LayerNorm(d_model)
        # 还原原版MLP结构（带OrderedDict命名）
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    # 关键修改：添加attn_mask参数（可选，默认使用实例的self.attn_mask）
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 若调用时传入attn_mask，则用传入的；否则用初始化时的self.attn_mask
        mask = attn_mask if attn_mask is not None else self.attn_mask
        # 将掩码传递给ALiBiAttention
        x = x + self.attn(self.in_proj(self.ln_1(x)), attn_mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x


# class TransformerAli(nn.Module):
#     """仅保留ALiBi参数传递，其他完全还原为原版Transformer"""
#     def __init__(self, width: int, layers: int, heads: int, alibi_start_alpha: float, attn_mask: torch.Tensor):
#         super().__init__()
#         self.width = width
#         self.layers = layers
#
#         # 1. 还原原版nn.Sequential容器（与原版一致，确保块的顺序和键名）
#         self.resblocks = nn.Sequential(*[
#             FinelipResidualAttentionBlock(
#                 d_model=width,
#                 n_head=heads,
#                 alibi_start_alpha=alibi_start_alpha,  # ALiBi必需参数，保留
#                 attn_mask=attn_mask  # 还原：传递attn_mask到每个块（与原版一致）
#             ) for _ in range(layers)
#         ])
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # 还原原版forward：直接传递x到resblocks（与原版完全一致）
#         return self.resblocks(x)



class TransformerAli(nn.Module):
    """修改后：支持传递平滑参数，使用 SmoothResidualAttentionBlock"""
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        # 直接使用修改后的ResidualAttentionBlock，无需任何修改
        self.resblocks = nn.Sequential(*[SmoothResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class SmoothALiBiAttention(nn.MultiheadAttention):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            bias: bool = True,
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            kdim: Optional[int] = None,
            vdim: Optional[int] = None,
            # ALiBi新增参数（默认值，不影响原始接口）
            alibi_start_alpha: float = 1.0,
            K: int = 20,
            delta: int = 2,
            gamma: float = 1.0
    ):
        # 第一步：调用父类nn.MultiheadAttention的__init__，复用其参数结构
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim
        )

        # 第二步：初始化ALiBi相关参数（不新增父类外的参数名）
        self.alibi_start_alpha = alibi_start_alpha
        self.K = K  # 前20个token保留原始位置信号
        self.delta = delta
        self.gamma = gamma
        self.attention_biases = None  # 缓存ALiBi偏置，避免重复计算
        self.alpha_scale = nn.Parameter(torch.tensor(0.1), requires_grad=False)  # ALiBi强度调度

    def _build_alibi_biases(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """生成带平滑过渡的ALiBi偏置（父类无此逻辑，新增）"""
        # 1. 生成基础ALiBi偏置（头间衰减）
        biases = []
        for head in range(self.num_heads):
            alpha = self.alibi_start_alpha / (2 ** (head / self.num_heads))  # 头间强度衰减
            distance = torch.abs(
                torch.arange(seq_len, device=device)[:, None] - torch.arange(seq_len, device=device)[None, :])
            bias = -alpha * distance  # 负距离惩罚（符合因果性）
            biases.append(bias)
        alibi_bias = torch.stack(biases, dim=0)  # [num_heads, seq_len, seq_len]

        # 2. 生成平滑过渡权重（仅对长于K的序列生效）
        if seq_len <= self.K:
            return alibi_bias  # 短序列直接用原始ALiBi

        m = torch.arange(seq_len, device=device)
        transition_point = self.K - self.delta
        smooth_weight = torch.sigmoid(self.gamma * (m - transition_point))  # [seq_len]
        # 扩展为[seq_len, seq_len]（每个(i,j)取max(i,j)对应的权重）
        max_ij = torch.max(torch.arange(seq_len, device=device)[:, None], torch.arange(seq_len, device=device)[None, :])
        smooth_weight = smooth_weight[max_ij].unsqueeze(0)  # [1, seq_len, seq_len]

        return alibi_bias * smooth_weight  # [num_heads, seq_len, seq_len]

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[torch.Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        第二步：重写forward方法，插入ALiBi偏置逻辑
        输入输出接口与父类nn.MultiheadAttention完全一致，确保原始代码兼容
        """
        seq_len = query.shape[0]  # query形状：[seq_len, batch_size, embed_dim]
        device = query.device

        # 1. 计算ALiBi偏置（缓存避免重复计算）
        if self.attention_biases is None or self.attention_biases.shape[-1] < seq_len:
            self.attention_biases = self._build_alibi_biases(seq_len=seq_len, device=device)
        alibi_bias = self.attention_biases[:, :seq_len, :seq_len]  # [num_heads, seq_len, seq_len]

        # 2. 调整ALiBi偏置维度，匹配父类注意力分数维度
        # 父类注意力分数维度：[num_heads*batch_size, seq_len, seq_len]
        batch_size = query.shape[1]
        alibi_bias = alibi_bias.repeat_interleave(batch_size, dim=0)  # [num_heads*batch_size, seq_len, seq_len]
        alibi_bias = alibi_bias * self.alpha_scale  # 应用ALiBi强度调度

        # 3. 合并ALiBi偏置与原始注意力掩码（若有）
        if attn_mask is not None:
            # 原始掩码维度：[seq_len, seq_len] → 扩展为[num_heads*batch_size, seq_len, seq_len]
            attn_mask = attn_mask.unsqueeze(0).repeat(self.num_heads * batch_size, 1, 1)
            attn_mask = attn_mask + alibi_bias  # 合并掩码与ALiBi偏置
        else:
            attn_mask = alibi_bias  # 无原始掩码时，直接用ALiBi偏置

        # 4. 调用父类forward方法，传入合并后的掩码（核心：复用父类注意力计算逻辑）
        return super().forward(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal
        )


# 配套的残差注意力块（复用原结构，替换注意力层）
class SmoothResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        # 核心修改：用SmoothALiBiAttention替换nn.MultiheadAttention
        # 初始化参数与原始nn.MultiheadAttention完全一致，不新增参数
        self.attn = SmoothALiBiAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=0.0,  # 原始CLIP默认无dropout
            bias=True,  # 原始CLIP默认有偏置
            # ALiBi参数用默认值（不影响原始接口）
            alibi_start_alpha=1.0,
            K=20,
            delta=0,
            gamma=0.5
        )

        # 以下代码完全复用原始ResidualAttentionBlock逻辑，不修改
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    # 完全复用原始attention方法，不修改
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    # 完全复用原始forward方法，不修改
    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


