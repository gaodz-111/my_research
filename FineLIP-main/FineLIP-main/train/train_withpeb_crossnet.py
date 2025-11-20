import os
import torch
import torch.distributed as dist
from tqdm import tqdm
import sys
import json
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, ReLU
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from textblob import TextBlob
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import clip
import math

# 设置路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'model'))
from model import finelip
from model import cross_net
from model.simple_tokenizer import SimpleTokenizer
from loss import loss_select  # 导入损失函数选择器

sys.path.append("..")
from arguments import get_args
from scheduler import cosine_lr
import subprocess
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import warnings
import wandb
import random

# 设置HF镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings("ignore")


# ----------------------------
# 数据集定义（保留词特征提取）
# ----------------------------
class MultiJSONWordDataset(Dataset):
    def __init__(self, datasets_config, preprocess, word_feat_cache_path):
        self.datasets_config = datasets_config
        self.preprocess = preprocess
        self.word_feat_cache_path = word_feat_cache_path

        # 关键：加载预编码的词向量字典（CPU tensor，不占GPU）
        print(f"加载预编码词向量: {word_feat_cache_path}")
        self.word_feat_dict = torch.load(word_feat_cache_path, map_location='cpu')
        self.clip_dim = next(iter(self.word_feat_dict.values())).shape[0]  # 768，自动获取维度

        self.samples = self._load_all_samples()
        self.simple_tokenizer = SimpleTokenizer()

    def _load_single_dataset(self, data_json_path, image_root):
        samples = []
        json_data = []
        with open(data_json_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_data.append(json.loads(line))

        for item in json_data:
            image_name = item.get("image", "")
            if not image_name:
                continue

            image_path = os.path.join(image_root, image_name)
            if not os.path.exists(image_path):
                print(f"Warning: Image not found - {image_path}")
                continue

            text = item.get("long_text", item.get("text", ""))
            if not text:
                continue

            text = text.replace("\n", " ").strip()
            samples.append((image_path, text))

        return samples

    def _load_all_samples(self):
        all_samples = []
        for cfg in self.datasets_config:
            data_json_path = cfg["data_json_path"]
            image_root = cfg["image_root"]

            if not os.path.exists(data_json_path):
                print(f"Error: JSON file not found - {data_json_path}")
                continue

            dataset_samples = self._load_single_dataset(data_json_path, image_root)
            all_samples.extend(dataset_samples)
            print(f"Loaded {len(dataset_samples)} samples from {data_json_path}")

        return all_samples



    def __getitem__(self, idx):
        image_path, text = self.samples[idx]
        text = text.replace("\n", " ")
        short_text = text.split(". ")[0] if "." in text else text[:100]

        # 1. TextBlob 分词（保留顺序，和之前一致）
        blob = TextBlob(text.lower())  # 转小写，和预编码时一致
        words = [word for word in blob.words if word.isalpha()]
        word_positions = []
        word_feats = []

        for word in words:
            # 词位置计算（和之前一致）
            start_idx = text.lower().find(word)
            if start_idx != -1:
                end_idx = start_idx + len(word)
                word_positions.append({
                    "word": word,
                    "start": start_idx,
                    "end": end_idx,
                    "center": (start_idx + end_idx) / 2
                })

            # 2. 查表获取预编码的词向量（核心修改：不再实时编码）
            if word in self.word_feat_dict:
                # 直接加载预计算的向量（CPU tensor）
                word_feat = self.word_feat_dict[word].clone()
            else:
                # 极端情况：遇到预编码时没见过的词，用零向量填充
                word_feat = torch.zeros(self.clip_dim, dtype=torch.float32)
            word_feats.append(word_feat)
        word_count = len(word_feats)
        # 后续步骤（input_ids、word_to_tokens、图像加载）完全不变
        input_ids = finelip.tokenize([text], truncate=True)[0]
        cap_len = (input_ids != 0).sum().item()
        tokens = [t.replace("</w>", "") for t in self.simple_tokenizer.decode(input_ids.tolist()).split()]
        word_to_tokens = self._align_words_to_tokens(text, tokens, words)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        image_tensor = self.preprocess(image)

        # 返回word_feats（和之前格式一致，collate_fn可直接用）
        return {
            "image_tensor": image_tensor,
            "text": text,
            "short_text": short_text,
            "words": words,
            "word_positions": word_positions,
            "word_to_tokens": word_to_tokens,
            "input_ids": input_ids,
            "cap_len":cap_len,
            "text_length": len(text),
            "img_id": idx,
            "image_path": image_path,
            "word_center_tokens": [wt["center"] for wt in word_to_tokens],
            "word_feats": torch.stack(word_feats) if word_feats else torch.zeros(0, self.clip_dim),
            "word_count": torch.tensor(word_count, dtype=torch.long)
        }

    def _align_words_to_tokens(self, text, tokens, words, token_start_offset=1):
        """
        tokens: list[str]，由 SimpleTokenizer 解码得到（不包含起始/终止 token）
        token_start_offset: CLIP 模型会在开头加 <|startoftext|>，需要加偏移
        """
        word_to_tokens = []
        ptr = 0
        for word in words:
            start = text.lower().find(word.lower(), ptr)
            if start == -1:
                # 找不到词，用 -1 标记
                word_to_tokens.append({"start": -1, "end": -1, "center": -1})
                continue
            end = start + len(word)
            ptr = end

            token_start = None
            token_end = None
            char_pos = 0
            for i, token in enumerate(tokens):
                token_clean = token.replace("</w>", "")
                token_len = len(token_clean)
                if token_start is None and char_pos + token_len > start:
                    token_start = i + token_start_offset
                if char_pos < end:
                    token_end = i + token_start_offset
                char_pos += token_len

            if token_start is None or token_end is None:
                word_to_tokens.append({"start": -1, "end": -1, "center": -1})
            else:
                center_token = (token_start + token_end) // 2
                word_to_tokens.append({
                    "start": token_start,
                    "end": token_end,
                    "center": center_token
                })

        return word_to_tokens

    def __len__(self):
        return len(self.samples)


def custom_collate_fn(batch):
    collated = {
        "image_tensor": torch.stack([item['image_tensor'] for item in batch]),
        "text": [item['text'] for item in batch],
        "short_text": [item['short_text'] for item in batch],
        "words": [item['words'] for item in batch],
        "word_positions": [item['word_positions'] for item in batch],
        "word_to_tokens": [item['word_to_tokens'] for item in batch],  # 新增
        "input_ids": torch.stack([item['input_ids'] for item in batch]),  # 新增
        "cap_lens": torch.tensor([item['cap_len'] for item in batch], dtype=torch.long),
        "text_length": [item['text_length'] for item in batch],
        "img_id": torch.tensor([item['img_id'] for item in batch]),
        "image_path": [item['image_path'] for item in batch],
        "word_count": torch.stack([item["word_count"] for item in batch], dim=0),  # [B]
    }

    # 提取词的中心 token 位置（CLIP token 索引）
    collated["word_center_tokens"] = []
    for item in batch:
        centers = item["word_center_tokens"]
        collated["word_center_tokens"].append(torch.tensor(centers, dtype=torch.long))

    word_feats = []
    for item in batch:
        wf = item['word_feats']
        if wf.numel() == 0:
            wf = torch.zeros(max_words, self.clip_dim)
        word_feats.append(wf)

    collated["word_feats"] = torch.nn.utils.rnn.pad_sequence(
        word_feats, batch_first=True, padding_value=0.0
    )

    return collated


# ----------------------------
# 核心模块定义（基于原始CLIP扩展）
# ----------------------------


class LayerAttention(nn.Module):
    """层注意力模块：学习不同层特征的权重"""

    def __init__(self, dim, num_layers):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        # 压缩特征到1维作为注意力分数
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1)
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(dim)

        # 融合投影
        self.fusion_proj = nn.Linear(dim, dim)

    def forward(self, layer_features):
        """
        Args:
            layer_features: 不同层的特征，形状为 [B, num_layers, N, dim]
        Returns:
            fused_features: 融合后的特征，形状为 [B, N, dim]
        """
        B, num_layers, N, dim = layer_features.shape

        # 关键修改：删除 reshape，直接对空间维度 N 求平均（原代码多做了一次 reshape，完全冗余）
        layer_feat_mean = layer_features.mean(dim=2)  # [B, num_layers, dim]

        # 计算注意力权重（后续逻辑不变）
        attn_scores = self.attention(layer_feat_mean)  # [B, num_layers, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, num_layers, 1]

        # 加权融合各层特征（后续逻辑不变）
        attn_weights = attn_weights.unsqueeze(-1)  # [B, num_layers, 1, 1]
        fused_features = (layer_features * attn_weights).sum(dim=1)  # [B, N, dim]

        fused_features = self.layer_norm(fused_features)
        fused_features = self.fusion_proj(fused_features)

        return fused_features


# class PEB(nn.Module):
#     """
#     位置桥接模块（改进版）
#     - 增强 rope_mapping，增加 BatchNorm 和相对位置编码
#     - 改进 pos_delta_mlp 初始化
#     - 优化 rel_loss 的 target 分布
#     - 位置特征归一化 + 小权重融合
#     """
#     def __init__(self, input_dim=768, hidden_dim=256, output_dim=768,
#                  max_clip_seq_len=248, window_size=3, init_pos_temp=1.0,
#                  use_cosine_residual=True, rel_tau=0.5):
#         super().__init__()
#         assert window_size % 2 == 1, "window_size should be odd"
#         self.max_clip_seq_len = max_clip_seq_len
#         self.window_size = window_size
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#
#         # rope mapping (语义+上下文 -> 位置特征)
#         self.rope_mapping = nn.Sequential(
#             nn.Linear(output_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),  # 稳定分布
#             nn.ReLU6(),                  # 增强非线性
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, output_dim),
#             nn.BatchNorm1d(output_dim)   # 进一步稳定
#         )
#
#         # residual MLP: 预测 delta
#         self.pos_delta_mlp = nn.Sequential(
#             nn.Linear(output_dim, output_dim),
#             nn.BatchNorm1d(output_dim),
#             nn.ReLU6(),
#             nn.Linear(output_dim, output_dim)
#         )
#         # 使用 Kaiming 初始化，避免权重过小
#         nn.init.kaiming_normal_(self.pos_delta_mlp[0].weight, mode='fan_in', nonlinearity='relu')
#         nn.init.kaiming_normal_(self.pos_delta_mlp[3].weight, mode='fan_in', nonlinearity='relu')
#
#         self.clip_pos_emb = nn.Embedding(max_clip_seq_len, output_dim)
#         self.pos_temperature = nn.Parameter(torch.tensor(float(init_pos_temp)))
#         self.ctx_proj = nn.Linear(window_size * input_dim, output_dim)
#         if input_dim != output_dim:
#             self.word_proj = nn.Linear(input_dim, output_dim)
#         else:
#             self.word_proj = nn.Identity()
#
#         self.context_attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=8, batch_first=True)
#         self.context_norm = nn.LayerNorm(output_dim)
#
#         self.use_cosine_residual = use_cosine_residual
#         self.use_legacy_kl = False
#         self.rel_tau = rel_tau
#
#     def _make_soft_target(self, word_pos, sigma=2.0):
#         B, N = word_pos.shape
#         device = word_pos.device
#         L = self.max_clip_seq_len
#         pos_idx = torch.arange(L, device=device).view(1, 1, L).float()
#         wp = word_pos.unsqueeze(-1).float()
#         d2 = (pos_idx - wp) ** 2
#         targ = torch.exp(-d2 / (2.0 * sigma ** 2))
#         targ = targ / (targ.sum(dim=-1, keepdim=True) + 1e-12)
#         return targ
#
#     def forward(self, word_feat, word_pos_248, text_feat_full,
#                 peb_weight=1.0, kl_weight=1.0, rel_weight=0.5, sigma=2.0):
#         B, N, input_dim = word_feat.shape
#         device = word_feat.device
#         L = text_feat_full.shape[1]
#         assert L == self.max_clip_seq_len, "text_feat_full 长度必须为 max_clip_seq_len"
#
#         valid_mask = (word_pos_248 != -1)
#         word_pos_248_valid = word_pos_248.clamp(min=0, max=L - 1)
#
#         # 提取上下文窗口
#         half = self.window_size // 2
#         offsets = torch.arange(-half, half + 1, device=device)
#         positions = word_pos_248_valid.unsqueeze(-1) + offsets.view(1, 1, -1)
#         positions = positions.long().clamp(min=0, max=L - 1)
#
#         D = text_feat_full.shape[-1]
#         expanded_feat = text_feat_full.unsqueeze(1).expand(-1, N, -1, -1)
#         idx = positions.unsqueeze(-1).expand(-1, -1, -1, D)
#         ctx_windows = torch.gather(expanded_feat, dim=2, index=idx)
#         ctx_anchor = ctx_windows.reshape(B, N, -1)
#         ctx_anchor = self.ctx_proj(ctx_anchor)
#         ctx_anchor = ctx_anchor * valid_mask.unsqueeze(-1)
#
#         # 注意力融合
#         q = self.word_proj(word_feat) * valid_mask.unsqueeze(-1)
#         attn_out, _ = self.context_attn(query=q, key=ctx_anchor, value=ctx_anchor)
#         word_feat_with_ctx = self.context_norm(q + attn_out)
#         word_feat_with_ctx = word_feat_with_ctx * valid_mask.unsqueeze(-1)
#
#         # rope mapping -> 位置特征
#         rope_position_features = self.rope_mapping(word_feat_with_ctx.reshape(B*N, -1)).reshape(B, N, -1)
#         rope_norm = F.normalize(rope_position_features, dim=-1)
#
#         clip_ref = self.clip_pos_emb(torch.arange(L, device=device))
#         clip_ref_norm = F.normalize(clip_ref, dim=-1)
#
#         # pos residual loss
#         pos_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
#         kl_loss_legacy = torch.tensor(0.0, device=device, dtype=torch.float32)
#
#         if self.use_legacy_kl:
#             temp = self.pos_temperature.clamp(min=1e-3)
#             logits = torch.matmul(rope_norm, clip_ref_norm.T) / temp
#             log_q = F.log_softmax(logits, dim=-1)
#             target_probs = self._make_soft_target(word_pos_248_valid, sigma=sigma)
#             valid_mask_3d = valid_mask.unsqueeze(-1)
#             kl_loss_legacy = F.kl_div(log_q * valid_mask_3d, target_probs * valid_mask_3d, reduction='sum') / (valid_mask.sum() + 1e-12)
#
#         # 残差位置损失
#         pos_losses = []
#         total_valid = 0
#         for b in range(B):
#             valid_idx = torch.nonzero(valid_mask[b], as_tuple=True)[0]
#             m = valid_idx.numel()
#             if m == 0:
#                 continue
#             total_valid += m
#             rope_feats_b = rope_position_features[b, valid_idx]
#             ref_pos_idx = word_pos_248_valid[b, valid_idx]
#             ref_pos = clip_ref[ref_pos_idx]
#
#             delta = self.pos_delta_mlp(rope_feats_b)
#             pos_pred = ref_pos + delta
#
#             if self.use_cosine_residual:
#                 cos = F.cosine_similarity(pos_pred, ref_pos, dim=-1)
#                 pos_losses.append((1.0 - cos).sum())
#             else:
#                 pos_losses.append(((pos_pred - ref_pos) ** 2).sum(dim=-1).sum())
#
#         if total_valid > 0:
#             pos_loss = torch.stack(pos_losses).sum() / total_valid
#
#         # 相对位置损失
#         wp = word_pos_248_valid.float()
#         diff = wp.unsqueeze(2) - wp.unsqueeze(1)
#         L = text_feat_full.shape[1]
#         frac = diff / float(L)
#         phase = frac * 2.0 * torch.pi
#
#         abs_diff = torch.abs(diff)
#         tau = 0.5 * torch.exp(-2.0 * abs_diff / float(L))  # 更分散的target
#         tau = torch.clamp(tau, min=1e-4)
#
#         target_phase_sim = torch.exp(-(1.0 - torch.cos(phase)) / (2.0 * tau ** 2))
#         pos_sim = torch.matmul(rope_norm, rope_norm.transpose(1, 2))
#         pos_sim01 = (pos_sim + 1.0) / 2.0
#
#         valid_mask_2d = valid_mask.unsqueeze(1) * valid_mask.unsqueeze(2)
#         pred_sim_valid = pos_sim01 * valid_mask_2d
#         target_sim_valid = target_phase_sim * valid_mask_2d
#
#         rel_pos_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
#         if valid_mask_2d.sum() > 0:
#             diff_sim = pred_sim_valid - target_sim_valid
#             huber_loss = torch.where(
#                 torch.abs(diff_sim) < 0.1,
#                 0.5 * diff_sim ** 2,
#                 0.1 * (torch.abs(diff_sim) - 0.05)
#             )
#             rel_pos_loss = huber_loss.sum() / (valid_mask_2d.sum() + 1e-12)
#
#         # 语义保留损失
#         word_feat_norm = F.normalize(word_feat, dim=-1)
#         sem_loss = 1.0 - F.cosine_similarity(rope_norm, word_feat_norm, dim=-1)
#         sem_loss = (sem_loss * valid_mask).sum() / (valid_mask.sum() + 1e-12)
#
#         # 上下文一致性损失
#         ctx_anchor_norm = F.normalize(ctx_anchor, dim=-1)
#         ctx_loss = 1.0 - F.cosine_similarity(rope_norm, ctx_anchor_norm, dim=-1)
#         ctx_loss = (ctx_loss * valid_mask).sum() / (valid_mask.sum() + 1e-12)
#
#         # 总损失
#         peb_loss = (2.0 * pos_loss + 1.5 * rel_pos_loss + 0.5 * sem_loss + 0.3 * ctx_loss) * peb_weight
#
#         # 构建 clip_style_pos_emb
#         clip_style_pos_emb = torch.zeros(B, L, self.output_dim, device=device)
#         if valid_mask.sum() > 0:
#             valid_b, valid_n = torch.nonzero(valid_mask, as_tuple=True)
#             valid_positions = word_pos_248_valid[valid_b, valid_n]
#             rope_feats = rope_position_features[valid_b, valid_n]
#             clip_feats = self.clip_pos_emb(valid_positions)
#
#             # 归一化 + 小权重融合
#             rope_feats = F.normalize(rope_feats, dim=-1)
#             clip_feats = F.normalize(clip_feats, dim=-1)
#             valid_features = rope_feats
#
#             clip_style_pos_emb[valid_b, valid_positions] = valid_features
#
#         losses = {'pos': pos_loss, 'rel': rel_pos_loss, 'sem': sem_loss, 'ctx': ctx_loss}
#         if self.use_legacy_kl:
#             losses['kl_legacy'] = kl_loss_legacy
#
#         return clip_style_pos_emb, peb_loss, losses, rope_position_features.detach()



class PEB(nn.Module):
    """
    位置桥接模块（全局注意力版，兼容中心词融合）
    - 核心改进：用全局文本序列作为注意力的key/value，实现全局上下文交互
    - 新增：支持可选的中心词位置融合（输入center_word_pos=None时不生效）
    """
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=768,
                 max_clip_seq_len=248, window_size=3, init_pos_temp=1.0,
                 use_cosine_residual=True, rel_tau=0.5):
        super().__init__()
        assert window_size % 2 == 1, "window_size should be odd"
        self.max_clip_seq_len = max_clip_seq_len
        self.window_size = window_size
        self.input_dim = input_dim
        self.output_dim = output_dim

        # rope mapping (语义+上下文 -> 位置特征)
        self.rope_mapping = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 稳定分布
            nn.ReLU6(),                  # 增强非线性
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)   # 进一步稳定
        )

        # residual MLP: 预测 delta
        self.pos_delta_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU6(),
            nn.Linear(output_dim, output_dim)
        )
        # 使用 Kaiming 初始化，避免权重过小
        nn.init.kaiming_normal_(self.pos_delta_mlp[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.pos_delta_mlp[3].weight, mode='fan_in', nonlinearity='relu')

        self.clip_pos_emb = nn.Embedding(max_clip_seq_len, output_dim)
        self.pos_temperature = nn.Parameter(torch.tensor(float(init_pos_temp)))
        # 新增：全局文本特征投影层
        self.global_ctx_proj = nn.Linear(input_dim, output_dim)
        if input_dim != output_dim:
            self.word_proj = nn.Linear(input_dim, output_dim)
        else:
            self.word_proj = nn.Identity()

        self.context_attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=8, batch_first=True)
        self.context_norm = nn.LayerNorm(output_dim)
        # 新增：中心词权重（控制中心词融合强度）
        self.center_weight = nn.Parameter(torch.tensor(0.3))

        self.use_cosine_residual = use_cosine_residual
        self.use_legacy_kl = False
        self.rel_tau = rel_tau

    def _make_soft_target(self, word_pos, sigma=2.0):
        B, N = word_pos.shape
        device = word_pos.device
        L = self.max_clip_seq_len
        pos_idx = torch.arange(L, device=device).view(1, 1, L).float()
        wp = word_pos.unsqueeze(-1).float()
        d2 = (pos_idx - wp) ** 2
        targ = torch.exp(-d2 / (2.0 * sigma ** 2))
        targ = targ / (targ.sum(dim=-1, keepdim=True) + 1e-12)
        return targ

    def forward(self, word_feat, word_pos_248, text_feat_full,
                center_word_pos=None,  # 新增：中心词位置（默认None，不融合）
                peb_weight=1.0, kl_weight=1.0, rel_weight=0.5, sigma=2.0):
        B, N, input_dim = word_feat.shape
        device = word_feat.device
        L = text_feat_full.shape[1]
        assert L == self.max_clip_seq_len, "text_feat_full 长度必须为 max_clip_seq_len"

        valid_mask = (word_pos_248 != -1)
        word_pos_248_valid = word_pos_248.clamp(min=0, max=L - 1)

        #######################################################
        # 步骤1：全局文本特征投影 + 中心词特征（若有）
        #######################################################
        # 全局文本序列投影为key/value的维度
        text_feat_proj = self.global_ctx_proj(text_feat_full)
        # 文本有效性mask（假设无padding则全为True，若有需根据实际逻辑生成）
        text_valid_mask = torch.ones(B, L, 1, device=device)
        text_feat_proj = text_feat_proj * text_valid_mask

        # 中心词特征融合（若传入center_word_pos）
        q = self.word_proj(word_feat) * valid_mask.unsqueeze(-1)
        if center_word_pos is not None and torch.any(center_word_pos != -1):
            # 提取每个样本中心词的特征 [B, 1, output_dim]
            center_feats = text_feat_proj.gather(
                dim=1,
                index=center_word_pos.unsqueeze(-1).expand(-1, 1, self.output_dim)
            )
            # 与query融合（通过中心词权重控制强度）
            q = q + center_feats.expand(-1, N, -1) * self.center_weight * valid_mask.unsqueeze(-1)

        #######################################################
        # 步骤2：全局自注意力计算
        #######################################################
        attn_out, _ = self.context_attn(query=q, key=text_feat_proj, value=text_feat_proj)
        word_feat_with_ctx = self.context_norm(q + attn_out)  # 残差连接 + 层归一化
        word_feat_with_ctx = word_feat_with_ctx * valid_mask.unsqueeze(-1)

        #######################################################
        # 步骤3：rope mapping -> 位置特征
        #######################################################
        rope_position_features = self.rope_mapping(word_feat_with_ctx.reshape(B*N, -1)).reshape(B, N, -1)
        rope_norm = F.normalize(rope_position_features, dim=-1)

        clip_ref = self.clip_pos_emb(torch.arange(L, device=device))
        clip_ref_norm = F.normalize(clip_ref, dim=-1)

        #######################################################
        # 步骤4：位置损失、相对位置损失等原有逻辑
        #######################################################
        pos_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        kl_loss_legacy = torch.tensor(0.0, device=device, dtype=torch.float32)



        # 残差位置损失
        pos_losses = []
        total_valid = 0
        for b in range(B):
            valid_idx = torch.nonzero(valid_mask[b], as_tuple=True)[0]
            m = valid_idx.numel()
            if m == 0:
                continue
            total_valid += m
            rope_feats_b = rope_position_features[b, valid_idx]
            ref_pos_idx = word_pos_248_valid[b, valid_idx]
            ref_pos = clip_ref[ref_pos_idx]

            delta = self.pos_delta_mlp(rope_feats_b)
            pos_pred = ref_pos + delta

            if self.use_cosine_residual:
                cos = F.cosine_similarity(pos_pred, ref_pos, dim=-1)
                pos_losses.append((1.0 - cos).sum())
            else:
                pos_losses.append(((pos_pred - ref_pos) ** 2).sum(dim=-1).sum())

        if total_valid > 0:
            pos_loss = torch.stack(pos_losses).sum() / total_valid

        # 相对位置损失
        wp = word_pos_248_valid.float()
        diff = wp.unsqueeze(2) - wp.unsqueeze(1)
        frac = diff / float(L)
        phase = frac * 2.0 * torch.pi

        abs_diff = torch.abs(diff)
        tau = 0.5 * torch.exp(-2.0 * abs_diff / float(L))  # 更分散的target
        tau = torch.clamp(tau, min=1e-4)

        target_phase_sim = torch.exp(-(1.0 - torch.cos(phase)) / (2.0 * tau ** 2))
        pos_sim = torch.matmul(rope_norm, rope_norm.transpose(1, 2))
        pos_sim01 = (pos_sim + 1.0) / 2.0

        valid_mask_2d = valid_mask.unsqueeze(1) * valid_mask.unsqueeze(2)
        pred_sim_valid = pos_sim01 * valid_mask_2d
        target_sim_valid = target_phase_sim * valid_mask_2d

        rel_pos_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        if valid_mask_2d.sum() > 0:
            diff_sim = pred_sim_valid - target_sim_valid
            huber_loss = torch.where(
                torch.abs(diff_sim) < 0.1,
                0.5 * diff_sim ** 2,
                0.1 * (torch.abs(diff_sim) - 0.05)
            )
            rel_pos_loss = huber_loss.sum() / (valid_mask_2d.sum() + 1e-12)

        # 语义保留损失
        word_feat_norm = F.normalize(word_feat, dim=-1)
        sem_loss = 1.0 - F.cosine_similarity(rope_norm, word_feat_norm, dim=-1)
        sem_loss = (sem_loss * valid_mask).sum() / (valid_mask.sum() + 1e-12)

        #######################################################
        # 步骤5：上下文一致性损失（基于全局文本的对应位置）
        #######################################################
        # 从全局投影的文本特征中，提取word_pos_248对应位置的特征
        ctx_anchor = text_feat_proj.gather(
            dim=1,
            index=word_pos_248_valid.unsqueeze(-1).expand(-1, -1, self.output_dim)
        )
        ctx_anchor_norm = F.normalize(ctx_anchor, dim=-1)
        ctx_loss = 1.0 - F.cosine_similarity(rope_norm, ctx_anchor_norm, dim=-1)
        ctx_loss = (ctx_loss * valid_mask).sum() / (valid_mask.sum() + 1e-12)

        #######################################################
        # 步骤6：总损失与clip_style_pos_emb构建
        #######################################################
        # 总损失
        # peb_loss = (2.0 * pos_loss + 1.5 * rel_pos_loss + 0.5 * sem_loss + 0.3 * ctx_loss) * peb_weight


        peb_loss = (2.0 * pos_loss + 0.5 * sem_loss) * peb_weight
        # 构建 clip_style_pos_emb
        clip_style_pos_emb = torch.zeros(B, L, self.output_dim, device=device)
        if valid_mask.sum() > 0:
            valid_b, valid_n = torch.nonzero(valid_mask, as_tuple=True)
            valid_positions = word_pos_248_valid[valid_b, valid_n]
            rope_feats = rope_position_features[valid_b, valid_n]
            clip_feats = self.clip_pos_emb(valid_positions)

            # 归一化 + 小权重融合
            rope_feats = F.normalize(rope_feats, dim=-1)
            clip_feats = F.normalize(clip_feats, dim=-1)
            valid_features = rope_feats

            clip_style_pos_emb[valid_b, valid_positions] = valid_features

        losses = {'pos': pos_loss, 'rel': rel_pos_loss, 'sem': sem_loss, 'ctx': ctx_loss}


        return clip_style_pos_emb, peb_loss, losses, rope_position_features.detach()





class FineGrainedCLIP(nn.Module):
    """基于CLIP扩展细粒度融合功能，使用层注意力融合多层图像特征"""

    def __init__(self, base_clip_model, peb_module, cross_net, criterion=None, num_layers=3):
        super().__init__()
        self.base_clip = base_clip_model
        self.peb = peb_module

        self.cross_net = cross_net
        self.criterion = criterion
        self.clip_dim = self.base_clip.visual.output_dim  # 768（最终输出维度）
        self.hidden_dim = 1024  # 1024（中间层维度）
        self.logit_scale = self.base_clip.logit_scale
        self.atrm_img_has_cls = False
        self.atrm_txt_has_cls = False
        self.patch_num = 257
        # 配置中间层提取（第4、8、12层，对应索引3、7、11）
        self.num_layers = num_layers
        self.selected_layers = [8,12]  # 目标层
        self.intermediate_features = {}  # 存储中间层特征
        self._register_hooks()  # 注册钩子
        self.layer_attention = LayerAttention(
            dim=self.clip_dim,  # 特征维度（768）
            num_layers=self.num_layers  # 层数（如3层）
        )

    def _register_hooks(self):
        """注册钩子提取中间层特征（1024维）"""
        if hasattr(self, 'hooks'):
            for hook in self.hooks:
                hook.remove()

        self.hooks = []
        for layer_idx in self.selected_layers:
            block_idx = layer_idx - 1  # 转换为0基索引
            block = self.base_clip.visual.transformer.resblocks[block_idx]

            # 钩子函数：存储中间层特征（[B, 197, 1024]）
            def hook_fn(model, input, output, layer_idx=layer_idx):
                self.intermediate_features[f"layer{layer_idx}"] = output.detach()

            self.hooks.append(block.register_forward_hook(hook_fn))

    def _get_multi_layer_features(self, images, return_final=True):
        self.intermediate_features.clear()
        # 正常 forward 就行，hook 会自动保存中间层
        img_feat_full = self.base_clip.visual(images)  # 这就是 encode_image_full 的 backbone

        # hook 已经拿到中间层了，不需要再跑一次
        sorted_layers = sorted(
            self.intermediate_features.items(),
            key=lambda x: int(x[0].replace("layer", ""))
        )
        layer_features = [feat for (_, feat) in sorted_layers]
        self.intermediate_features.clear()
        processed_layers = []
        for feat in layer_features:
            # [patch_num, B, hidden_dim] -> [B, patch_num, hidden_dim]
            feat = feat.transpose(0, 1)
            patch_feat = feat[:, 1:, :]  # 去掉 CLS

            B_feat, N_feat, C_feat = patch_feat.shape
            patch_feat_flat = patch_feat.reshape(-1, C_feat)

            # 投影到 output_dim
            if isinstance(self.base_clip.visual.proj, torch.nn.Parameter):
                patch_feat_proj = patch_feat_flat @ self.base_clip.visual.proj
            else:
                patch_feat_proj = self.base_clip.visual.proj(patch_feat_flat)

            patch_feat_proj = patch_feat_proj.reshape(B_feat, N_feat, -1)
            processed_layers.append(patch_feat_proj.unsqueeze(1))

        multi_layer_feat = torch.cat(processed_layers, dim=1)
        fused_feat = self.layer_attention(multi_layer_feat)

        if return_final:
            # 返回完整的 final 特征序列（等价于 encode_image_full）
            final_feat = self.base_clip.encode_image_full(images)
            return fused_feat, final_feat
        else:
            return fused_feat

    def encode_image_for_crossnet(self, images):
        """输出cross_net兼容的图像序列特征（含CLS）"""
        img_fine_feat, img_feat_full = self._get_multi_layer_features(images, return_final=True)
        img_patch_feat = img_feat_full[:, 1:, :]  # 去掉CLS的patch特征
        fused_img_patch = img_patch_feat + 0.2 * img_fine_feat  # 融合多层特征
        # 拼接CLS（cross_net会自动判断has_cls_token）
        img_embs = torch.cat([img_feat_full[:, :1, :], fused_img_patch], dim=1)  # [B, 197, 768]
        return img_embs

    def encode_text_for_crossnet(self, texts, clip_style_pos_emb=None, cap_lengths=None):
        text_feat_raw = self.base_clip.encode_text_full(texts)  # [B, 248, D]（CLIP 原特征，不改动）
        B, L, D = text_feat_raw.shape

        if clip_style_pos_emb is not None and cap_lengths is not None:
            cap_lengths = cap_lengths.to(text_feat_raw.device)
            valid_len = torch.clamp(cap_lengths, min=2)  # 至少 CLS + SEP
            text_body_with_pos = []

            for i in range(B):
                vl = valid_len[i].item()
                body_len = vl - 2  # 正文长度（去掉 CLS 和 SEP）

                if body_len > 0:
                    # --------------------------
                    # 1. 提取 CLIP 原正文特征（不归一化，保留预训练分布）
                    # --------------------------
                    text_body = text_feat_raw[i, 1:vl - 1, :]  # [body_len, D]（CLIP 原特征）

                    # --------------------------
                    # 2. 处理 PEB 位置特征：归一化到和 CLIP 正文特征幅度一致
                    # --------------------------
                    pos_emb_body = clip_style_pos_emb[i, 1:vl - 1, :]  # [body_len, D]（PEB 位置特征）
                    # 计算 CLIP 正文特征的平均 L2 范数（幅度基准）
                    text_body_norm = text_body.norm(dim=-1, keepdim=True).mean()  # 标量，每个样本的平均幅度
                    # 归一化 PEB 位置特征到该幅度（避免幅度不匹配）
                    pos_emb_body = F.normalize(pos_emb_body, dim=-1) * text_body_norm  # [body_len, D]

                    # --------------------------
                    # 3. 融合：小权重叠加 PEB 位置特征（不破坏 CLIP 原语义）

                    body_with_pos = text_body + 0.1 * pos_emb_body  # [body_len, D]

                    text_body_with_pos.append(body_with_pos)
                else:
                    text_body_with_pos.append(torch.empty(0, D, device=text_feat_raw.device))

            # 拼接回完整序列（CLS 和 SEP 用 CLIP 原特征，不改动）
            cap_embs = []
            for i in range(B):
                vl = valid_len[i].item()
                # CLS 和 SEP 直接用 CLIP 原始特征，不做任何修改
                cls_emb = text_feat_raw[i:i + 1, :1, :]  # [1, 1, D]（CLIP 原 CLS）
                sep_emb = text_feat_raw[i:i + 1, vl - 1:vl, :]  # [1, 1, D]（CLIP 原 SEP）
                body_emb = text_body_with_pos[i].unsqueeze(0)  # [1, body_len, D]（融合后正文）
                # 补 PAD（用零向量，和 CLIP 原 PAD 一致）
                pad_len = L - cls_emb.size(1) - body_emb.size(1) - sep_emb.size(1)
                pad_emb = torch.zeros(1, pad_len, D, device=text_feat_raw.device)

                cap_embs.append(torch.cat([cls_emb, body_emb, sep_emb, pad_emb], dim=1))
            cap_embs = torch.cat(cap_embs, dim=0)
        else:
            cap_embs = text_feat_raw  # 无 PEB 时，完全用 CLIP 原特征

        return cap_embs


    def forward(self, images, texts, img_ids, warmup_alpha, local_rank,
                word_center_tokens=None, word_feats=None, word_count=None, cap_lens=None):
        B = images.shape[0]
        device = images.device
        peb_loss = torch.tensor(0.0, device=device)
        losses = {}

        # 1. 图像特征：层注意力融合后输出cross_net格式
        img_embs = self.encode_image_for_crossnet(images)  # [B, 197, 768]（含CLS）

        # 2. 文本特征：PEB补全后输出cross_net格式
        text_feat_full = self.base_clip.encode_text_full(texts)
        if word_center_tokens is not None and word_feats is not None and word_count is not None:
            # PEB位置补全（不变）
            word_feats = word_feats.to(device)
            word_count = word_count.to(device)
            # 处理word_pos_248（文本词对应CLIP token位置）
            word_pos_248_padded = []
            for centers in word_center_tokens:
                pos_248 = torch.tensor(centers, dtype=torch.long, device=device)
                word_pos_248_padded.append(pos_248)
            max_words = max([wp.size(0) for wp in word_pos_248_padded]) if word_pos_248_padded else 0
            word_pos_248_padded = torch.stack([
                torch.nn.functional.pad(wp, (0, max_words - wp.size(0)), mode='constant', value=-1)
                for wp in word_pos_248_padded
            ]).to(device)
            # 调用PEB
            clip_style_pos_emb, peb_loss, losses, rope_position_features = self.peb(
                word_feats, word_pos_248_padded, text_feat_full
            )
        else:
            clip_style_pos_emb = None
        # 文本特征最终格式
        cap_embs = self.encode_text_for_crossnet(texts, clip_style_pos_emb=clip_style_pos_emb,cap_lengths=cap_lens)  # [B, 248, 768]

        # 3. cross_net计算图文相似度矩阵（核心替换）
        improve_sims, _ = self.cross_net.forward_dual_aggr(img_embs, cap_embs, cap_lens)

        # 4. 损失计算：基于cross_net的相似度矩阵
        contrast_loss = self.criterion(img_embs, cap_embs, img_ids, improve_sims) * warmup_alpha  # 对比损失
        total_loss = contrast_loss + 0.5 * peb_loss  # 总损失=对比损失+PEB损失

        return contrast_loss, peb_loss, total_loss, losses


# ----------------------------
# 训练相关函数
# ----------------------------
def START_SEED(seed=71):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def push_to_s3(local_path, s3_path):
    command = f"aws s3 cp {local_path} {s3_path}"
    subprocess.run(command, shell=True)


def setup_distributed(backend="nccl", port=None):
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29522"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(device=f'cuda:{rank % num_gpus}')
    return rank % num_gpus


def get_embed_size(vit_variant: str) -> int:
    vit_variant = vit_variant.lower()
    if "bigg" in vit_variant:
        return 1280
    elif "l" in vit_variant:
        return 768
    elif "b" in vit_variant:
        return 512
    else:
        raise ValueError(f"Unknown ViT variant: {vit_variant}")


# ----------------------------
# 训练类（集成损失函数选择）
# ----------------------------
class CLIP_FineGrained_Train():
    def __init__(self, args, local_rank=0):
        self.args = args
        self.local_rank = local_rank
        self.exp_name = args.exp_name
        self.base_model = args.base_model

        # 1. 加载基础模型
        self.base_clip, self.clip_preprocess = finelip.load_from_clip(
            self.base_model,
            device='cpu',
            run_finelip=not self.args.run_baseline,
        )
        self.base_clip = self.base_clip.cuda()
        self.clip_dim = get_embed_size(vit_variant=self.base_model)

        if not self.args.run_baseline:
            self.cross_net = cross_net.CrossSparseAggrNet_v2(opt=self.args)  # 初始化cross_net
        else:
            self.cross_net = None

        # 2. 初始化新增模块（移除了图像细粒度提取器）
        self.peb_module = PEB(
            input_dim=self.clip_dim, hidden_dim=256, output_dim=self.clip_dim
        ).cuda()



        # 3. 初始化损失函数
        self.criterion = loss_select(opt=args, loss_type=args.loss_finegrain)

        # 4. 构建完整模型（不再传入img_fine_module）
        self.model = FineGrainedCLIP(
            base_clip_model=self.base_clip,
            peb_module=self.peb_module,
            cross_net=self.cross_net,
            criterion=self.criterion,
            num_layers=3
        ).cuda()
        # 5. 训练配置

        self.batch_size = args.global_batch_size // torch.cuda.device_count()
        self.accumulation_steps = 512 // args.global_batch_size
        self.num_epoch = args.epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.warmup_length = args.warmup_length
        self.logdir = f"experiments/{self.exp_name}"
        self.ckptdir = self.logdir + "/ckpt"
        os.makedirs(self.ckptdir, exist_ok=True)

        # 6. 日志配置
        if self.local_rank == 0:
            hyperparameter_defaults = {
                "weight_decay": args.weight_decay,
                "warmup_length": args.warmup_length,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "num_epoch": self.num_epoch,
                "loss_type": args.loss_finegrain,
            }
            if args.enable_wandb:
                wandb.tensorboard.patch(root_logdir=self.logdir)
                wandb.init(
                    config=hyperparameter_defaults,
                    project="FineGrainedCLIP",
                    sync_tensorboard=True,
                    save_code=True,
                    name=self.exp_name
                )
        self.writer = SummaryWriter(self.logdir)

        # 7. DDP和优化器
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
        self.optimizer = self.create_optimizer()
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()

    def create_optimizer(self):
        """优化器参数分组：基础CLIP参数 + cross_net+PEB+层注意力参数"""
        finelip_params = []
        cross_net_params = []
        for n, p in self.model.named_parameters():
            if any(nd in n for nd in ["cross_net", "peb", "layer_attention","criterion"]):
                # cross_net、PEB、层注意力用更高学习率
                cross_net_params.append(p)
            else:
                # 基础CLIP参数用基础学习率
                finelip_params.append(p)
        param_groups = [
            {'params': finelip_params, 'lr': self.lr},
            {'params': cross_net_params, 'lr': self.args.cross_net_lr}  # cross_net学习率单独设置
        ]
        return optim.AdamW(param_groups, weight_decay=self.weight_decay)

    def resume_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        checkpoint = torch.load(checkpoint_path.replace('.pt', '_other.pt'), map_location='cpu')

        self.model.module.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        return checkpoint['epoch']

    def save_checkpoint(self, epoch):
        if self.base_model == "ViT-B/16":
            name = 'fineclip-B.pt'
        elif self.base_model == "ViT-L/14":
            name = 'fineclip-L.pt'
        else:
            name = "fineclip-others.pt"

        experiment_name = f'{self.ckptdir}/{self.exp_name}_{self.args.global_batch_size}_epoch_{epoch + 1}_{name}'
        state_dict = self.model.module.state_dict()
        other_state_dict = {
            'epoch': epoch + 1,
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(state_dict, experiment_name)
        torch.save(other_state_dict, experiment_name.replace('.pt', '_other.pt'))
        if self.args.s3_bucket is not None:
            push_to_s3(experiment_name, self.args.s3_bucket)
        print(f"saved model to {experiment_name}")

    def train_epoch(self, dataloader, epoch, start_iter=0):
        running_loss = 0.0
        self.model.train()
        num_batches_per_epoch = len(dataloader)
        self.optimizer.zero_grad()

        for i, batch in enumerate(tqdm(dataloader, disable=(self.local_rank != 0))):
            step = num_batches_per_epoch * epoch + i
            if step < start_iter:
                continue

            # 准备数据：新增cap_lens
            images = batch["image_tensor"].cuda(non_blocking=True)
            texts_raw = batch["text"]
            texts = finelip.tokenize(texts_raw, truncate=True).cuda(non_blocking=True)
            img_ids = batch["img_id"].cuda(non_blocking=True)
            cap_lens = batch["cap_lens"].cuda(non_blocking=True)  # 传入cross_net
            word_center_tokens = batch["word_center_tokens"]
            word_feats = batch["word_feats"].cuda(non_blocking=True)
            word_count = batch["word_count"].cuda(non_blocking=True)

            # 预热系数（控制新增模块权重）
            warmup_alpha = (
                float(i) / num_batches_per_epoch
                if epoch < self.args.embedding_warmup_epochs
                else 1.0
            )

            # 前向传播：传入cap_lens
            with torch.cuda.amp.autocast():
                contrast_loss, peb_loss, total_loss, losses = self.model(
                    images=images,
                    texts=texts,
                    img_ids=img_ids,
                    warmup_alpha=warmup_alpha,
                    local_rank=self.local_rank,
                    word_center_tokens=word_center_tokens,
                    word_feats=word_feats,
                    word_count=word_count,
                    cap_lens=cap_lens  # 关键：给cross_net传文本长度
                )

            # 异常损失处理
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: Abnormal loss at step {step}, skipping")
                continue

            # 梯度累积
            loss = total_loss / self.accumulation_steps
            self.scaler.scale(loss).backward()
            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler(step)

            # 损失累积与同步
            running_loss += total_loss.item()
            dist.all_reduce(total_loss)
            avg_loss = total_loss.item() / dist.get_world_size()




            # 日志记录
            if step % 200 == 0 and self.local_rank == 0:

                # 学习率日志
                for idx, param_group in enumerate(self.optimizer.param_groups):
                    current_lr = param_group['lr']
                    print(f"step {step} lr_{idx}: {current_lr:.6f}")
                    self.writer.add_scalar(f"hyper/lr_{idx}", current_lr, step)
                # 损失日志
                print(f"step {step} contrast_loss: {contrast_loss.item():.4f}")
                print(f"step {step} peb_loss: {peb_loss.item():.4f}")
                print(f"step {step} avg_total_loss: {avg_loss:.4f}")
                print(f"step {step} losses: {losses}")
                self.writer.add_scalar("Loss/contrast", contrast_loss.item(), step)
                self.writer.add_scalar("Loss/peb", peb_loss.item(), step)
                self.writer.add_scalar("Loss/total_avg", avg_loss, step)
                print("=====================================")

        return running_loss / num_batches_per_epoch

    def train(self, resume=False):
        # 1. 数据集配置
        datasets_config = [
            {"data_json_path": "/data2/gaodz/Re-Align/hypernet_train_data_short_core.json",
             "image_root": "/data2/gaodz/OmniConsistency"},
            {"data_json_path": "/data2/gaodz/Re-Align/COCO_short_core_1.json",
             "image_root": "/data2/gaodz/train2014"},
            {"data_json_path": "/data2/gaodz/WikiArt/OpenDataLab___WikiArt/raw/train_txt/image_text_new.json",
             "image_root": "/data2/gaodz/WikiArt/OpenDataLab___WikiArt/raw/train_image/wikiart"},
            {"data_json_path": "/data2/gaodz/sharegpt4v/sharegpt4v_coco.json",
             "image_root": "/data2/gaodz/coco2017/PAI/COCO2017"},
        ]
        word_feat_cache_path = "/data2/gaodz/FineLIP-main/word_feat_cache.pt"
        # 2. 初始化数据集和加载器
        trainset = MultiJSONWordDataset(
            datasets_config=datasets_config,
            preprocess=self.clip_preprocess,
            word_feat_cache_path=word_feat_cache_path
        )
        train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
        train_loader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )

        # 3. 学习率调度器
        lrs = [p["lr"] for p in self.optimizer.param_groups]
        total_steps = (self.num_epoch * len(train_loader)) // self.accumulation_steps
        self.scheduler = cosine_lr(
            self.optimizer,
            base_lrs=lrs,
            warmup_length=self.warmup_length,
            steps=total_steps
        )

        # 4. 恢复训练
        start_epoch = 0
        resume_iter = 0
        if resume and self.args.resume_path is not None:
            start_epoch = self.resume_checkpoint(self.args.resume_path)
            print(f"从 epoch {start_epoch} 恢复训练")

        # 5. 训练循环
        for epoch in range(start_epoch, self.num_epoch):
            train_sampler.set_epoch(epoch)  # 确保每个epoch的采样不同
            epoch_loss = self.train_epoch(train_loader, epoch, start_iter=resume_iter)

            # 打印epoch平均损失
            if self.local_rank == 0:
                print(f"Epoch {epoch + 1}/{self.num_epoch} 平均损失: {epoch_loss:.4f}")
                self.writer.add_scalar("Loss/epoch_avg", epoch_loss, epoch)
                # 保存模型
                self.save_checkpoint(epoch)


if __name__ == "__main__":
    parser = get_args()
    # 新增ATRM相关参数（需要在get_args()中定义，这里补充示例）
    # parser.add_argument('--atrm_sparse_ratio', type=float, default=0.5, help='ATRM稀疏筛选比例')
    # parser.add_argument('--atrm_aggr_ratio', type=float, default=0.4, help='ATRM聚合比例')
    # parser.add_argument('--atrm_dim_ratio', type=float, default=0.2, help='ATRM低维投影比例')
    # parser.add_argument('--embedding_warmup_epochs', type=int, default=1, help='新增模块预热轮数')
    # parser.add_argument('--enable_wandb', action='store_true', help='是否启用wandb日志')

    args = parser.parse_args()
    START_SEED(args.seed)

    local_rank = setup_distributed()
    print("DDP初始化完成")
    if local_rank == 0:
        print(f"参数: {args}")

    trainer = CLIP_FineGrained_Train(
        args=args,
        local_rank=local_rank
    )
    trainer.train(resume=(args.resume_path is not None))
    torch.distributed.destroy_process_group()