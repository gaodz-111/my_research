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

    def _extract_word_features(self, text):
        """使用TextBlob提取词及位置信息"""
        blob = TextBlob(text)
        words = list(set([word.lower() for word in blob.words if word.isalpha()]))

        word_positions = []
        word_token_ids = []
        for word in words:
            start_idx = text.lower().find(word)
            if start_idx != -1:
                end_idx = start_idx + len(word)
                word_positions.append({
                    "word": word,
                    "start": start_idx,
                    "end": end_idx,
                    "center": (start_idx + end_idx) / 2
                })
            word_tokens = finelip.tokenize([word], truncate=True)  # [1, 248]
            word_token_ids.append(word_tokens)  # 存每个词的 token 序列
        return {
            "words": words,
            "word_positions": word_positions,
            "raw_text": text,
            "text_length": len(text),
            "word_token_ids": word_token_ids
        }

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

        # 压缩空间维度，计算每一层的重要性
        layer_feat_reshaped = layer_features.reshape(B, num_layers, -1, dim)  # [B, num_layers, N, dim]
        layer_feat_mean = layer_feat_reshaped.mean(dim=2)  # [B, num_layers, dim]

        # 计算注意力权重
        attn_scores = self.attention(layer_feat_mean)  # [B, num_layers, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, num_layers, 1]

        # 加权融合各层特征
        attn_weights = attn_weights.unsqueeze(-1)  # [B, num_layers, 1, 1]
        fused_features = (layer_features * attn_weights).sum(dim=1)  # [B, N, dim]

        # 归一化和投影
        fused_features = self.layer_norm(fused_features)
        fused_features = self.fusion_proj(fused_features)

        return fused_features


class PEB(nn.Module):
    """位置桥接模块（修正版）
    变化要点：
      - self.ctx_proj 在 __init__ 中创建并会被 optimizer 更新
      - 向量化窗口提取，避免 python for-loop
      - word_feat 投影到 output_dim 后做 attention（保持 dim 对齐）
      - 使用平滑的高斯 target 代替 one-hot（sigma 可调）
      - pos_temperature 初始为 1.0（避免过尖锐 logits）
      - pos_sim 映射到 [0,1] 再与 1-distance 做 MSE
    返回：
      clip_style_pos_emb: [B, num_words, output_dim]
      peb_loss: 标量 tensor（可直接乘以 peb_weight）
      losses: dict { 'kl': kl_loss, 'rel': rel_pos_loss } （便于日志）
    """
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=768,
                 max_clip_seq_len=248, window_size=3, init_pos_temp=1.0):
        super().__init__()
        assert window_size % 2 == 1, "window_size should be odd"
        self.max_clip_seq_len = max_clip_seq_len
        self.window_size = window_size
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 把 rope_mapping 的输入维度改为 output_dim（因为我们在 attention 后输入）
        self.rope_mapping = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

        self.clip_pos_emb = nn.Embedding(max_clip_seq_len, output_dim)
        # pos_temperature 用来平滑 logits；初始设为 1.0（不是 0.07）
        self.pos_temperature = nn.Parameter(torch.tensor(float(init_pos_temp)))
        # 将上下文拼接投影到 output_dim （可学习）
        self.ctx_proj = nn.Linear(self.window_size * input_dim, output_dim)
        # 如果 word_feat 的 dim != output_dim，需要一个投影
        if input_dim != output_dim:
            self.word_proj = nn.Linear(input_dim, output_dim)
        else:
            self.word_proj = nn.Identity()

        # 上下文注意力层
        self.context_attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=8, batch_first=True)
        self.context_norm = nn.LayerNorm(output_dim)

    def _make_soft_target(self, word_pos, sigma=2.0):
        # word_pos: [B, N] long
        # 输出 target_probs: [B, N, L]
        B, N = word_pos.shape
        device = word_pos.device
        L = self.max_clip_seq_len
        # pos_idx: [1,1,L]
        pos_idx = torch.arange(L, device=device).view(1, 1, L).float()
        wp = word_pos.unsqueeze(-1).float()  # [B,N,1]
        d2 = (pos_idx - wp) ** 2
        targ = torch.exp(- d2 / (2.0 * (sigma ** 2)))
        targ = targ / (targ.sum(dim=-1, keepdim=True) + 1e-12)
        return targ  # [B,N,L]

    def forward(self, word_feat, word_pos_248, text_feat_full,
                peb_weight=1.0, kl_weight=1.0, rel_weight=0.5, sigma=2.0):
        """
        Args:
          word_feat: [B, N, input_dim]  (原始 clip 词向量)
          word_pos_248: [B, N] (long) 每个词对应 CLIP 序列中的 index（有效：0~247，无效：-1）
          text_feat_full: [B, L, input_dim]  CLIP 完整序列特征（L=248）
          peb_weight: 全局缩放 peb loss（建议做 warmup）
          kl_weight, rel_weight: 子项权重
          sigma: soft target 的高斯宽度（越大越平滑）
        Returns:
          clip_style_pos_emb: [B, 248, output_dim] 对齐到 248 token 序列的位置特征
          peb_loss: 标量 tensor （仅有效位置贡献）
          losses_dict: dict { 'kl': kl_loss, 'rel': rel_pos_loss }
        """
        B, N, input_dim = word_feat.shape
        device = word_feat.device
        L = text_feat_full.shape[1]
        assert L == self.max_clip_seq_len == 248, "text_feat_full 必须是 248 长度"

        # ---------------- 1. 处理 -1 无效位置：创建 valid_mask ----------------
        valid_mask = (word_pos_248 != -1)  # [B, N]：True=有效词，False=无效（-1）
        # 把无效位置的 pos 强制设为 0（后续用 mask 屏蔽，避免越界）
        word_pos_248_valid = word_pos_248.clamp(min=0, max=L - 1)  # [B, N]

        # ---------------- 2. 向量化窗口提取（仅有效位置参与） ----------------
        half = self.window_size // 2
        offsets = torch.arange(-half, half + 1, device=device)  # [W]（如 window=3 时为 [-1,0,1]）
        # 计算每个词的窗口位置：[B, N, W]
        positions = word_pos_248_valid.unsqueeze(-1) + offsets.view(1, 1, -1)  # 广播到 batch
        positions = positions.long().clamp(min=0, max=L - 1)  # 确保窗口位置在 0~247 内

        # 提取窗口特征：[B, N, W, D]
        D = text_feat_full.shape[-1]
        expanded_feat = text_feat_full.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, L, D]
        idx = positions.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, N, W, D]（索引对齐特征维度）
        ctx_windows = torch.gather(expanded_feat, dim=2, index=idx)  # [B, N, W, D]

        # 压缩窗口并投影到 output_dim：[B, N, output_dim]
        ctx_anchor = ctx_windows.reshape(B, N, -1)  # [B, N, W*D]
        ctx_anchor = self.ctx_proj(ctx_anchor)  # [B, N, output_dim]

        # 屏蔽无效位置的 ctx_anchor（无效位置设为 0，不参与注意力）
        ctx_anchor = ctx_anchor * valid_mask.unsqueeze(-1)  # [B, N, output_dim]

        # ---------------- 3. 注意力融合（仅有效位置贡献） ----------------
        # 词特征投影到 output_dim（与 ctx_anchor 维度一致）
        q = self.word_proj(word_feat)  # [B, N, output_dim]
        # 屏蔽无效位置的 query（避免无效词参与注意力计算）
        q = q * valid_mask.unsqueeze(-1)  # [B, N, output_dim]

        # 多头注意力：Q=词特征，K/V=上下文锚点
        attn_out, _ = self.context_attn(query=q, key=ctx_anchor, value=ctx_anchor)
        # 残差连接 + LayerNorm：融合上下文的词特征
        word_feat_with_ctx = self.context_norm(q + attn_out)  # [B, N, output_dim]

        # 再次屏蔽无效位置（确保无效位置不参与后续位置特征提取）
        word_feat_with_ctx = word_feat_with_ctx * valid_mask.unsqueeze(-1)  # [B, N, output_dim]

        # ---------------- 4. 提取位置特征（仅有效位置有意义） ----------------
        rope_position_features = self.rope_mapping(word_feat_with_ctx)  # [B, N, output_dim]
        rope_norm = F.normalize(rope_position_features, dim=-1)  # [B, N, output_dim]

        # ---------------- 5. KL 损失（仅计算有效位置） ----------------
        # CLIP 原始位置嵌入（参考）：[L, output_dim]
        clip_ref = self.clip_pos_emb(torch.arange(L, device=device))
        clip_ref_norm = F.normalize(clip_ref, dim=-1)  # [L, output_dim]

        # 计算 logits：位置特征与 CLIP 位置嵌入的相似度
        temp = self.pos_temperature.clamp(min=1e-3)  # 温度参数（避免 logits 过尖）
        logits = torch.matmul(rope_norm, clip_ref_norm.T) / temp  # [B, N, L]
        log_q = F.log_softmax(logits, dim=-1)  # [B, N, L]（对每个词的位置分布做 softmax）

        # 生成软目标（基于有效位置）：[B, N, L]
        target_probs = self._make_soft_target(word_pos_248_valid, sigma=sigma)  # [B, N, L]

        # 屏蔽无效位置的损失贡献（仅有效位置参与 KL 计算）
        valid_mask_3d = valid_mask.unsqueeze(-1)  # [B, N, 1]
        log_q_valid = log_q * valid_mask_3d
        target_probs_valid = target_probs * valid_mask_3d

        # 计算 KL 散度（仅有效位置的平均，避免无效位置干扰）
        kl_loss = F.kl_div(
            log_q_valid, target_probs_valid,
            reduction='sum'  # 先求和，再除以有效位置数
        ) / (valid_mask.sum() + 1e-12)  # 防止除以 0

        # ---------------- 6. 相对位置损失（仅计算有效位置） ----------------
        # 计算词之间的位置相似度：[B, N, N]
        pos_sim = torch.matmul(rope_norm, rope_norm.transpose(1, 2))  # [-1, 1]
        pos_sim01 = (pos_sim + 1.0) / 2.0  # 归一化到 [0, 1]

        # 计算词之间的真实位置距离：[B, N, N]
        wp_float = word_pos_248_valid.float()  # [B, N]
        diff = wp_float.unsqueeze(2) - wp_float.unsqueeze(1)  # [B, N, N]（每个词对的位置差）
        word_pos_dist = torch.abs(diff) / float(L)  # 归一化到 [0, 1]
        target_sim = 1.0 - word_pos_dist  # 距离越近，目标相似度越高

        # 屏蔽无效位置的损失贡献（无效词之间的相似度不计入）
        valid_mask_2d = valid_mask.unsqueeze(1) * valid_mask.unsqueeze(2)  # [B, N, N]
        pos_sim01_valid = pos_sim01 * valid_mask_2d
        target_sim_valid = target_sim * valid_mask_2d

        # 计算 MSE 损失（仅有效词对）
        rel_pos_loss = F.mse_loss(
            pos_sim01_valid, target_sim_valid,
            reduction='sum'
        ) / (valid_mask_2d.sum() + 1e-12)  # 防止除以 0

        # ---------------- 7. 总 PEB 损失 ----------------
        peb_loss = (kl_weight * kl_loss + rel_weight * rel_pos_loss) * peb_weight

        # ---------------- 8. 构建对齐到 248 序列的位置特征 ----------------
        # 初始化 [B, 248, output_dim] 的全零张量（无效位置默认 0）
        clip_style_pos_emb = torch.zeros(B, L, self.output_dim, device=device)
        # 遍历每个样本，把有效词的位置特征放到正确的 token 位置
        for b in range(B):
            # 找到当前样本的有效词索引和对应 token 位置
            valid_indices = torch.nonzero(valid_mask[b], as_tuple=True)[0]  # [M]（M=当前样本有效词数）
            if len(valid_indices) == 0:
                continue  # 无有效词，跳过
            # 有效词的 token 位置：[M]
            valid_positions = word_pos_248_valid[b, valid_indices]
            # 有效词的位置特征：[M, output_dim]
            valid_features = 0.8 * self.clip_pos_emb(valid_positions) + 0.2 * rope_position_features[b, valid_indices]
            # 把特征填充到 248 序列的对应位置
            clip_style_pos_emb[b, valid_positions] = valid_features

        # ---------------- 9. 输出 ----------------
        losses = {'kl': kl_loss, 'rel': rel_pos_loss}
        return clip_style_pos_emb, peb_loss, losses


class TokenSparse(nn.Module):
    """复用原代码：稀疏筛选关键token，融合冗余token"""

    def __init__(self, embed_dim=512, sparse_ratio=0.6):
        super().__init__()
        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio

    def forward(self, tokens, attention_x, attention_y):
        B_v, L_v, C = tokens.size()
        score = attention_x + attention_y  # 融合自注意力和交叉注意力分数
        num_keep_token = math.ceil(L_v * self.sparse_ratio)

        # 筛选top-k关键token
        score_sort, score_index = torch.sort(score, dim=1, descending=True)
        keep_policy = score_index[:, :num_keep_token]
        select_tokens = torch.gather(tokens, dim=1, index=keep_policy.unsqueeze(-1).expand(-1, -1, C))

        # 融合冗余token（信息完全分配）
        non_keep_policy = score_index[:, num_keep_token:]
        non_tokens = torch.gather(tokens, dim=1, index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C))
        non_keep_score = F.softmax(score_sort[:, num_keep_token:], dim=1).unsqueeze(-1)
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True)

        return select_tokens, extra_token


class TokenAggregation(nn.Module):
    """复用原代码：动态聚合token，生成精炼特征"""

    def __init__(self, dim=512, keeped_patches=64, dim_ratio=0.2):
        super().__init__()
        hidden_dim = int(dim * dim_ratio)
        self.weight = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),  # 低维投影
            nn.GELU(),
            nn.Linear(hidden_dim, keeped_patches)  # 聚合权重生成
        )
        self.scale = nn.Parameter(torch.ones(1, 1, 1))  # 类似温度参数的缩放因子

    def forward(self, x):
        weight = self.weight(x)  # [B, N', keeped_patches]
        weight = weight.transpose(2, 1) * self.scale  # [B, keeped_patches, N']
        weight = F.softmax(weight, dim=2)  # 列和为1，确保信息完全分配
        return torch.bmm(weight, x)  # [B, keeped_patches, C]


class ATRM(nn.Module):
    """
    复用CrossSparseAggrNet逻辑的Adaptive Token Refinement Module
    功能：从原始细粒度token中动态精炼出关键特征
    """

    def __init__(
            self,
            dim=512,
            sparse_ratio=0.5,  # 筛选关键token的比例（如0.5保留一半）
            aggr_ratio=0.4,  # 聚合后token数量的比例（相对于原始token）
            dim_ratio=0.2,  # 低维投影维度比例（d_k = dim * dim_ratio）
            has_cls_token=False  # 输入是否包含[CLS] token
    ):
        super().__init__()
        self.dim = dim
        self.has_cls_token = has_cls_token

        # 计算最终精炼token数量：原始token数 × 稀疏比例 × 聚合比例
        # （假设原始token数为196，则196×0.5×0.4≈39，与论文一致）
        self.keeped_patches = int(256 * sparse_ratio * aggr_ratio)  # 196为默认图像patch数

        # 复用原代码的稀疏筛选和聚合模块
        self.sparse_net = TokenSparse(embed_dim=dim, sparse_ratio=sparse_ratio)
        self.aggr_net = TokenAggregation(
            dim=dim,
            keeped_patches=self.keeped_patches,
            dim_ratio=dim_ratio
        )

    def forward(self, x, atrm_type=None,cross_attention=None):
        """
        Args:
            x: 细粒度特征 [B, N, dim]（N为原始token数，如196、248）
               若has_cls_token=True，则x[:,0,:]为[CLS] token
            cross_attention: 跨模态注意力分数 [B, N]（如文本对图像的注意力）
                             若为None，则仅用自注意力筛选
        Returns:
            refined_feat: 精炼后的特征 [B, N_refined, dim]
        """
        B, N, C = x.shape
        if atrm_type == 'image':
            # 分离[CLS] token（若存在）
            if self.has_cls_token:
                cls_token = x[:, 0:1, :]  # [B, 1, C]
                spatial_feat = x[:, 1:, :]  # [B, N-1, C]（原始token，排除[CLS]）
            else:
                spatial_feat = x  # [B, N, C]

            # 1. 计算自注意力分数（token与全局特征的匹配度）
            spatial_glo = F.normalize(spatial_feat.mean(dim=1, keepdim=True), dim=-1)  # [B, 1, C]
            self_attention = (spatial_glo * F.normalize(spatial_feat, dim=-1)).sum(dim=-1)  # [B, N-1]或[B, N]

            # 2. 计算交叉注意力分数（若提供）
            if cross_attention is None:
                # 若未提供跨模态注意力，则仅用自注意力
                cross_attention = torch.zeros_like(self_attention)

            # 3. 稀疏筛选：保留关键token，融合冗余token
            select_tokens, extra_token = self.sparse_net(
                tokens=spatial_feat,
                attention_x=self_attention,
                attention_y=cross_attention
            )  # select_tokens: [B, N_sparse, C], extra_token: [B, 1, C]

            # 4. 动态聚合：进一步减少token数量
            aggr_tokens = self.aggr_net(select_tokens)  # [B, keeped_patches, C]

            # 5. 整合精炼token（包含融合token和[CLS]）
            refined_feat = torch.cat([aggr_tokens, extra_token], dim=1)  # [B, keeped_patches+1, C]
            if self.has_cls_token:
                refined_feat = torch.cat([cls_token, refined_feat], dim=1)  # 拼接[CLS]
        else:
            if self.has_cls_token:
                cls_token = x[:, 0:1, :]  # [B, 1, C]
                end_token = x[:, -1:, :]
                spatial_feat = x[:, 1:-1, :]  # [B, N-1, C]（原始token，排除[CLS]）
            else:
                spatial_feat = x  # [B, N, C]

            # 1. 计算自注意力分数（token与全局特征的匹配度）
            spatial_glo = F.normalize(spatial_feat.mean(dim=1, keepdim=True), dim=-1)  # [B, 1, C]
            self_attention = (spatial_glo * F.normalize(spatial_feat, dim=-1)).sum(dim=-1)  # [B, N-1]或[B, N]

            # 2. 计算交叉注意力分数（若提供）
            if cross_attention is None:
                # 若未提供跨模态注意力，则仅用自注意力
                cross_attention = torch.zeros_like(self_attention)

            # 3. 稀疏筛选：保留关键token，融合冗余token
            select_tokens, extra_token = self.sparse_net(
                tokens=spatial_feat,
                attention_x=self_attention,
                attention_y=cross_attention
            )  # select_tokens: [B, N_sparse, C], extra_token: [B, 1, C]

            # 4. 动态聚合：进一步减少token数量
            aggr_tokens = self.aggr_net(select_tokens)  # [B, keeped_patches, C]

            # 5. 整合精炼token（包含融合token和[CLS]）
            refined_feat = torch.cat([aggr_tokens, extra_token], dim=1)  # [B, keeped_patches+1, C]
            if self.has_cls_token:
                refined_feat = torch.cat([cls_token, refined_feat, end_token], dim=1)  # 拼接[CLS]
        return refined_feat


class FineGrainedCLIP(nn.Module):
    """基于CLIP扩展细粒度融合功能，使用层注意力融合多层图像特征"""

    def __init__(self, base_clip_model, peb_module, atrm_module, criterion=None, num_layers=3):
        super().__init__()
        self.base_clip = base_clip_model
        self.peb = peb_module

        self.atrm = atrm_module
        self.criterion = criterion
        self.clip_dim = self.base_clip.visual.output_dim  # 768（最终输出维度）
        self.hidden_dim = 1024  # 1024（中间层维度）
        self.logit_scale = self.base_clip.logit_scale
        self.atrm_img_has_cls = False
        self.atrm_txt_has_cls = False
        self.patch_num = 257
        # 配置中间层提取（第4、8、12层，对应索引3、7、11）
        self.num_layers = num_layers
        self.selected_layers = [4, 8, 12]  # 目标层
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

    # 以下方法仅修改图像特征提取部分，其他逻辑保持不变
    def encode_image(self, images, cross_attention=None):
        """图像特征提取：使用层注意力融合多层特征"""
        img_fine_feat, img_feat_full = self._get_multi_layer_features(images, return_final=True)  # [B, 196, 768]
        img_patch_feat = img_feat_full[:, 1:, :]
        fused_img_feat = img_patch_feat + 0.5 * img_fine_feat  # [B, 196, 768]

        img_feat_atrm = self.atrm(
            fused_img_feat,
            cross_attention=cross_attention,
            atrm_type='image'
        )
        return img_feat_atrm.mean(dim=1)


    def encode_text(self, texts, cross_attention=None, clip_style_pos_emb=None):
        """文本特征提取：严格遵循 原始特征 → PEB补全 → ATRM精炼 的流程"""
        # 1. 获取原始文本特征（含CLIP原生位置嵌入）
        text_feat_raw = self.base_clip.encode_text_full(texts)  # [B, 248, 768]
        text_feat_raw = text_feat_raw @ self.base_clip.text_projection  # [B, 248, 768]

        # 2. PEB位置嵌入补全（核心：在ATRM之前完成位置信息注入）
        if clip_style_pos_emb is not None:
            # 移除首尾特殊字符后再融合位置嵌入（避免污染特殊字符）
            text_body = text_feat_raw[:, 1:-1, :]  # [B, 246, 768]（去除[CLS]和[SEP]）
            pos_emb_body = clip_style_pos_emb[:, 1:-1, :]  # 位置嵌入同步去除首尾

            # 位置嵌入补全：将PEB生成的位置信息注入文本特征
            text_body_with_pos = text_body + 0.1 * pos_emb_body  # 0.1为融合权重

            # 拼接回特殊字符（保持序列结构完整）
            text_feat_with_pos = torch.cat([
                text_feat_raw[:, :1, :],  # [CLS]
                text_body_with_pos,  # 补全后的主体
                text_feat_raw[:, -1:, :]  # [SEP]
            ], dim=1)  # [B, 248, 768]
        else:
            # 无PEB时直接使用原始特征
            text_feat_with_pos = text_feat_raw

        # 3. ATRM精炼（基于补全位置信息后的特征）
        text_feat_atrm = self.atrm(
            text_feat_with_pos,  # 传入已补全位置信息的特征
            cross_attention=cross_attention,
            atrm_type='text'
        )

        return text_feat_atrm.mean(dim=1)  # [B, 768]

    # def _encode_words(self, words_batch, device):
    #     """编码词特征（复用原逻辑）"""
    #     all_encoded = []
    #     for words in words_batch:
    #         if not words:
    #             empty_text = finelip.tokenize("").to(device)
    #             encoded = self.base_clip.encode_text(empty_text)
    #             all_encoded.append(encoded)
    #             continue
    #         word_tokens = finelip.tokenize(words, truncate=True).to(device)
    #         with torch.no_grad():
    #             word_feat = self.base_clip.encode_text(word_tokens)
    #         all_encoded.append(word_feat)
    #
    #     max_words = max([enc.shape[0] for enc in all_encoded]) if all_encoded else 0
    #     B = len(all_encoded)
    #     word_feats = torch.zeros(B, max_words, self.clip_dim, device=device)
    #     word_masks = torch.zeros(B, max_words, dtype=torch.bool, device=device)
    #     for i in range(B):
    #         num_w = all_encoded[i].shape[0]
    #         word_feats[i, :num_w, :] = all_encoded[i]
    #         word_masks[i, :num_w] = True
    #     return word_feats, word_masks

    def encode_image_full(self, images, cross_attention=None):
        """
        图像完整特征提取（返回完整序列特征，不做均值聚合）
        返回：融合细粒度特征后的patch序列 + ATRM精炼炼后的序列
        """
        # 1. 获取基础CLIP的完整图像特征（含[CLS]+所有patch）


        # 2. 细粒度特征提取与融合
        img_fine_feat,img_feat_full = self._get_multi_layer_features(images, return_final=True)   # [B, 196, 768]
        img_patch_feat = img_feat_full[:, 1:, :]
        fused_img_feat = img_patch_feat + 0.5 * img_fine_feat  # [B, 196, 768]

        # 3. ATRM精炼（保留完整序列，不做均值）
        img_feat_atrm = self.atrm(
            fused_img_feat,
            cross_attention=cross_attention,
            atrm_type='image'
        )  # [B, 196, 768]

        # 返回完整特征字典（方便后续灵活使用）
        return  img_feat_atrm

    def encode_text_full(self, texts, cross_attention=None, clip_style_pos_emb=None):
        """
        文本完整特征提取（返回完整序列特征，不做均值聚合）
        返回：原始文本序列 + PEB补全位置信息后的序列 + ATRM精炼后的序列
        """
        # 1. 获取基础CLIP的完整文本特征（含[CLS]+所有token）
        text_feat_raw = self.base_clip.encode_text_full(texts)  # [B, 248, 768]
        text_feat_raw = text_feat_raw @ self.base_clip.text_projection  # [B, 248, 768]

        # 2. PEB位置嵌入补全（保留完整序列结构）
        if clip_style_pos_emb is not None:
            text_body = text_feat_raw[:, 1:-1, :]  # [B, 246, 768]（主体部分）
            pos_emb_body = clip_style_pos_emb[:, 1:-1, :]  # 位置嵌入主体
            text_body_with_pos = text_body + 0.1 * pos_emb_body  # 注入位置信息

            # 拼接回特殊字符，保持完整序列
            text_feat_with_pos = torch.cat([
                text_feat_raw[:, :1, :],  # [CLS]
                text_body_with_pos,
                text_feat_raw[:, -1:, :]  # [SEP]
            ], dim=1)  # [B, 248, 768]
        else:
            text_feat_with_pos = text_feat_raw  # 无PEB时直接使用原始序列

        # 3. ATRM精炼（保留完整序列，不做均值）
        text_feat_atrm = self.atrm(
            text_feat_with_pos,
            cross_attention=cross_attention,
            atrm_type='text'
        )  # [B, 248, 768]

        # 返回完整特征字典
        return  text_feat_atrm




    def forward(self, images, texts, img_ids, warmup_alpha, local_rank, word_center_tokens = None, word_feats = None,word_count = None):
        B = images.shape[0]
        device = images.device
        peb_loss = torch.tensor(0.0, device=device)
        losses = {}
        # 图像特征提取（使用层注意力融合的多层特征）
        img_fine_feat, img_feat_full = self._get_multi_layer_features(images, return_final=True)
        img_patch_feat = img_feat_full[:, 1:, :]
        fused_img_feat = img_patch_feat + 0.5 * img_fine_feat * warmup_alpha
        


        # 文本特征处理（保持不变）
        text_feat_full = self.base_clip.encode_text_full(texts)
        text_feat_base = text_feat_full @ self.base_clip.text_projection

        # 跨模态注意力+PEB处理（保持不变）
        if word_center_tokens is not None and word_feats is not None and word_count is not None:


            word_feats = word_feats.cuda()
            word_count = word_count.to(device, non_blocking=True)
            word_pos_248_padded = []
            for centers in word_center_tokens:
                pos_248 = torch.tensor(centers, dtype=torch.long, device=device)
                word_pos_248_padded.append(pos_248)

            max_words = max([wp.size(0) for wp in word_pos_248_padded]) if word_pos_248_padded else 0

            # 用 -1 填充
            word_pos_248_padded = torch.stack([
                torch.nn.functional.pad(wp, (0, max_words - wp.size(0)), mode='constant', value=-1)
                for wp in word_pos_248_padded
            ])


            clip_style_pos_emb, peb_loss, losses = self.peb(
                word_feats,
                word_pos_248_padded,
                text_feat_full
            )

            word_feats_norm = F.normalize(word_feats, dim=-1)
            img_patch_norm = F.normalize(fused_img_feat, dim=-1)
            img_word_sim = torch.bmm(img_patch_norm, word_feats_norm.permute(0, 2, 1))
            img_cross_attn = img_word_sim.sum(dim=2) / (word_count.unsqueeze(1) + 1e-8)

            text_token_norm = F.normalize(text_feat_base, dim=-1)
            text_word_sim = torch.bmm(text_token_norm, word_feats_norm.permute(0, 2, 1))
            text_cross_attn = text_word_sim.sum(dim=2) / (word_count.unsqueeze(1) + 1e-8)
        else:
            img_cross_attn = torch.zeros(B, 196, device=device)
            text_cross_attn = torch.zeros(B, 248, device=device)
            clip_style_pos_emb = None

        # 位置嵌入处理（如果PEB未做简单的加权融合，这里可以用注意力机制去融合）
        # if clip_style_pos_emb is not None:
        #     token_indices = torch.arange(248, device=device).unsqueeze(0).repeat(B, 1)
        #     token_queries = self.peb.clip_pos_emb(token_indices)
        #     token_pos_emb, _ = nn.MultiheadAttention(
        #         embed_dim=self.clip_dim, num_heads=8, batch_first=True
        #     ).to(device)(
        #         query=token_queries,
        #         key=clip_style_pos_emb,
        #         value=clip_style_pos_emb
        #     )
        #     clip_style_pos_emb = token_pos_emb

        # 后续处理（保持不变）
        img_cross_attn = img_cross_attn[:, 1:]
        text_cross_attn = text_cross_attn[:, 1:-1]

        img_feat_atrm = self.atrm(fused_img_feat, cross_attention=img_cross_attn, atrm_type="image")
        img_feat_avg = img_feat_atrm.mean(dim=1)

        text_feat_avg = self.encode_text(
            texts,
            cross_attention=text_cross_attn,
            clip_style_pos_emb=clip_style_pos_emb
        )

        img_feat_norm = F.normalize(img_feat_avg, dim=-1)
        text_feat_norm = F.normalize(text_feat_avg, dim=-1)
        contrast_loss = self.criterion(img_feat_norm, text_feat_norm, img_ids) * warmup_alpha
        total_loss = contrast_loss + 0.5 * peb_loss

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
    torch.use_deterministic_algorithms(True,warn_only=True)
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

        # 2. 初始化新增模块（移除了图像细粒度提取器）
        self.peb_module = PEB(
            input_dim=self.clip_dim, hidden_dim=256, output_dim=self.clip_dim
        ).cuda()

        self.atrm_module = ATRM(
            dim=self.clip_dim,
            has_cls_token=True
        ).cuda()

        # 3. 初始化损失函数
        self.criterion = loss_select(opt=args, loss_type=args.loss_finegrain)

        # 4. 构建完整模型（不再传入img_fine_module）
        self.model = FineGrainedCLIP(
            base_clip_model=self.base_clip,
            peb_module=self.peb_module,
            atrm_module=self.atrm_module,
            criterion=self.criterion,
            num_layers=3  # 使用3层特征融合
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
        """优化器：区分CLIP主模型和新增模块的学习率"""
        clip_params = []
        new_module_params = []

        for n, p in self.model.named_parameters():
            # 新增模块包括PEB、图像细粒度提取器、ATRM
            if any(nd in n for nd in ["peb", "layer_attention", "atrm"]):
                new_module_params.append(p)
            else:
                clip_params.append(p)  # CLIP原始参数

        param_groups = [
            {'params': clip_params, 'lr': self.lr},
            {'params': new_module_params, 'lr': self.lr * 5}  # 新增模块学习率放大5倍
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
        self.model.train()  # 确保模型处于训练模式

        num_batches_per_epoch = len(dataloader)
        self.optimizer.zero_grad()

        for i, batch in enumerate(tqdm(dataloader, disable=(self.local_rank != 0))):
            step = num_batches_per_epoch * epoch + i
            if step < start_iter:
                continue

            # 1. 准备数据（补充词特征相关参数）
            images = batch["image_tensor"].cuda(non_blocking=True)
            texts_raw = batch["text"]
            texts = finelip.tokenize(texts_raw, truncate=True).cuda(non_blocking=True)
            img_ids = batch["img_id"].cuda(non_blocking=True)





            # 预热系数（控制新增模块的权重逐渐生效）
            warmup_alpha = (
                float(i) / num_batches_per_epoch
                if epoch < self.args.embedding_warmup_epochs  # 修正epoch判断条件
                else 1.0
            )

            # 2. 前向传播（传递跨模态注意力所需参数）
            with torch.cuda.amp.autocast():
                contrast_loss,peb_loss, total_loss, losses = self.model(
                    images=images,
                    texts=texts,
                    img_ids=img_ids,
                    warmup_alpha=warmup_alpha,
                    local_rank=self.local_rank,
                    word_center_tokens=batch["word_center_tokens"],
                    word_feats = batch["word_feats"],
                    word_count = batch["word_count"]
                )

            # 3. 处理异常损失
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"警告：在step {step} 出现异常损失，跳过该步")
                continue

            # 4. 梯度累积
            loss = total_loss / self.accumulation_steps  # 用总损失计算梯度
            self.scaler.scale(loss).backward()  # 混合精度训练

            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler(step)  # 学习率调度

            # 5. 损失累计（用于日志）
            running_loss += total_loss.item()

            # 6. 分布式损失同步
            dist.all_reduce(total_loss)
            avg_loss = total_loss.item() / dist.get_world_size()

            # 7. 日志记录（每1000步）
            if step % 200 == 0 and self.local_rank == 0:
                print("=====================================")
                # 记录学习率
                for idx, param_group in enumerate(self.optimizer.param_groups):
                    current_lr = param_group['lr']
                    print(f"step {step} 学习率_{idx}: {current_lr:.6f}")
                    self.writer.add_scalar(f"hyper/lr_{idx}", current_lr, step)
                # 记录logit_scale
                logit_scale = self.model.module.logit_scale.item()
                print(f"step {step} logit_scale: {logit_scale:.4f}")
                self.writer.add_scalar("logit_scale/train", logit_scale, step)
                # 记录损失
                print(f"step {step} peb损失: {peb_loss.item():.4f}")
                self.writer.add_scalar("Loss/train_avg", peb_loss.item(), step)
                print(f"step {step} 对比损失: {contrast_loss.item():.4f}")
                self.writer.add_scalar("Loss/contrast", contrast_loss.item(), step)
                print(f"step {step} PEB损失组成: {losses}")
                print(f"step {step} 平均损失: {avg_loss:.4f}")
                self.writer.add_scalar("Loss/train_avg", avg_loss, step)

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
            num_workers=32,
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
    parser.add_argument("--atrm_txt_has_cls",default=True)
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