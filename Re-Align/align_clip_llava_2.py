import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from tqdm import tqdm
import sys
import warnings
import re

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32  # 可改为float16节省显存（需模型支持）

parent_dir = os.path.abspath("./llava")
sys.path.append(parent_dir)

from constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from model.builder import load_pretrained_model
from mm_utils import tokenizer_image_token, get_model_name_from_path
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor  # 使用CLIP模型提取特征


# 分层压缩器：先局部注意力再全局压缩
class HierarchicalCompressor(nn.Module):
    def __init__(self, input_seq_len, target_seq_len, input_dim=4096, local_window_size=32, num_heads=8):
        super().__init__()
        self.target_seq_len = target_seq_len
        self.input_dim = input_dim
        self.local_window_size = local_window_size
        self.num_heads = num_heads  # 保存头数，用于掩码扩展

        # 第一阶段：局部注意力（捕捉局部语义块）
        self.local_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,  # 使用传入的头数
            batch_first=True
        )
        self.local_norm = nn.LayerNorm(input_dim)
        self.local_proj = nn.Linear(input_dim, input_dim)

        # 第二阶段：全局压缩（适配目标长度）
        self.global_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.global_norm = nn.LayerNorm(input_dim)

        # 可学习的查询向量（用于全局压缩）
        self.query_tokens = nn.Parameter(
            torch.randn(1, target_seq_len, input_dim)  # [1, target_len, dim]
        )

        # 输出投影
        self.output_proj = nn.Linear(input_dim, input_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, vlm_emb, valid_mask):
        # vlm_emb: [B, seq_len, input_dim]
        # valid_mask: [B, seq_len] (True表示有效token)
        batch_size, seq_len = vlm_emb.shape[0], vlm_emb.shape[1]

        # --------------------------
        # 第一阶段：局部注意力（滑窗处理）
        # --------------------------
        local_outputs = []
        for i in range(0, seq_len, self.local_window_size):
            # 提取窗口内特征
            window_end = min(i + self.local_window_size, seq_len)
            window_emb = vlm_emb[:, i:window_end, :]  # [B, window_len, dim]
            window_mask = valid_mask[:, i:window_end]  # [B, window_len]

            # 窗口内自注意力掩码：[B, window_len, window_len]
            attn_mask = torch.where(
                window_mask.unsqueeze(1) & window_mask.unsqueeze(2),
                0.0, -1e9
            ).to(vlm_emb.device)  # [B, window_len, window_len]

            # 关键修复：将掩码扩展到 [B * num_heads, window_len, window_len]
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)  # 按头数复制

            # 窗口内自注意力计算
            local_attn_out, _ = self.local_attn(
                query=window_emb,
                key=window_emb,
                value=window_emb,
                attn_mask=attn_mask  # 使用扩展后的掩码
            )
            local_out = self.local_norm(window_emb + local_attn_out)
            local_out = self.local_proj(local_out)
            local_outputs.append(local_out)

        # 拼接所有窗口输出
        local_features = torch.cat(local_outputs, dim=1)  # [B, seq_len, dim]

        # --------------------------
        # 第二阶段：全局压缩
        # --------------------------
        # 扩展查询向量到批次维度
        queries = self.query_tokens.repeat(batch_size, 1, 1)  # [B, target_len, dim]

        # 全局注意力掩码（无需扩展，MultiheadAttention会自动处理）
        global_mask = valid_mask.unsqueeze(1)  # [B, 1, seq_len]

        # 计算全局注意力
        global_attn_out, _ = self.global_attn(
            query=queries,
            key=local_features,
            value=local_features,
            key_padding_mask=~valid_mask  # 注意：key_padding_mask是[B, seq_len]，无需扩展
        )
        global_out = self.global_norm(queries + global_attn_out)
        compressed = self.output_proj(global_out)  # [B, target_len, dim]

        return compressed

# 共享中间投影网络（将不同模态特征映射到同一空间）
class SharedProjectionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.proj(x)


# VAE模块（使用CLIP图像特征进行跨模态注意力）
class CrossModalVAE(nn.Module):
    def __init__(self, latent_dim=256, clip_dim=768, target_seq_len=77):
        super().__init__()
        self.latent_dim = latent_dim
        self.target_seq_len = target_seq_len  # 固定为77 token

        # 共享投影网络（CLIP特征 → 桥接空间）
        self.clip_projection = SharedProjectionNetwork(
            input_dim=clip_dim,
            hidden_dim=latent_dim,
            output_dim=latent_dim
        )

        # 跨模态注意力（用CLIP图像特征指导补全）
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=8,
            batch_first=True
        )

        # 编码器：从投影特征到隐空间
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # 均值和方差投影（序列级）
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        # 解码器：从隐空间重构特征
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, clip_dim)  # 输出与CLIP同维度
        )

        # 位置编码（确保序列顺序信息）
        self.pos_embedding = nn.Embedding(target_seq_len, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, clip_truncated_emb, clip_img_feat):
        # clip_truncated_emb: [B, 77, clip_dim]（CLIP截断特征）
        # clip_img_feat: [B, clip_dim]（CLIP图像特征，用于指导）

        batch_size = clip_truncated_emb.shape[0]
        seq_len = clip_truncated_emb.shape[1]  # 固定为77

        # 1. 共享投影：将CLIP特征映射到桥接空间
        clip_proj = self.clip_projection(clip_truncated_emb)  # [B, 77, latent_dim]
        img_proj = self.clip_projection(clip_img_feat.unsqueeze(1))  # [B, 1, latent_dim]
        img_proj = img_proj.repeat(1, seq_len, 1)  # [B, 77, latent_dim]

        # 2. 加入位置编码
        pos_ids = torch.arange(seq_len, device=clip_truncated_emb.device)
        pos_emb = self.pos_embedding(pos_ids).unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 77, latent_dim]
        clip_proj = clip_proj + pos_emb  # [B, 77, latent_dim]

        # 3. 跨模态注意力（文本为query，图像为key/value）
        attn_output, _ = self.cross_attention(
            query=clip_proj,
            key=img_proj,
            value=img_proj
        )  # [B, 77, latent_dim]

        # 4. 编码到隐空间
        encoded = self.encoder(attn_output)  # [B, 77, latent_dim]
        mu = self.fc_mu(encoded)  # [B, 77, latent_dim]
        logvar = self.fc_logvar(encoded)  # [B, 77, latent_dim]

        # 5. 重参数化
        z = self.reparameterize(mu, logvar)  # [B, 77, latent_dim]

        # 6. 解码为补全特征
        completed_emb = self.decoder(z)  # [B, 77, clip_dim]

        return completed_emb, mu, logvar


CONFIG = {
    "clip_model_name": "openai/clip-vit-large-patch14",
    "llava_model_path": "/data2/gaodz/llava-v1.6-vicuna-7b",
    "output_dir": "/data2/gaodz/Compression_VAE_CLIP",
    "batch_size": 16,
    "epochs_compression": 2,  # 压缩模块训练
    "epochs_vae": 5,  # VAE训练
    "learning_rate_compression": 2e-5,
    "learning_rate_vae": 5e-6,
    "log_interval": 5,
    "llava_dim": 4096,
    "vae_latent_dim": 256,
    "num_heads": 8,
    "max_clip_long_length": 248,
    "max_compressed_length": 77,  # 文本压缩后长度
    "max_image_compressed_length": 32,  # 图像token压缩后长度
    "local_window_size": 32,  # 分层压缩的局部窗口大小
    "datasets": [
        {"data_json_path": "/data2/gaodz/Re-Align/hypernet_train_data_short_core.json",
         "image_root": "/data2/gaodz/OmniConsistency"},
        {"data_json_path": "/data2/gaodz/Re-Align/COCO_short_core_1.json", "image_root": "/data2/gaodz/train2014"},
        {"data_json_path": "/data2/gaodz/WikiArt/OpenDataLab___WikiArt/raw/train_txt/image_text_new.json",
         "image_root": "/data2/gaodz/WikiArt/OpenDataLab___WikiArt/raw/train_image/wikiart"},
        {"data_json_path": "/data2/gaodz/sharegpt4v/sharegpt4v_coco.json",
         "image_root": "/data2/gaodz/coco2017/PAI/COCO2017"},
    ]
}
os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(os.path.join(CONFIG["output_dir"], "samples"), exist_ok=True)


# 数据集
class CompressionVAEDataset(Dataset):
    def __init__(self, json_path, image_root, clip_processor):
        self.samples = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                required = ["image", "long_text", "target_image"]
                if not all(k in sample for k in required):
                    raise ValueError(f"样本缺少字段：{sample.keys()}，需包含{required}")
                self.samples.append(sample)
        self.image_root = image_root
        self.clip_processor = clip_processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 加载图像（用于CLIP特征提取和VAE）
        img_path = os.path.join(self.image_root, sample["image"])
        image = Image.open(img_path).convert("RGB")

        return {
            "image": image,
            "long_text": sample["long_text"],
            "image_path": sample["image"],
            "sample_image_root": self.image_root
        }


def custom_collate_fn(batch):
    images = [item['image'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    long_texts = [item['long_text'] for item in batch]
    sample_image_roots = [item['sample_image_root'] for item in batch]

    return {
        "image": images,
        "image_path": image_paths,
        "long_text": long_texts,
        "sample_image_root": sample_image_roots,
    }


# 主模块：双压缩 + VAE + CLIP对齐
class CompressionVAEModule(nn.Module):
    def __init__(self, vlm_model, vlm_tokenizer, image_processor, clip_model, clip_processor):
        super().__init__()
        self.vlm_model = vlm_model
        self.vlm_tokenizer = vlm_tokenizer
        self.image_processor = image_processor
        self.clip_model = clip_model
        self.clip_processor = clip_processor

        self.llava_dim = CONFIG["llava_dim"]
        self.clip_dim = clip_model.config.projection_dim  # CLIP特征维度
        self.bridge_dim = CONFIG["vae_latent_dim"]  # 桥接空间维度

        # 文本分层压缩器
        self.text_compressor = HierarchicalCompressor(
            input_seq_len=CONFIG["max_clip_long_length"],
            target_seq_len=CONFIG["max_compressed_length"],
            input_dim=self.llava_dim,
            local_window_size=CONFIG["local_window_size"]
        )

        # 图像token分层压缩器
        self.image_token_compressor = HierarchicalCompressor(
            input_seq_len=576,  # LLaVA图像token长度
            target_seq_len=CONFIG["max_image_compressed_length"],
            input_dim=self.llava_dim,
            local_window_size=CONFIG["local_window_size"]
        )

        # 共享投影网络（LLaVA特征 → 桥接空间）
        self.llava_projection = SharedProjectionNetwork(
            input_dim=self.llava_dim,
            hidden_dim=self.bridge_dim,
            output_dim=self.bridge_dim
        )

        # CLIP特征投影网络（保持与LLaVA投影空间一致）
        self.clip_projection = SharedProjectionNetwork(
            input_dim=self.clip_dim,
            hidden_dim=self.bridge_dim,
            output_dim=self.bridge_dim
        )

        # 跨模态VAE（使用CLIP图像特征）
        self.vae = CrossModalVAE(
            latent_dim=CONFIG["vae_latent_dim"],
            clip_dim=self.clip_dim,
            target_seq_len=CONFIG["max_compressed_length"]  # 77
        )

        # 初始冻结预训练模型
        for param in self.vlm_model.parameters():
            param.requires_grad = False
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # 默认只训练压缩器和投影网络
        for param in self.text_compressor.parameters():
            param.requires_grad = True
        for param in self.image_token_compressor.parameters():
            param.requires_grad = True
        for param in self.llava_projection.parameters():
            param.requires_grad = True
        for param in self.clip_projection.parameters():
            param.requires_grad = True
        for param in self.vae.parameters():
            param.requires_grad = False

    def process_vlm_inputs(self, images, texts):
        # 准备LLaVA输入
        texts = [f"{DEFAULT_IMAGE_TOKEN}{text}" for text in texts]
        input_text = []
        for txt in texts:
            tokenized = tokenizer_image_token(
                txt, self.vlm_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).squeeze(0)
            input_text.append(tokenized)
        llava_inputs = pad_sequence(
            input_text, batch_first=True, padding_value=self.vlm_tokenizer.pad_token_id
        ).to(device)

        # 处理图像
        processed_imgs = self.image_processor(
            images=images,
            return_tensors="pt",
        )["pixel_values"].to(device=device, dtype=dtype, non_blocking=True)

        # 获取LLaVA特征
        with torch.no_grad():
            llava_outputs = self.vlm_model(
                llava_inputs, images=processed_imgs, output_hidden_states=True
            )
        hidden_states = llava_outputs.hidden_states[-1]  # [B, seq_len, 4096]

        # 分离图像token和文本token（跳过cls）
        image_tokens = hidden_states[:, 1:577, :]  # 图像token部分
        text_tokens = hidden_states[:, 577:, :]  # 文本token部分

        # 创建掩码
        text_mask_in_inputs = (llava_inputs[:, 1:] != self.vlm_tokenizer.pad_token_id)  # [B, 文本token数]
        text_token_len = text_tokens.shape[1]
        mask_len = text_mask_in_inputs.shape[1]
        if text_token_len < mask_len:
            text_mask = text_mask_in_inputs[:, :text_token_len]  # 截断掩码至实际文本长度
        else:
            text_mask = text_mask_in_inputs

        return {
            "image_tokens": image_tokens,
            "text_tokens": text_tokens,
            "text_mask": text_mask
        }

    def get_clip_features(self, images, texts):
        # CLIP图像特征
        img_inputs = self.clip_processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            clip_img_feat = self.clip_model.get_image_features(**img_inputs)  # [B, 768]

        # CLIP文本特征
        text_inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            clip_text_feat = self.clip_model.get_text_features(**text_inputs)  # [B, 768]

        return clip_img_feat, clip_text_feat

    def fine_grained_contrastive_loss(self, compressed, original, mask, temperature=0.1):
        """细粒度对比损失：让压缩特征与原始特征在token级对齐"""
        batch_size = compressed.shape[0]
        comp_seq_len = compressed.shape[1]
        orig_seq_len = original.shape[1]
        feat_dim = compressed.shape[2]

        # 处理原始token的掩码
        mask_original = mask.unsqueeze(1).bool()  # [B, 1, orig_seq_len]
        attn_mask = torch.where(
            mask_original,
            torch.tensor(0.0, device=compressed.device),
            torch.tensor(-1e9, device=compressed.device)
        )  # [B, 1, orig_seq_len]

        # 计算相似度矩阵
        sim_matrix = torch.matmul(
            F.normalize(compressed, dim=-1),
            F.normalize(original, dim=-1).transpose(1, 2)
        )  # [B, comp_seq_len, orig_seq_len]

        # 应用掩码并计算注意力权重
        attn_weights = F.softmax(
            sim_matrix / temperature + attn_mask,
            dim=-1
        )  # [B, comp_seq_len, orig_seq_len]

        # 基于软对齐重构原始特征
        reconstructed = torch.matmul(attn_weights, original)  # [B, comp_seq_len, feat_dim]

        # 计算损失
        cos_loss = 1 - F.cosine_similarity(compressed, reconstructed, dim=-1).mean()
        l2_loss = F.mse_loss(compressed, reconstructed) / feat_dim
        return cos_loss + l2_loss * 0.5

    def clip_alignment_loss(self, compressed_image, compressed_text, clip_img_feat, clip_text_feat):
        """压缩特征与CLIP特征的对齐损失（在共享桥接空间中计算）"""
        # 图像压缩特征 → 桥接空间
        img_compress_proj = self.llava_projection(compressed_image)  # [B, 32, bridge_dim]
        img_compress_mean = img_compress_proj.mean(dim=1)  # [B, bridge_dim]

        # CLIP图像特征 → 桥接空间
        clip_img_proj = self.clip_projection(clip_img_feat)  # [B, bridge_dim]

        # 文本压缩特征 → 桥接空间
        txt_compress_proj = self.llava_projection(compressed_text)  # [B, 77, bridge_dim]
        txt_compress_mean = txt_compress_proj.mean(dim=1)  # [B, bridge_dim]

        # CLIP文本特征 → 桥接空间
        clip_text_proj = self.clip_projection(clip_text_feat)  # [B, bridge_dim]

        # 归一化并计算相似度损失
        img_align_loss = 1 - F.cosine_similarity(
            F.normalize(img_compress_mean, dim=-1),
            F.normalize(clip_img_proj, dim=-1),
            dim=-1
        ).mean()

        text_align_loss = 1 - F.cosine_similarity(
            F.normalize(txt_compress_mean, dim=-1),
            F.normalize(clip_text_proj, dim=-1),
            dim=-1
        ).mean()

        # 跨模态一致性损失
        cross_align_loss = 1 - F.cosine_similarity(
            F.normalize(img_compress_mean, dim=-1),
            F.normalize(txt_compress_mean, dim=-1),
            dim=-1
        ).mean()

        return (img_align_loss + text_align_loss) * 2.0 + cross_align_loss

    def vae_loss(self, completed_emb, target_compressed_text, mu, logvar, clip_img_feat):
        """VAE损失：重构损失 + KL散度 + 跨模态对齐"""
        # 1. 重构损失
        recon_loss = F.mse_loss(completed_emb, target_compressed_text)

        # 2. KL散度（序列级）
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (mu.size(0) * mu.size(1))  # 按批次和序列长度归一化

        # 3. 与CLIP图像特征的对齐约束（在共享空间中）
        completed_proj = self.clip_projection(completed_emb)  # [B, 77, bridge_dim]
        completed_mean = completed_proj.mean(dim=1)  # [B, bridge_dim]
        img_feat_proj = self.clip_projection(clip_img_feat)  # [B, bridge_dim]
        align_loss = 1 - F.cosine_similarity(
            F.normalize(completed_mean, dim=-1),
            F.normalize(img_feat_proj, dim=-1),
            dim=-1
        ).mean()

        return recon_loss * 10.0 + kl_loss + align_loss * 3.0

    def forward(self, images, texts, stage="compression"):
        # 1. 获取LLaVA特征和CLIP特征
        vlm_features = self.process_vlm_inputs(images, texts)
        clip_img_feat, clip_text_feat = self.get_clip_features(images, texts)
        image_mask = torch.ones(vlm_features["image_tokens"].shape[:2], dtype=torch.bool, device=device)

        # 2. 双压缩（分层压缩）
        compressed_image = self.image_token_compressor(
            vlm_features["image_tokens"],
            image_mask
        )
        compressed_text = self.text_compressor(
            vlm_features["text_tokens"],
            vlm_features["text_mask"]
        )

        # 3. 压缩阶段：压缩损失 + CLIP对齐损失
        if stage == "compression":
            # 压缩损失
            img_compress_loss = self.fine_grained_contrastive_loss(
                compressed_image,
                vlm_features["image_tokens"][:, :compressed_image.size(1), :],
                image_mask[:, :compressed_image.size(1)]
            )
            txt_compress_loss = self.fine_grained_contrastive_loss(
                compressed_text,
                vlm_features["text_tokens"][:, :compressed_text.size(1), :],
                vlm_features["text_mask"][:, :compressed_text.size(1)]
            )

            # CLIP对齐损失（在共享空间中）
            clip_align_loss = self.clip_alignment_loss(
                compressed_image, compressed_text,
                clip_img_feat, clip_text_feat
            )

            return (img_compress_loss + txt_compress_loss) / 2 + clip_align_loss

        # 4. VAE阶段：用CLIP图像特征调整文本
        elif stage == "vae":
            # 融合压缩特征（先投影到CLIP维度）

            #这里直接用线性层压缩维度，能改进吗
            text_to_clip_dim = nn.Linear(self.llava_dim, self.clip_dim).to(device)(compressed_text)
            completed_emb, mu, logvar = self.vae(
                clip_truncated_emb=text_to_clip_dim,  # 压缩文本特征转CLIP维度
                clip_img_feat=clip_img_feat  # CLIP图像特征作为指导
            )

            # 计算VAE损失
            vae_loss = self.vae_loss(
                completed_emb=completed_emb,
                target_compressed_text=text_to_clip_dim,  # 目标是CLIP维度的压缩文本
                mu=mu,
                logvar=logvar,
                clip_img_feat=clip_img_feat
            )

            return vae_loss, completed_emb


# 模型加载
def load_models(device):
    # 加载LLaVA
    vlm_model_path = CONFIG["llava_model_path"]
    model_name = get_model_name_from_path(vlm_model_path)
    vlm_tokenizer, vlm_model, image_processor, _ = load_pretrained_model(
        vlm_model_path, None, model_name, device, torch_dtype=dtype
    )
    vlm_model.eval()

    # 加载CLIP
    clip_model = CLIPModel.from_pretrained(CONFIG["clip_model_name"]).to(device, dtype=dtype)
    clip_processor = CLIPProcessor.from_pretrained(CONFIG["clip_model_name"])
    clip_model.eval()

    return vlm_model, vlm_tokenizer, image_processor, clip_model, clip_processor


if __name__ == "__main__":
    # 1. 加载模型
    vlm_model, vlm_tokenizer, image_processor, clip_model, clip_processor = load_models(device)
    print("模型加载完成！")

    # 2. 初始化压缩+VAE模块
    main_module = CompressionVAEModule(
        vlm_model=vlm_model,
        vlm_tokenizer=vlm_tokenizer,
        image_processor=image_processor,
        clip_model=clip_model,
        clip_processor=clip_processor
    ).to(device, dtype=dtype)
    print("压缩+VAE模块初始化完成")

    # 3. 加载数据集
    all_datasets = []
    for dataset_info in CONFIG["datasets"]:
        dataset = CompressionVAEDataset(
            json_path=dataset_info["data_json_path"],
            image_root=dataset_info["image_root"],
            clip_processor=clip_processor
        )
        all_datasets.append(dataset)
        print(f"加载数据集：{dataset_info['data_json_path']}，样本数：{len(dataset)}")
    combined_dataset = ConcatDataset(all_datasets)
    print(f"合并后总样本数：{len(combined_dataset)}")

    # 4. 创建DataLoader
    dataloader = DataLoader(
        combined_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=True
    )
    print(f"DataLoader创建完成，总Batch数：{len(dataloader)}")

    # --------------------------
    # 阶段1：训练压缩模块（带CLIP对齐）
    # --------------------------
    print("\n===== 开始训练压缩模块 =====")
    optimizer_compression = optim.AdamW(
        [
            *list(main_module.text_compressor.parameters()),
            *list(main_module.image_token_compressor.parameters()),
            *list(main_module.llava_projection.parameters()),
            *list(main_module.clip_projection.parameters())
        ],
        lr=CONFIG["learning_rate_compression"],
        weight_decay=0.01
    )

    for epoch in range(CONFIG["epochs_compression"]):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Compression Epoch {epoch + 1}/{CONFIG['epochs_compression']}")
        main_module.train()

        for batch_idx, batch in enumerate(progress_bar):
            images = batch["image"]
            texts = batch["long_text"]

            optimizer_compression.zero_grad()
            loss = main_module(
                images=images,
                texts=texts,
                stage="compression"
            )
            loss.backward()
            optimizer_compression.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Compression Loss": f"{loss:.4f}"})

            if (batch_idx + 1) % CONFIG["log_interval"] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Avg Loss: {avg_loss:.4f}")

        # 保存压缩模块
        epoch_avg_loss = total_loss / len(dataloader)
        save_path = os.path.join(CONFIG["output_dir"], f"compression_epoch_{epoch + 1}.pth")
        torch.save({
            "text_compressor": main_module.text_compressor.state_dict(),
            "image_token_compressor": main_module.image_token_compressor.state_dict(),
            "llava_projection": main_module.llava_projection.state_dict(),
            "clip_projection": main_module.clip_projection.state_dict()
        }, save_path)
        print(f"压缩模块保存至：{save_path}\n")

    # --------------------------
    # 阶段2：训练VAE（CLIP图像特征指导）
    # --------------------------
    print("\n===== 开始训练VAE =====")
    # 冻结压缩模块和投影网络，解冻VAE
    for param in main_module.text_compressor.parameters():
        param.requires_grad = False
    for param in main_module.image_token_compressor.parameters():
        param.requires_grad = False
    for param in main_module.llava_projection.parameters():
        param.requires_grad = False
    for param in main_module.clip_projection.parameters():
        param.requires_grad = False
    for param in main_module.vae.parameters():
        param.requires_grad = True

    optimizer_vae = optim.AdamW(
        main_module.vae.parameters(),
        lr=CONFIG["learning_rate_vae"],
        weight_decay=0.01
    )

    for epoch in range(CONFIG["epochs_vae"]):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"VAE Epoch {epoch + 1}/{CONFIG['epochs_vae']}")
        main_module.train()

        for batch_idx, batch in enumerate(progress_bar):
            images = batch["image"]
            texts = batch["long_text"]

            optimizer_vae.zero_grad()
            loss, _ = main_module(
                images=images,
                texts=texts,
                stage="vae"
            )
            loss.backward()
            optimizer_vae.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"VAE Loss": f"{loss:.4f}"})

            if (batch_idx + 1) % CONFIG["log_interval"] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"VAE Epoch {epoch + 1}, Batch {batch_idx + 1}, Avg Loss: {avg_loss:.4f}")

        # 保存VAE模型
        epoch_avg_loss = total_loss / len(dataloader)
        save_path = os.path.join(CONFIG["output_dir"], f"vae_epoch_{epoch + 1}.pth")
        torch.save(main_module.vae.state_dict(), save_path)
        print(f"VAE模型保存至：{save_path}\n")
