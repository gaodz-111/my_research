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
from diffusers import StableDiffusionPipeline

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
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer  # 仅使用Transformers库的CLIP




CONFIG = {

    "clip_model_name": "openai/clip-vit-large-patch14",
    "sd_model_path": "/data2/gaodz/stable-diffusion-2-1-base",
    "llava_model_path": "/data2/gaodz/llava-v1.6-vicuna-7b",
    "output_dir": "/data2/gaodz/Alignclip_llava",
    "batch_size": 16,
    "epochs": 6,
    "learning_rate": 1e-5,
    "log_interval": 5,
    "clip_dim": 1024,  # 与所选CLIP模型匹配（ViT-L/14-336为1024）
    "llava_dim": 4096,
    "num_heads": 8,
    "max_clip_length": 77,  # CLIP文本固定77token
    "max_clip_long_length": 248,
    "datasets": [
        {"data_json_path": "/data2/gaodz/Re-Align/hypernet_train_data_short_core.json",
         "image_root": "/data2/gaodz/OmniConsistency"},
        {"data_json_path": "/data2/gaodz/Re-Align/COCO_short_core_1.json", "image_root": "/data2/gaodz/train2014"},
        {"data_json_path": "/data2/gaodz/WikiArt/OpenDataLab___WikiArt/raw/train_txt/image_text_new.json",
         "image_root": "/data2/gaodz/WikiArt/OpenDataLab___WikiArt/raw/train_image/wikiart"},
        {"data_json_path": "/data2/gaodz/sharegpt4v/sharegpt4v_coco.json", "image_root": "/data2/gaodz/coco2017/PAI/COCO2017"},
    ]
}
os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(os.path.join(CONFIG["output_dir"], "samples"), exist_ok=True)

# def print_memory_summary(tag=""):
#     torch.cuda.synchronize()
#     print(f"\n===== GPU Memory Summary: {tag} =====")
#     print(torch.cuda.memory_summary(device=torch.cuda.current_device(), abbreviated=False))
#     print("=====================================\n")


# 数据集：使用Transformers的CLIPProcessor预处理
class LLM_CoreTextDataset(Dataset):
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
        self.clip_processor = clip_processor  # 使用Transformers的CLIPProcessor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 加载目标图像（用CLIPProcessor预处理）
        target_img_path = os.path.join(self.image_root, sample["target_image"])
        target_image = Image.open(target_img_path).convert("RGB")
        # 仅预处理图像（文本处理在后续单独进行）
        target_image = self.clip_processor(images=target_image, return_tensors="pt")["pixel_values"].squeeze(0)
        return {
            "image_path": sample["image"],
            "target_image": target_image,
            "long_text": sample["long_text"],
            "sample_image_root": self.image_root
        }


def custom_collate_fn(batch):
    target_images = torch.stack([item['target_image'] for item in batch], dim=0)
    image_paths = [item['image_path'] for item in batch]
    long_texts = [item['long_text'] for item in batch]
    sample_image_roots = [item['sample_image_root'] for item in batch]
    return {
        "target_image": target_images,
        "image_path": image_paths,
        "long_text": long_texts,
        "sample_image_roots": sample_image_roots,
    }


# 补全解码器（保持不变）
class CompletionDecoder(nn.Module):
    def __init__(self, clip_dim=1024, vlm_dim=4096, num_cross_attn_layers=1, num_self_attn_layers=2):
        super().__init__()
        self.vlm_fuser = nn.Sequential(
            nn.Linear(vlm_dim, clip_dim),
            nn.LayerNorm(clip_dim),
            nn.GELU()
        )
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=clip_dim,
                nhead=CONFIG["num_heads"],
                dim_feedforward=clip_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_cross_attn_layers)
        ])
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=clip_dim,
                nhead=CONFIG["num_heads"],
                dim_feedforward=clip_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_self_attn_layers)
        ])
        self.output_proj = nn.Linear(clip_dim, clip_dim)

    def forward(self, clip_truncated_emb, vlm_full_emb):
        vlm_proj = self.vlm_fuser(vlm_full_emb)
        x = clip_truncated_emb
        for cross_layer in self.cross_attn_layers:
            x = cross_layer(tgt=x, memory=vlm_proj)
        for self_layer in self.self_attn_layers:
            x = self_layer(x)
        return self.output_proj(x)


class LLaVATokenAggregator(nn.Module):
    def __init__(self, input_seq_len=248, target_seq_len=77, input_dim=4096):
        super().__init__()
        self.target_seq_len = target_seq_len  # 目标聚合长度：77
        self.input_dim = input_dim

        # 学习聚合权重矩阵 W_ref: [target_seq_len, input_seq_len]
        self.W_q = nn.Linear(input_dim, 2048)  # 投影到低维，降低计算量
        self.W_k = nn.Linear(input_dim, 2048)
        self.temperature = nn.Parameter(torch.tensor(1.0))  # 控制注意力稀疏度

    def forward(self, vlm_emb, valid_mask_stretched):
        # vlm_emb: [B, 248, 4096]（拉伸后的嵌入）
        # valid_mask_stretched: [B, 248]（有效位置True，Pad位置False）
        batch_size = vlm_emb.shape[0]

        # 生成Q和K（保持不变）
        q_base = torch.randn(batch_size, self.target_seq_len, self.input_dim, device=vlm_emb.device) * 0.01
        q = self.W_q(q_base)  # [B, 77, 256]
        k = self.W_k(vlm_emb)  # [B, 248, 256]

        # 扩展掩码维度以匹配注意力权重（[B,77,248]）
        valid_mask = valid_mask_stretched.unsqueeze(1).expand(-1, self.target_seq_len, -1)  # [B,77,248]

        # 应用掩码（Pad位置设为负无穷）
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.temperature * torch.sqrt(torch.tensor(2048, device=device)))
        attn_weights = attn_weights.masked_fill(~valid_mask, -1e9)  # 仅Pad位置被掩码
        attn_weights = F.softmax(attn_weights, dim=-1)

        aggregated_emb = torch.matmul(attn_weights, vlm_emb)  # [B,77,4096]
        return aggregated_emb


# 对齐模块：核心修改为Transformers的CLIP接口
class CLIPAlignmentModule(nn.Module):
    def __init__(self, vlm_model, clip_model, vlm_tokenizer, clip_processor, image_processor):
        super().__init__()
        self.vlm_model = vlm_model
        self.vlm_tokenizer = vlm_tokenizer
        self.clip_model = clip_model  # Transformers的CLIP模型
        self.clip_processor = clip_processor  # 统一处理图像和文本
        self.clip_tokenizer = clip_processor.tokenizer  # 从processor中提取tokenizer
        self.image_processor = image_processor

        # 从CLIP模型获取特征维度（避免硬编码）
        self.clip_dim = clip_model.config.projection_dim
        self.llava_dim = CONFIG["llava_dim"]

        self.llava_aggregator = LLaVATokenAggregator(
            input_seq_len=248,  # 248
            target_seq_len=77,  # 77
            input_dim=self.llava_dim  # 4096
        )

        # 冻结预训练模型
        for param in self.vlm_model.parameters():
            param.requires_grad = False
        for param in self.clip_model.parameters():
            param.requires_grad = False

        for param in self.llava_aggregator.parameters():
            param.requires_grad = True

        # 可训练模块
        self.completion_decoder = CompletionDecoder(
            clip_dim=self.clip_dim,
            vlm_dim=self.llava_dim,
            num_cross_attn_layers=1,
            num_self_attn_layers=2
        )
        self.temperature = nn.Parameter(torch.tensor(0.07))  # CLIP默认温度参数


    def _stretch_llava_seq(self, vlm_text_embeds):
        batch_size, seq_len, dim = vlm_text_embeds.shape
        target_len = CONFIG["max_clip_long_length"]
        if seq_len > target_len:
            return vlm_text_embeds[:, :target_len, :]
        elif seq_len < target_len:
            pad_emb = self.vlm_model.model.embed_tokens(
                torch.tensor([self.vlm_tokenizer.pad_token_id], device=vlm_text_embeds.device)
            ).squeeze(0).to(dtype=vlm_text_embeds.dtype)
            pad = pad_emb.unsqueeze(0).unsqueeze(0).repeat(batch_size, target_len - seq_len, 1)
            return torch.cat([vlm_text_embeds, pad], dim=1)
        return vlm_text_embeds

    def process_vlm_inputs(self, sample_image_roots, image_paths, long_texts):
        # LLaVA处理逻辑不变
        texts = [f"{DEFAULT_IMAGE_TOKEN}{text}" for text in long_texts]
        input_text = []
        for txt in texts:
            tokenized = tokenizer_image_token(
                txt, self.vlm_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).squeeze(0)
            input_text.append(tokenized)
        llava_inputs = pad_sequence(
            input_text, batch_first=True, padding_value=self.vlm_tokenizer.pad_token_id
        ).to(device)
        L_total = llava_inputs.shape[1]
        batch_size = llava_inputs.shape[0]
        # processed_imgs = []
        # for img_path, img_root in zip(image_paths, sample_image_roots):
        #     img = Image.open(os.path.join(img_root, img_path)).convert("RGB")
        #     processed = self.image_processor(img, return_tensors="pt")["pixel_values"].to(dtype=dtype)
        #     processed_imgs.append(processed)
        # processed_imgs = torch.cat(processed_imgs, dim=0).to(device)
        imgs = [
            Image.open(os.path.join(img_root, img_path)).convert("RGB")
            for img_path, img_root in zip(image_paths, sample_image_roots)
        ]
        # 2. 批量预处理（直接返回 [B, C, H, W]，无需循环 cat）
        processed_imgs = self.image_processor(
            images=imgs,
            return_tensors="pt",
        )["pixel_values"].to(device=device, dtype=dtype, non_blocking=True)

        with torch.no_grad():
            llava_outputs = self.vlm_model(
                llava_inputs, images=processed_imgs, output_hidden_states=True
            )
        hidden_states = llava_outputs.hidden_states[-1]

        start_pos = min(577, hidden_states.shape[1])

        vlm_text_embeds = hidden_states[:, start_pos:, :]
        #有效文本长度
        non_pad_mask = (llava_inputs != self.vlm_tokenizer.pad_token_id)  # [B, L_total]

        # 创建新掩码：仅保留文本部分（从start_pos开始）的有效标记，排除图像token
        original_valid_mask = torch.full((batch_size, L_total), False, device=llava_inputs.device)

        # 文本部分的有效标记 = 非pad 且 在start_pos之后
        original_valid_mask[:, 1:] = non_pad_mask[:, 1:]

        valid_mask_text = original_valid_mask[:, 1:]  # [B, seq_len]（与vlm_text_embeds对齐）

        seq_len = vlm_text_embeds.shape[1]
        eos_token_id = self.vlm_tokenizer.eos_token_id
        # 找到每个样本中EOS的索引（从start_pos_input开始）
        for i in range(batch_size):
            # 输入文本中从start_pos_input开始的token
            sample_tokens = llava_inputs[i, 1:]  # [246]
            # 查找EOS位置（通常在最后）
            eos_indices = (sample_tokens == eos_token_id).nonzero().squeeze(-1)
            if len(eos_indices) > 0:
                last_eos_pos = eos_indices[-1].item()  # 最后一个EOS的位置
                # 截断掩码到EOS之前（或直接截断最后一个位置，因EOS通常在末尾）
                if valid_mask_text[i].shape[0] > seq_len:
                    valid_mask_text[i] = valid_mask_text[i, :seq_len]
                    print(f"样本{i}掩码截断EOS位置，长度从{valid_mask_text[i].shape[0]}→{seq_len}")

        # 批量截断（兜底）
        if valid_mask_text.shape[1] > seq_len:
            valid_mask_text = valid_mask_text[:, :seq_len]
        target_len = 248
        # 修正：拉伸掩码到248长度，仅对补充的Pad部分标记为False

        if seq_len < target_len:

            # 需要补充的Pad长度 = 248 - seq_len
            pad_length = target_len - seq_len
            # 生成Pad部分的掩码（全为False）
            pad_mask = torch.full((batch_size, pad_length), False, device=valid_mask_text.device)
            # 拼接：有效文本掩码（True） + Pad掩码（False）→ 总长度248
            valid_mask_stretched = torch.cat([valid_mask_text, pad_mask], dim=1)  # [B, 248]

        elif seq_len > target_len:

            # 截断到248，仅保留前248个位置的掩码
            valid_mask_stretched = valid_mask_text[:, :target_len]  # [B, 248]
        else:
            # 长度刚好为248，无需处理
            valid_mask_stretched = valid_mask_text  # [B, 248]

        # 拉伸嵌入到248（原有逻辑保持不变）
        stretched_emb = self._stretch_llava_seq(vlm_text_embeds)  # [B, 248, 4096]
        assert valid_mask_stretched.shape == (batch_size, target_len), \
            f"拉伸后掩码形状应为({batch_size}, {target_len})，实际为{valid_mask_stretched.shape}"
        return stretched_emb, valid_mask_stretched

    # 使用Transformers的CLIP处理文本
    def process_clip_text(self, texts):
        """
        返回：
        - text_hidden: [B, 77, clip_dim] 文本序列隐藏状态（用于补全解码器）
        - text_global_feat: [B, clip_dim] 全局文本特征（CLIP标准输出）
        """
        # 1. 文本预处理（使用Transformers的processor）
        inputs = self.clip_processor(
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_clip_length"],
            return_tensors="pt"
        ).to(device)

        # 2. 使用官方API获取全局文本特征
        with torch.no_grad():
            # 获取全局文本特征（官方API）
            text_global_feat = self.clip_model.get_text_features(**inputs)  # [B, 1024]
            text_global_feat = F.normalize(text_global_feat, dim=-1)  # 归一化

            # 为了获取text_hidden，仍需访问内部结构
            text_outputs = self.clip_model.text_model(**inputs)
            text_hidden = text_outputs.last_hidden_state  # [B,77,768]

        return text_hidden, text_global_feat

    # 使用Transformers的CLIP处理图像
    def process_clip_image(self, image_paths, sample_image_roots, return_patch=False):
        """
        返回：
        - img_global_feat: [B, clip_dim] 全局图像特征（CLIP标准输出）
        - img_patch_feat: [B, num_patches, clip_dim] 图像patch特征（用于细粒度损失）
        """
        # 1. 图像预处理
        images = []
        for img_path, img_root in zip(image_paths, sample_image_roots):
            img = Image.open(os.path.join(img_root, img_path)).convert("RGB")
            images.append(img)

        # 2. 统一预处理（使用Transformers的processor）
        inputs = self.clip_processor(
            images=images,
            return_tensors="pt"
        ).to(device)
        del images
        # 3. 使用官方API获取图像特征
        with torch.no_grad():
            # 获取全局图像特征（官方API）
            img_global_feat = self.clip_model.get_image_features(**inputs)  # [B, 1024]
            img_global_feat = F.normalize(img_global_feat, dim=-1)  # 归一化

            if return_patch:
                # 为了获取patch特征，仍需访问内部结构
                image_outputs = self.clip_model.vision_model(**inputs)
                # patch特征（去除CLS token）
                img_patch_feat = image_outputs.last_hidden_state[:, 1:, :]  # [B, 196, 768]
                # 投影到clip_dim（1024）用于细粒度匹配
                img_patch_feat = self.clip_model.visual_projection(img_patch_feat)  # [B,196,1024]

        if return_patch:
            return img_global_feat, img_patch_feat
        return img_global_feat

    # 损失函数（保持不变）
    def fine_grain_loss(self, text_token_feat, img_patch_feat):
        text_proj = F.normalize(text_token_feat, dim=-1)
        img_patch_feat = F.normalize(img_patch_feat, dim=-1)
        sim = torch.matmul(text_proj, img_patch_feat.transpose(-2, -1))
        return F.mse_loss(sim.mean(dim=1), torch.ones_like(sim.mean(dim=1)))

    def contrastive_loss(self, image_feats, text_feats):
        sim_matrix = torch.matmul(image_feats, text_feats.T) / self.temperature
        batch_size = image_feats.size(0)
        labels = torch.arange(batch_size, device=device)
        loss_img2txt = F.cross_entropy(sim_matrix, labels)
        loss_txt2img = F.cross_entropy(sim_matrix.T, labels)
        return (loss_img2txt + loss_txt2img) / 2

    # 前向传播（适配Transformers的CLIP输出）
    def forward(self, sample_image_roots, image_paths, long_texts, is_test=False):
        with torch.no_grad():
        # 1. CLIP截断文本的隐藏状态和全局特征
            clip_text_hidden, _ = self.process_clip_text(long_texts)  # [B,77,768]
            clip_text_hidden = clip_text_hidden.detach()
            # 投影到clip_dim（1024）供补全解码器使用
            clip_truncated_emb_proj = self.clip_model.text_projection(clip_text_hidden)  # [B,77,1024]

            # 2. LLaVA完整语义编码
            vlm_stretched_emb,valid_mask_stretched = self.process_vlm_inputs(sample_image_roots, image_paths, long_texts)  # [B,248,4096]
            vlm_stretched_emb = vlm_stretched_emb.detach()
        vlm_full_emb = self.llava_aggregator(
            vlm_emb=vlm_stretched_emb,
            valid_mask_stretched=valid_mask_stretched
        )  # [B,77,4096]
        vlm_full_emb = vlm_full_emb.detach()
        # 3. 补全解码
        completed_emb = self.completion_decoder(clip_truncated_emb_proj, vlm_full_emb)  # [B,77,1024]
        completed_global = completed_emb.mean(dim=1)  # [B,1024]
        completed_global = F.normalize(completed_global, dim=-1)

        if is_test:
            return completed_global

        # 4. 图像特征（全局+patch）
        clip_img_global, clip_img_patch = self.process_clip_image(image_paths, sample_image_roots, return_patch=True)

        # 5. 损失计算
        global_loss = self.contrastive_loss(clip_img_global, completed_global)
        fine_loss = self.fine_grain_loss(completed_emb, clip_img_patch)
        vlm_global_clip = self.completion_decoder.vlm_fuser(vlm_full_emb.mean(dim=1))  # [B,1024]
        vlm_global_clip = F.normalize(vlm_global_clip, dim=-1)
        consistency_loss = 1 - F.cosine_similarity(vlm_global_clip, completed_global, dim=-1).mean()

        total_loss = 0.5 * global_loss + 0.3 * fine_loss + 0.2 * consistency_loss
        return total_loss, completed_global


# 模型加载：使用Transformers的CLIP
def load_models(device):
    # 1. 加载LLaVA（不变）
    vlm_model_path = CONFIG["llava_model_path"]
    model_name = get_model_name_from_path(vlm_model_path)
    vlm_tokenizer, vlm_model, image_processor, _ = load_pretrained_model(
        vlm_model_path, None, model_name, device, torch_dtype=dtype
    )
    vlm_model.eval()

    # 2. 加载Transformers的CLIP（核心修改）
    clip_model = CLIPModel.from_pretrained(CONFIG["clip_model_name"]).to(device, dtype=dtype)
    clip_processor = CLIPProcessor.from_pretrained(CONFIG["clip_model_name"])  # 统一处理图像和文本
    clip_model.eval()

    return vlm_model, vlm_tokenizer, image_processor, clip_model, clip_processor


# 训练步骤（不变）
def train_step(model, sample_image_roots, image_paths, long_texts, optimizer):
    model.train()
    optimizer.zero_grad()

    loss, _ = model(sample_image_roots, image_paths, long_texts, is_test=False)

    loss.backward()

    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    # 1. 加载模型
    vlm_model, vlm_tokenizer, image_processor, clip_model, clip_processor = load_models(device)
    print(f"模型加载完成！CLIP特征维度：{clip_model.config.projection_dim}")

    # 2. 初始化对齐模块
    alignment_module = CLIPAlignmentModule(
        vlm_model=vlm_model,
        clip_model=clip_model,
        vlm_tokenizer=vlm_tokenizer,
        clip_processor=clip_processor,
        image_processor=image_processor
    ).to(device, dtype=dtype)
    print("对齐模块初始化完成")

    # 3. 加载数据集
    all_datasets = []
    for dataset_info in CONFIG["datasets"]:
        dataset = LLM_CoreTextDataset(
            json_path=dataset_info["data_json_path"],
            image_root=dataset_info["image_root"],
            clip_processor=clip_processor  # 使用Transformers的processor
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

    # 5. 优化器
    trainable_params = [
        # 补全解码器：复杂模块，用基础学习率
        {"params": alignment_module.completion_decoder.parameters(), "lr": CONFIG["learning_rate"]},
        # 聚合器：小模块，用更高学习率加速收敛
        {"params": alignment_module.llava_aggregator.parameters(), "lr": 2 * CONFIG["learning_rate"]},
        # 温度参数：敏感参数，用低学习率微调
        {"params": [alignment_module.temperature], "lr": CONFIG["learning_rate"] / 10}
    ]
    optimizer = optim.AdamW(trainable_params, lr=CONFIG["learning_rate"], weight_decay=0.01)

    # 6. 训练循环
    for epoch in range(CONFIG["epochs"]):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}")
        alignment_module.train()  # 确保模型处于训练模式

        for batch_idx, batch in enumerate(progress_bar):
            sample_image_roots = batch["sample_image_roots"]
            image_paths = batch["image_path"]
            long_texts = batch["long_text"]

            # 训练步骤（简化梯度清理）
            optimizer.zero_grad()  # 替代手动遍历参数清零，更高效
            loss, _ = alignment_module(
                sample_image_roots=sample_image_roots,
                image_paths=image_paths,
                long_texts=long_texts,
                is_test=False
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Batch Loss": f"{loss:.4f}"})

            # 日志输出（不变）
            if (batch_idx + 1) % CONFIG["log_interval"] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Avg Loss: {avg_loss:.4f}")

        # 保存模型（核心修改：包含聚合器参数）
        epoch_avg_loss = total_loss / len(dataloader)
        # 需保存的可训练参数键值：补全解码器、聚合器、温度参数
        trainable_keys = {"completion_decoder.", "llava_aggregator.", "temperature"}
        filtered_state_dict = {}
        for key, value in alignment_module.state_dict().items():
            if any(key.startswith(prefix) for prefix in trainable_keys):
                filtered_state_dict[key] = value
        save_path = os.path.join(CONFIG["output_dir"], f"alignment_epoch_{epoch + 1}_loss_{epoch_avg_loss:.4f}.pth")
        torch.save(filtered_state_dict, save_path)
        print(f"Epoch {epoch + 1} 模型保存至：{save_path}，大小：{os.path.getsize(save_path) / 1024 / 1024:.2f} MB\n")

        # 清理旧模型（不变）
        model_files = []
        pattern = r"alignment_epoch_(\d+)_loss_[\d.]+.pth"
        for file in os.listdir(CONFIG["output_dir"]):
            match = re.match(pattern, file)
            if match:
                model_files.append((int(match.group(1)), file))
        if len(model_files) > 2:
            for _, file_to_delete in sorted(model_files, reverse=True)[2:]:
                os.remove(os.path.join(CONFIG["output_dir"], file_to_delete))
                print(f"已删除旧模型：{file_to_delete}")


