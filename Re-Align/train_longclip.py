import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import CLIPTextModel, CLIPVisionModel, CLIPTokenizer
from tqdm import tqdm


# 改进1：静态偏向低索引PE的α计算逻辑
class StaticAlphaPositionEmbedding(nn.Module):
    def __init__(self, max_len, d_model, bias=True):
        super().__init__()
        self.d_model = d_model
        # 静态α：随索引增大指数衰减，强化低索引（早期token）的PE权重
        self.alpha = nn.Parameter(torch.exp(-torch.arange(max_len) * 0.1), requires_grad=False)  # 0.1为衰减系数

        # 原始CLIP位置编码（正弦余弦）
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        seq_len = x.size(0)
        # 应用静态α权重：低索引α值更高，强化早期token的位置信息
        weighted_pe = self.pe[:seq_len] * self.alpha[:seq_len].unsqueeze(-1)  # [seq_len, 1, d_model]
        x = x + weighted_pe  # 位置编码与文本特征相加
        return x


# 改进2&3：动态主成分筛选（风格相关性）+ 风格/内容token注意力加权
class StyleAwareTextEncoder(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", max_len=512, d_model=768):
        super().__init__()
        self.original_clip = CLIPTextModel.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.d_model = d_model

        # 替换原始位置编码为改进的静态α版本
        self.position_embedding = StaticAlphaPositionEmbedding(max_len, d_model)

        # 风格/内容注意力权重层：学习区分风格token和内容token
        self.style_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)  # 输出每个token的风格相关性权重
        )

        # 主成分筛选相关参数
        self.num_pcs = 32  # 主成分数量
        self.style_temporal_weight = nn.Parameter(torch.linspace(0.1, 1.0, 8), requires_grad=False)  # 时序加权系数

    def forward(self, text, style_token_mask=None):
        """
        text: 输入长文本
        style_token_mask: 可选，[batch_size, seq_len]，1表示风格token，0表示内容token（若未提供则自动学习）
        """
        # 1. 文本编码与位置嵌入
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(next(self.parameters()).device)
        attention_mask = inputs.attention_mask.to(next(self.parameters()).device)

        # 获取CLIP文本编码器的底层特征（未加最终LN和投影）
        outputs = self.original_clip.text_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, d_model]
        seq_len = hidden_states.size(1)

        # 应用改进的位置编码（静态α偏向低索引）
        hidden_states = hidden_states.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        hidden_states = self.position_embedding(hidden_states)
        hidden_states = hidden_states.permute(1, 0, 2)  # [batch_size, seq_len, d_model]

        # 2. 风格/内容token注意力加权（改进3）
        # 计算每个token的风格相关性权重
        style_weights = self.style_attention(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        # 若提供风格掩码，用掩码增强权重
        if style_token_mask is not None:
            style_weights = style_weights * style_token_mask + (1 - style_token_mask) * (-10.0)  # 抑制内容token
        style_weights = F.softmax(style_weights, dim=1)  # 归一化权重

        # 加权特征：放大风格相关token，抑制内容token
        weighted_hidden = hidden_states * style_weights.unsqueeze(-1)  # [batch_size, seq_len, d_model]

        # 3. 计算风格时序加权特征 F_text-temporal（用于主成分筛选）
        # 按token索引分块（模拟时序），应用时序权重
        chunk_size = seq_len // len(self.style_temporal_weight)
        chunks = torch.split(weighted_hidden, chunk_size, dim=1)[:len(self.style_temporal_weight)]  # 分块
        chunk_weights = self.style_temporal_weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, num_chunks, 1, 1]
        weighted_chunks = [chunk * chunk_weights[i] for i, chunk in enumerate(chunks)]
        F_text_temporal = torch.cat(weighted_chunks, dim=1).mean(dim=1)  # [batch_size, d_model]

        # 4. 动态主成分筛选（改进2：基于风格相关性而非固定排名）
        # 计算文本特征的主成分
        batch_size = weighted_hidden.size(0)
        flattened_features = weighted_hidden.reshape(batch_size * seq_len, -1)  # [N, d_model]
        mean = flattened_features.mean(dim=0, keepdim=True)
        centered = flattened_features - mean
        cov = torch.matmul(centered.T, centered) / (flattened_features.size(0) - 1)  # 协方差矩阵
        eigvals, eigvecs = torch.linalg.eigh(cov)  # 特征值和特征向量（主成分）
        pcs = eigvecs.T[:self.num_pcs]  # 取前num_pcs个主成分（未筛选）

        # 计算每个主成分与F_text_temporal的余弦相似度（风格相关性）
        pcs_norm = F.normalize(pcs, dim=1)  # [num_pcs, d_model]
        F_text_norm = F.normalize(F_text_temporal, dim=1)  # [batch_size, d_model]
        sim = torch.matmul(pcs_norm, F_text_norm.T).mean(dim=1)  # [num_pcs]，平均相似度

        # 按相似度动态筛选主成分（取前k个高相似度主成分）
        topk = int(self.num_pcs * 0.7)  # 保留70%高相似度主成分
        _, top_indices = torch.topk(sim, topk)
        selected_pcs = pcs[top_indices]  # [topk, d_model]

        # 5. 最终文本特征：加权特征+筛选后的主成分投影
        text_features = weighted_hidden.mean(dim=1)  # [batch_size, d_model]
        text_features = torch.matmul(text_features, selected_pcs.T)  # 投影到筛选后的主成分空间
        text_features = F.normalize(text_features, dim=1)  # 归一化

        return {
            "text_features": text_features,
            "style_weights": style_weights,
            "selected_pcs": selected_pcs,
            "F_text_temporal": F_text_temporal
        }


# 完整的改进版Long-CLIP模型
class ImprovedLongCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = StyleAwareTextEncoder()
        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.image_projection = nn.Linear(768, 512)  # 图像特征投影
        self.text_projection = nn.Linear(32, 512)  # 文本特征投影（32为筛选后主成分维度）
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, text, images, style_token_mask=None):
        # 文本编码（含改进逻辑）
        text_outputs = self.text_encoder(text, style_token_mask)
        text_features = self.text_projection(text_outputs["text_features"])  # [batch_size, 512]

        # 图像编码
        image_outputs = self.image_encoder(images)
        image_features = self.image_projection(image_outputs.pooler_output)  # [batch_size, 512]
        image_features = F.normalize(image_features, dim=1)

        # 对比损失计算（CLIP原逻辑）
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_features, image_features.T) * logit_scale
        logits_per_image = logits_per_text.T

        return {
            "logits_per_text": logits_per_text,
            "logits_per_image": logits_per_image,
            **text_outputs  # 包含中间特征
        }


# 训练代码示例
def train_improved_long_clip(train_dataset, epochs=10, batch_size=32, lr=5e-5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImprovedLongCLIP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            texts = batch["text"]
            images = batch["image"].to(device)
            style_masks = batch.get("style_mask", None)  # 可选的风格掩码
            if style_masks is not None:
                style_masks = style_masks.to(device)

            outputs = model(texts, images, style_masks)
            logits_per_text = outputs["logits_per_text"]
            logits_per_image = outputs["logits_per_image"]

            # 对比损失（CLIP标准损失）
            batch_size = logits_per_text.size(0)
            labels = torch.arange(batch_size, device=device)
            loss = (criterion(logits_per_text, labels) + criterion(logits_per_image, labels)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        # 保存中间状态特征（如筛选后的主成分、F_text_temporal）
        if (epoch + 1) % 2 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "selected_pcs": outputs["selected_pcs"].detach().cpu(),
                "F_text_temporal": outputs["F_text_temporal"].detach().cpu()
            }, f"improved_long_clip_epoch_{epoch + 1}.pth")

    return model


# 数据集示例（需根据实际数据格式实现）
class LongCLIPDataset(Dataset):
    def __init__(self, data):
        self.data = data  # 包含"text", "image", "style_mask"字段

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 使用示例
if __name__ == "__main__":
    # 假设已准备好训练数据
    dummy_data = [
        {
            "text": "复古风格的书房，胡桃木色书桌放在靠窗位置，书桌上有黄铜台灯和皮质笔记本...",
            "image": torch.randn(3, 224, 224),  # 示例图像张量
            "style_mask": torch.tensor([1, 1, 0, 0, 0, 1, 0, 0])  # 1表示风格token（如"复古风格"）
        }
        # ...更多数据
    ]
    dataset = LongCLIPDataset(dummy_data)
    model = train_improved_long_clip(dataset, epochs=10)
