import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import sys
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import CLIPProcessor  # 替换CLIPTokenizer，使用processor统一处理

# 配置GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 导入FineLIP的load_from_clip（根据实际路径调整）
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from model.finelip import load_from_clip

SAVE_DIR = "/data2/gaodz/attr_attention_result_3"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------- 定义必要的属性列表（与AttrDataset对应） --------------------------
REQUIRED_ATTRS = ['color', 'shape', 'material', 'style', 'object']  # 必须包含的属性


# -------------------------- 数据加载模块（替换为AttrDataset，兼容jsonlines） --------------------------
class AttrDataset(Dataset):
    def __init__(self, json_path, image_root, clip_processor):
        self.samples = []
        # 加载jsonlines格式（每行一个JSON对象）
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())  # 解析每行的JSON
                except json.JSONDecodeError:
                    continue  # 跳过格式错误的行
                # 检查必要的键是否存在
                if not all(k in sample for k in ["image", "long_text"]):
                    continue
                # 补全空白detail（确保包含所有必要属性）
                if "detail" not in sample:
                    sample["detail"] = {attr: [] for attr in REQUIRED_ATTRS}
                else:
                    for attr in REQUIRED_ATTRS:
                        if attr not in sample["detail"]:
                            sample["detail"][attr] = []
                self.samples.append(sample)
        self.image_root = image_root
        self.clip_processor = clip_processor  # 用于图像预处理和文本编码

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 加载图像
        img_path = os.path.join(self.image_root, sample["image"])
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            return self.__getitem__((idx + 1) % len(self))  # 跳过损坏图像
        # 返回原始样本（后续在collate_fn中处理增强）
        return {
            'image': image,
            'text': sample["long_text"],  # 全局文本
            'detail': sample["detail"]    # 属性详情
        }


# -------------------------- 模型定义模块（保持不变，确保使用扩展CLIP） --------------------------
class SimpleAttributeCLIP(nn.Module):
    def __init__(self, clip_model_name="ViT-L/14", num_attributes=len(REQUIRED_ATTRS), hidden_dim=512, run_finelip=False):
        super().__init__()
        # 加载扩展CLIP模型（含位置编码扩展）
        self.clip, self.clip_preprocess = load_from_clip(
            name=clip_model_name,
            run_finelip=run_finelip
        )
        assert self.clip.context_length == 248, "CLIP模型未正确扩展至248 token长度"

        self.num_attributes = num_attributes
        self.embed_dim = self.clip.text_projection.shape[1]

        # 解冻可学习位置编码
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.positional_embedding_res.requires_grad = True  # 可学习位置编码

        # 属性投影头
        self.attribute_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim)
            ) for _ in range(num_attributes)
        ])

    def forward(self, images, input_ids, attention_mask):
        # 图像预处理（使用CLIP的preprocess）
        pixel_values = torch.stack([self.clip_preprocess(img) for img in images]).to(device)

        # 获取CLIP特征（含可学习位置编码）
        image_features = self.clip.encode_image(pixel_values)
        text_features = self.clip.encode_text(input_ids)

        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 属性投影
        attr_image_features = []
        attr_text_features = []
        for projector in self.attribute_projectors:
            attr_img = projector(image_features)
            attr_img = F.normalize(attr_img, dim=-1)
            attr_image_features.append(attr_img)

            attr_txt = projector(text_features)
            attr_txt = F.normalize(attr_txt, dim=-1)
            attr_text_features.append(attr_txt)

        return {
            'global_image': image_features,
            'global_text': text_features,
            'attr_image': attr_image_features,
            'attr_text': attr_text_features
        }


# -------------------------- 训练模块（调整数据增强和collate_fn） --------------------------
def clip_contrastive_loss(image_feat, text_feat, temperature=0.07):
    logits = torch.matmul(image_feat, text_feat.t()) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_img = F.cross_entropy(logits, labels)
    loss_txt = F.cross_entropy(logits.t(), labels)
    return (loss_img + loss_txt) / 2


def train_one_epoch(model, dataloader, optimizer, scaler, attr2idx):
    model.train()
    total_loss = 0.0
    global_losses = []
    attr_losses = []

    for batch in tqdm(dataloader, desc="Training"):
        # 数据移至设备
        images = batch['image']  # PIL图像列表
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        is_global = batch['is_global'].to(device)
        attributes = batch['attribute'].to(device)

        # 前向传播
        with torch.cuda.amp.autocast():
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # 1. 全局对比损失（仅全局样本）
            global_mask = is_global
            global_loss = torch.tensor(0.0, device=device)
            if global_mask.any():
                global_img = outputs['global_image'][global_mask]
                global_txt = outputs['global_text'][global_mask]
                global_loss = clip_contrastive_loss(global_img, global_txt)

            # 2. 属性对比损失（仅属性样本）
            attr_mask = ~is_global
            attr_loss = torch.tensor(0.0, device=device)
            if attr_mask.any():
                attr_img_list = [f[attr_mask] for f in outputs['attr_image']]
                attr_txt_list = [f[attr_mask] for f in outputs['attr_text']]
                batch_attrs = attributes[attr_mask]

                for attr_idx in range(model.num_attributes):
                    mask = (batch_attrs == attr_idx)
                    if mask.any():
                        img_feat = attr_img_list[attr_idx][mask]
                        txt_feat = attr_txt_list[attr_idx][mask]
                        attr_loss += clip_contrastive_loss(img_feat, txt_feat)
                attr_loss /= model.num_attributes  # 平均各属性损失

            total_batch_loss = 0.7 * global_loss + 0.3 * attr_loss

        # 反向传播
        scaler.scale(total_batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # 记录损失
        total_loss += total_batch_loss.item()
        global_losses.append(global_loss.item())
        attr_losses.append(attr_loss.item())

    avg_loss = total_loss / len(dataloader)
    avg_global = np.mean(global_losses)
    avg_attr = np.mean(attr_losses)
    return avg_loss, avg_global, avg_attr


# -------------------------- 主函数（调整数据加载和增强逻辑） --------------------------
def main():
    # 数据配置
    data_config = {
        "datasets": [
            {
                "data_json_path": "/data2/gaodz/sam_data/coco_with_detail_partial.json",
                "image_root": "/data2/gaodz/train2014"
            },
            {
                "data_json_path": "/data2/gaodz/sam_data/wikiart_with_detail_partial.json",
                "image_root": "/data2/gaodz/WikiArt/OpenDataLab___WikiArt/raw/train_image/wikiart"
            },
            {
                "data_json_path": "/data2/gaodz/sharegpt4v/sharegpt4v_coco.json",
                "image_root": "/data2/gaodz/coco2017/PAI/COCO2017"
            }
        ]
    }

    # 超参数
    num_epochs = 6
    batch_size = 16
    projector_lr = 1e-4
    pos_emb_lr = 2e-4
    clip_model_name = "ViT-L/14"  # 使用large模型
    max_length = 248

    # 1. 初始化CLIP处理器（替代单独的tokenizer和preprocess）
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # 2. 加载多数据集（使用AttrDataset）
    datasets = []
    for cfg in data_config["datasets"]:
        dataset = AttrDataset(
            json_path=cfg["data_json_path"],
            image_root=cfg["image_root"],
            clip_processor=clip_processor  # 传入processor
        )
        print(f"加载数据集 {cfg['data_json_path']}，样本数：{len(dataset)}")
        datasets.append(dataset)
    combined_dataset = ConcatDataset(datasets)
    num_attributes = len(REQUIRED_ATTRS)
    attr2idx = {attr: i for i, attr in enumerate(REQUIRED_ATTRS)}  # 属性到索引的映射
    print(f"合并后总样本数：{len(combined_dataset)}，属性列表：{REQUIRED_ATTRS}")

    # 3. 数据加载器（自定义collate_fn，处理样本增强）
    def collate_fn(batch):
        """将原始样本增强为全局样本和属性样本"""
        enhanced = []
        for item in batch:
            image = item['image']
            global_text = item['text']
            detail = item['detail']

            # 1. 添加全局样本
            enhanced.append({
                'image': image,
                'text': global_text,
                'is_global': True,
                'attribute': -1  # 全局样本无属性
            })

            # 2. 添加属性样本（仅处理有值的属性）
            for attr in REQUIRED_ATTRS:
                attr_values = detail[attr]
                if attr_values:  # 仅当属性有值时添加
                    attr_text = f"This image has {', '.join(attr_values)} {attr}."
                    enhanced.append({
                        'image': image,
                        'text': attr_text,
                        'is_global': False,
                        'attribute': attr2idx[attr]
                    })

        # 对增强后的样本进行编码
        texts = [x['text'] for x in enhanced]
        inputs = clip_processor(
            text=texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return {
            'image': [x['image'] for x in enhanced],
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'is_global': torch.tensor([x['is_global'] for x in enhanced]),
            'attribute': torch.tensor([x['attribute'] for x in enhanced])
        }

    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # 4. 初始化模型
    model = SimpleAttributeCLIP(
        clip_model_name=clip_model_name,
        num_attributes=num_attributes,
        hidden_dim=512,
        run_finelip=False
    ).to(device)
    model = model.float()
    # 5. 优化器（含位置编码和投影头）
    optimizer = torch.optim.Adam([
        {'params': model.attribute_projectors.parameters(), 'lr': projector_lr},
        {'params': [model.clip.positional_embedding_res], 'lr': pos_emb_lr}
    ], weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    # 6. 训练循环
    for epoch in range(num_epochs):
        avg_loss, avg_global, avg_attr = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=scaler,
            attr2idx=attr2idx
        )
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"总损失: {avg_loss:.4f} | 全局对比损失: {avg_global:.4f} | 属性对比损失: {avg_attr:.4f}")

        # 保存模型
        if (epoch + 1) % 2 == 0:
            save_path = os.path.join(SAVE_DIR, f"attribute_clip_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss
            }, save_path)
            print(f"模型已保存至 {save_path}")


if __name__ == "__main__":
    main()