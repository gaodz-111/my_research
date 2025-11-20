#!/usr/bin/env python3
"""
train_router_multimodal.py

用途：
  - 从 JSON 数据集中读取样本（字段: id,image,question,label）
  - 用 LLaVA 提取 text pooled embedding（可替换为任意 text encoder）
  - 用 CLIP 提取 image embedding
  - 拼接 embedding 训练轻量 classifier（Router）
  - 支持 embedding 预计算与缓存以加速迭代

依赖:
  pip install torch torchvision transformers accelerate tqdm Pillow numpy

注意:
  - 如果你的 LLaVA 不支持 transformers 的默认接口，请替换 extract_text_embedding_batch 中的实现以适配你的 API。
  - CLIP 用的是 "openai/clip-vit-base-patch32"（稳定）。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from torch.utils.data import Dataset
from PIL import Image

from llava.mm_utils import tokenizer_image_token
from torch.nn.utils.rnn import pad_sequence
import torch

from peft import LoraConfig, get_peft_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import Trainer, TrainingArguments
from llava.model.builder import load_pretrained_model
import json
from torch.utils.data import DataLoader
from transformers import Trainer
import traceback
from llava.constants import IMAGE_TOKEN_INDEX

import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPProcessor,
    CLIPModel,
    get_linear_schedule_with_warmup,
)

# ---------------------------
# Config / Label mapping
# ---------------------------
LABELS = ["yesno","count","location","object","attribute","description","action"]
label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {v:k for k,v in label2id.items()}

# ---------------------------
# Dataset helpers
# ---------------------------
class SimpleRecordDataset(Dataset):
    """
    A dataset wrapper for precomputed embeddings.
    If embeddings are cached (np files), this class simply loads them.
    Otherwise, you'll call precompute functions first.
    """
    def __init__(self, embeddings_text_path, embeddings_img_path, labels_path, ids_path=None):
        # embeddings should be numpy arrays: (N, D_text), (N, D_img)
        self.text_emb = np.load(embeddings_text_path)   # (N, Dt)
        self.img_emb = np.load(embeddings_img_path)     # (N, Di)
        self.labels = np.load(labels_path)               # (N,)
        self.ids = np.load(ids_path) if ids_path and Path(ids_path).exists() else None

        assert len(self.text_emb) == len(self.img_emb) == len(self.labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        # return torch tensors
        t = torch.from_numpy(self.text_emb[idx]).float()
        v = torch.from_numpy(self.img_emb[idx]).float()
        y = int(self.labels[idx])
        sample_id = None
        if self.ids is not None:
            sample_id = str(self.ids[idx])
        return {"text_emb": t, "img_emb": v, "label": y, "id": sample_id}

def collate_fn(batch):
    text = torch.stack([b["text_emb"] for b in batch], dim=0)
    img  = torch.stack([b["img_emb"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    ids = [b["id"] for b in batch]
    return {"text_emb": text, "img_emb": img, "labels": labels, "ids": ids}

# ---------------------------
# Model: simple MLP classifier
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """双向 Cross-Attention + 残差 + LayerNorm"""
    def __init__(self, proj_dim, num_heads=4, dropout=0.1):
        super().__init__()
        # 文本关注图像
        self.cross_attn_t2i = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        # 图像关注文本
        self.cross_attn_i2t = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True, dropout=dropout)

        # 残差 + LayerNorm
        self.ln_t2i = nn.LayerNorm(proj_dim)
        self.ln_i2t = nn.LayerNorm(proj_dim)

        # Feed Forward (位置前馈网络)
        self.ffn_t = nn.Sequential(
            nn.Linear(proj_dim, proj_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 4, proj_dim),
            nn.Dropout(dropout)
        )
        self.ffn_i = nn.Sequential(
            nn.Linear(proj_dim, proj_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 4, proj_dim),
            nn.Dropout(dropout)
        )

        self.ln_ffn_t = nn.LayerNorm(proj_dim)
        self.ln_ffn_i = nn.LayerNorm(proj_dim)

    def forward(self, t, i):
        # t: [B, 1, D], i: [B, 1, D]
        # 文本关注图像
        t2i_out, _ = self.cross_attn_t2i(t, i, i)
        t = self.ln_t2i(t + t2i_out)  # 残差

        # 图像关注文本
        i2t_out, _ = self.cross_attn_i2t(i, t, t)
        i = self.ln_i2t(i + i2t_out)  # 残差

        # FFN + 残差
        t_ffn_out = self.ffn_t(t)
        t = self.ln_ffn_t(t + t_ffn_out)

        i_ffn_out = self.ffn_i(i)
        i = self.ln_ffn_i(i + i_ffn_out)

        return t, i


class RouterClassifier(nn.Module):
    def __init__(self, text_dim, img_dim, hidden_dim=1024, proj_dim=512, num_classes=len(LABELS),
                 dropout=0.1, num_heads=4, num_layers=4):
        super().__init__()
        # 投影到同一维度
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.img_proj = nn.Linear(img_dim, proj_dim)
        self.text_ln = nn.LayerNorm(proj_dim)
        self.img_ln = nn.LayerNorm(proj_dim)

        # 多层 CrossAttentionBlock
        self.layers = nn.ModuleList([
            CrossAttentionBlock(proj_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, text_emb, img_emb):
        # L2 归一化 + 投影 + 增加序列维度
        t = self.text_ln(self.text_proj(F.normalize(text_emb, p=2, dim=-1))).unsqueeze(1)  # [B, 1, D]
        i = self.img_ln(self.img_proj(F.normalize(img_emb, p=2, dim=-1))).unsqueeze(1)    # [B, 1, D]

        # 堆叠 CrossAttentionBlock
        for layer in self.layers:
            t, i = layer(t, i)

        # 融合
        fused = (t.squeeze(1) + i.squeeze(1)) / 2
        return self.classifier(fused)

        return self.classifier(fused)
from sentence_transformers import SentenceTransformer
# ---------------------------
# Embedding extraction utilities
# ---------------------------
def extract_text_embedding_batch_llava(model, tokenizer, questions: List[str], device, max_length=256, pooling="first"):
    """
    Extract pooled text embedding from LLaVA (or any causal LM model from transformers).
    pooling: "first" -> use first token; "mean" -> mean pooling over tokens (mask-aware)
    Returns: numpy array (B, H)
    """
    # model= SentenceTransformer('/data2/gaodz/bge_m3')
    # model.to(device)
    # pooled = model.encode(questions, batch_size=len(questions), device='cuda')
    enc = tokenizer(questions, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    # note: many causal LM models support output_hidden_states=True
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        # last hidden state:
        last_hidden = outputs.hidden_states[-1]   # (B, S, H)
        if pooling == "first":
            pooled = last_hidden[:, 0, :].cpu().numpy()
        elif pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (summed / denom).cpu().numpy()
        else:
            raise ValueError("Unsupported pooling")
    return pooled  # numpy (B, H)

def extract_image_embedding_clip(clip_model, clip_processor, image_paths: List[str], device, image_root=".", batch_size=16):
    """
    Extract CLIP pooled image embeddings (model.get_image_features).
    Returns numpy array (B, D_img)
    """
    feats = []
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        imgs = [Image.open(os.path.join(image_root, p)).convert("RGB") for p in batch_paths]
        inputs = clip_processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = clip_model.get_image_features(**inputs)  # (B, D)
            out = out.cpu().numpy()
        feats.append(out)
    return np.concatenate(feats, axis=0)

# ---------------------------
# Full preprocessing pipeline
# ---------------------------
def precompute_embeddings(records: List[Dict],
                          model_name_llava: str,
                          clip_model_name: str,
                          out_dir: str,
                          image_root: str,
                          device,
                          pooling="first"):
    """
    records: list of dict with keys: id,image,question,label
    Saves:
      - text_emb.npy
      - img_emb.npy
      - labels.npy
      - ids.npy (strings)
    """
    os.makedirs(out_dir, exist_ok=True)
    # tokenizer & model for llava
    model_path = "/data2/gaodz/llava-v1.6-vicuna-7b"
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_v1.6", device="cuda")
    model.eval()
    model.to(device)
    # Clip
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_model.eval()

    N = len(records)
    # prepare containers
    all_text_emb = []
    all_img_emb = []
    all_labels = []
    all_ids = []

    # We will extract text embeddings in batches
    B_text = 32
    questions = [r["question"] for r in records]
    print("Extracting text embeddings with LLaVA...")
    for i in tqdm(range(0, len(questions), B_text)):
        batch_q = questions[i:i+B_text]
        emb = extract_text_embedding_batch_llava(model, tokenizer, batch_q, device, pooling=pooling)
        all_text_emb.append(emb)
    all_text_emb = np.concatenate(all_text_emb, axis=0)
    print("text emb shape:", all_text_emb.shape)

    # image embeddings in batches
    image_paths = ['COCO_train2014_'+r["image"] for r in records]
    print("Extracting image embeddings with CLIP...")
    all_img_emb = extract_image_embedding_clip(clip_model, clip_processor, image_paths, device, image_root=image_root, batch_size=32)
    print("img emb shape:", all_img_emb.shape)

    # labels and ids
    for r in records:
        all_labels.append(label2id[r["label"]])
        all_ids.append(r["id"])

    all_labels = np.array(all_labels, dtype=np.int64)
    all_ids = np.array(all_ids, dtype=np.str_)

    # Save
    np.save(os.path.join(out_dir, "text_emb.npy"), all_text_emb)
    np.save(os.path.join(out_dir, "img_emb.npy"), all_img_emb)
    np.save(os.path.join(out_dir, "labels.npy"), all_labels)
    np.save(os.path.join(out_dir, "ids.npy"), all_ids)
    print("Saved cached embeddings to", out_dir)
    return os.path.join(out_dir, "text_emb.npy"), os.path.join(out_dir, "img_emb.npy"), os.path.join(out_dir, "labels.npy"), os.path.join(out_dir, "ids.npy")

# ---------------------------
# Training loop
# ---------------------------
def train_classifier(text_emb_path, img_emb_path, labels_path, ids_path,
                     out_dir, epochs=8, batch_size=64, lr=1e-4, weight_decay=1e-2,
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                     hidden_dim=1024, val_split=0.1, seed=42):
    # load data
    text_emb = np.load(text_emb_path)
    img_emb  = np.load(img_emb_path)
    labels   = np.load(labels_path)
    ids      = np.load(ids_path) if ids_path and Path(ids_path).exists() else None
    N = len(labels)
    assert text_emb.shape[0] == img_emb.shape[0] == labels.shape[0]
    # shuffle and split
    rng = np.random.RandomState(seed)
    idxs = np.arange(N)
    rng.shuffle(idxs)
    n_val = max(1, int(N * val_split))
    val_idx = idxs[:n_val]
    train_idx = idxs[n_val:]

    # create temp files for train/val (or use in-memory arrays)
    train_text = text_emb[train_idx]
    train_img  = img_emb[train_idx]
    train_labels = labels[train_idx]
    val_text = text_emb[val_idx]
    val_img  = img_emb[val_idx]
    val_labels = labels[val_idx]

    # Save train/val caches to temp files
    work_dir = Path(out_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    train_text_path = work_dir / "train_text_emb.npy"
    train_img_path  = work_dir / "train_img_emb.npy"
    train_labels_path = work_dir / "train_labels.npy"
    val_text_path = work_dir / "val_text_emb.npy"
    val_img_path  = work_dir / "val_img_emb.npy"
    val_labels_path = work_dir / "val_labels.npy"

    np.save(train_text_path, train_text)
    np.save(train_img_path, train_img)
    np.save(train_labels_path, train_labels)
    np.save(val_text_path, val_text)
    np.save(val_img_path, val_img)
    np.save(val_labels_path, val_labels)

    train_ds = SimpleRecordDataset(str(train_text_path), str(train_img_path), str(train_labels_path), ids_path=None)
    val_ds   = SimpleRecordDataset(str(val_text_path), str(val_img_path), str(val_labels_path), ids_path=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    text_dim = train_text.shape[1]
    img_dim  = train_img.shape[1]
    model = RouterClassifier(text_dim=text_dim, img_dim=img_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05*total_steps), num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    best_ckpt = None

    print("Starting training: epochs:", epochs, "train size:", len(train_ds), "val size:", len(val_ds))
    import matplotlib.pyplot as plt

    train_losses = []
    val_accs = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"train epoch {epoch}"):
            text_emb_batch = batch["text_emb"].to(device)
            img_emb_batch  = batch["img_emb"].to(device)
            labels_batch   = batch["labels"].to(device)

            logits = model(text_emb_batch, img_emb_batch)
            loss = loss_fn(logits, labels_batch)
            optimizer.zero_grad()
            try:
                loss.backward()
            except RuntimeError as e:
                print("RuntimeError during backward:", e)
                # 再次检查常见问题
                print("logits.shape:", logits.shape)
                print("labels.shape:", labels.shape, "labels.dtype:", labels.dtype)
                print("labels.min/max:", labels.min().item(), labels.max().item())
                print("num_classes (logits.size(-1)):", logits.size(-1))
                # optional: save offending batch for postmortem
                torch.save({
                    'logits': logits.detach().cpu(),
                    'labels': labels.detach().cpu(),
                    'inputs': inputs.detach().cpu() if 'inputs' in locals() else None,
                }, "/tmp/failing_batch.pt")
                raise
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * labels_batch.size(0)
        avg_loss = running_loss / len(train_ds)

        # eval
        model.eval()
        total, correct = 0, 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                text_emb_batch = batch["text_emb"].to(device)
                img_emb_batch  = batch["img_emb"].to(device)
                labels_batch   = batch["labels"].to(device)

                logits = model(text_emb_batch, img_emb_batch)
                preds = logits.argmax(dim=-1)

                total += labels_batch.size(0)
                correct += (preds == labels_batch).sum().item()
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels_batch.cpu().numpy())
        val_acc = correct / total
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f} val_acc={val_acc:.4f}")
        # train_losses.append(avg_loss)
        # val_accs.append(val_acc)
        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt = work_dir / "best_router.pt"
            torch.save({"model_state": model.state_dict(),
                        "text_dim": text_dim,
                        "img_dim": img_dim,
                        "hidden_dim": hidden_dim},
                       best_ckpt)
            print("Saved best checkpoint:", best_ckpt)
    print("Training done. Best val acc:", best_val_acc)
    return best_ckpt
    # plt.figure(figsize=(8, 5))
    # epochs_range = range(epochs)
    # plt.plot(epochs_range, train_losses, label="Train Loss", marker='o')
    # plt.plot(epochs_range, val_accs, label="Val Accuracy", marker='s')
    #
    # plt.xlabel("Epoch")
    # plt.ylabel("Value")
    # plt.title("Training Loss and Validation Accuracy")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("loss_acc_curve.png", dpi=300)  # 保存为图片

# ---------------------------
# CLI main
# ---------------------------
def parse_records_from_json(json_path: str):
    """
    Expect JSON list or newline delimited json objects.
    Each record should contain at least: id, image, conversations (list)
    We'll extract question as the last human message or first human message.
    Also requires 'label' key for supervised router training.
    """
    records = []
    with open(json_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        try:
            data = json.loads(content)
        except Exception:
            # try ndjson
            data = [json.loads(line) for line in content.splitlines() if line.strip()]
    for r in data:
        # try to extract question from conversations

        q = None
        if "question" in r:
            q = r["question"]
        elif "conversations" in r:
            # prefer last human message
            convs = r["conversations"]
            # find last from human
            for msg in reversed(convs):
                if msg.get("from","").lower() == "human":
                    q = msg.get("value", "").strip()
                    break
            if q is None and len(convs)>0:
                q = convs[0].get("value","").strip()
        else:
            raise ValueError("Record missing 'conversations' or 'question' field")
        if q is None:
            q = ""
        # label should exist
        label = r.get("question_type", None)
        if label is None:
            raise ValueError(f"Record {r.get('id')} missing 'label' field")
        records.append({"id": r.get("id",""), "image": r.get("image",""), "question": q, "label": label})
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", type=str, default="/data2/gaodz/Re-Align/combined_data.json",required=True, help="Path to JSON dataset (list or NDJSON). Each record needs id,image,conversations OR question, and label.")
    parser.add_argument("--image_root", type=str, default="/data2/gaodz/train2014",required=True, help="Root dir where image files are located (image paths are relative to this).")
    parser.add_argument("--llava_model", type=str, default="/data2/gaodz/llava-v1.6-vicuna-7b", help="LLaVA model name or path (transformers compatible).")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14-336", help="CLIP model name.")
    parser.add_argument("--cache_dir", type=str, default="./cache_router", help="Where to store embeddings")
    parser.add_argument("--precompute", action="store_true", help="If set, precompute embeddings and exit (useful for caching).")
    parser.add_argument("--pooling", type=str, default="first", choices=["first","mean"], help="How to pool text embeddings")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    records = parse_records_from_json(args.data_json)
    print("Loaded", len(records), "records")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    text_emb_path = cache_dir / "text_emb.npy"
    img_emb_path = cache_dir / "img_emb.npy"
    labels_path  = cache_dir / "labels.npy"
    ids_path     = cache_dir / "ids.npy"

    # Precompute if needed or missing
    if args.precompute or not (text_emb_path.exists() and img_emb_path.exists() and labels_path.exists()):
        print("Precomputing embeddings...")
        te, ie, labp, idp = precompute_embeddings(records,
                                                  model_name_llava=args.llava_model,
                                                  clip_model_name=args.clip_model,
                                                  out_dir=str(cache_dir),
                                                  image_root=args.image_root,
                                                  device=device,
                                                  pooling=args.pooling)
        print("Embeddings computed:", te, ie, labp)
        if args.precompute:
            print("Precompute only; exiting.")
            return

    # Train classifier
    best_ckpt = train_classifier(str(text_emb_path), str(img_emb_path), str(labels_path), str(ids_path),
                                 out_dir=str(cache_dir), epochs=args.epochs, batch_size=args.batch_size,
                                 lr=args.lr, device=device, hidden_dim=args.hidden_dim, val_split=args.val_split)
    print("Done. Best checkpoint:", best_ckpt)

if __name__ == "__main__":
    main()
