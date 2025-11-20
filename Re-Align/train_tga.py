# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
#
# import sys
# import json
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from tqdm import tqdm
#
# from peft import LoraConfig, get_peft_model, TaskType
#
# # ===== LLaVA 依赖 =====
# parent_dir = os.path.abspath("./llava")  # 指向本机 llava 目录
# sys.path.append(parent_dir)
#
# from conversation import conv_templates
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import tokenizer_image_token
# from llava.constants import IMAGE_TOKEN_INDEX
#
#
# # =========================
# # 数据集
# # =========================
# class MultimodalTGADataset(Dataset):
#     def __init__(self, json_file, image_folder, tokenizer, image_processor, max_length=512):
#         self.image_folder = image_folder
#         self.tokenizer = tokenizer
#         self.image_processor = image_processor
#         self.max_length = max_length
#
#         self.data = []
#         with open(json_file, "r", encoding="utf-8") as f:
#             for line in f:
#                 self.data.append(json.loads(line))
#
#         self.conv_mode = "vicuna_v1"
#         self.conv_template = conv_templates[self.conv_mode]
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         qs = item["conversations"][0]["value"]        # 用户问题
#         answer = item["conversations"][1]["value"]    # 模型回答
#
#         conv = self.conv_template.copy()
#         conv.append_message(conv.roles[0], qs)
#         conv.append_message(conv.roles[1], answer)
#         prompt = conv.get_prompt()
#
#         input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
#         image_path = os.path.join(self.image_folder, "COCO_train2014_" + item["image"])
#         image = Image.open(image_path).convert("RGB")
#         image_tensor = self.image_processor(image, return_tensors="pt")["pixel_values"][0]
#
#         attn_target = torch.tensor(item["attention_distribution"], dtype=torch.float16)
#         important_idx = None
#         if "important_q_idx" in item and item["important_q_idx"] is not None:
#             important_idx = torch.tensor(item["important_q_idx"], dtype=torch.long)
#
#         return {
#             "input_ids": input_ids,
#             "image": image_tensor,
#             "attention_distribution": attn_target,
#             "important_q_idx": important_idx
#         }
#
#
# # =========================
# # collate_fn
# # =========================
# def collate_fn(batch, pad_id=0):
#     input_ids_list = [item["input_ids"].squeeze(0) for item in batch]
#     input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
#     attention_mask = torch.nn.utils.rnn.pad_sequence([torch.ones_like(x) for x in input_ids_list], batch_first=True, padding_value=0)
#     images = torch.stack([item["image"] for item in batch], dim=0)
#     attn_targets = [item["attention_distribution"] for item in batch]
#     important_idx_list = [item["important_q_idx"] for item in batch]
#
#     return {
#         "input_ids": input_ids_padded,
#         "attention_mask": attention_mask,
#         "image": images,
#         "attention_distribution": attn_targets,
#         "important_q_idx": important_idx_list
#     }
#
# class TGALoss(nn.Module):
#     def __init__(self, eps: float = 1e-6, smooth: float = 1e-6, verbose: bool = False):
#         """
#         eps   : 防止 log(0) / 除0 的小常数
#         smooth: (保留，用于将来扩展平滑策略，目前不自动平滑全零行)
#         verbose: 是否打印 debug 信息
#         """
#         super().__init__()
#         self.eps = float(eps)
#         self.smooth = float(smooth)
#         self.verbose = bool(verbose)
#         self.last_valid_idx = None
#
#     def forward(self, pred_maps: torch.Tensor, tgt_maps: torch.Tensor) -> torch.Tensor:
#         """
#         只做 KL 计算（在 tgt_maps 非全零行上）
#         pred_maps: [N, L_img]  模型预测注意力（可以是未归一化或已归一化）
#         tgt_maps : [N, L_img]  目标注意力（可能为 mask/计数/概率）
#         返回: scalar loss
#         """
#         device = pred_maps.device
#
#         # 强制 float16 提高数值稳定性
#         pred = pred_maps.float()
#         tgt  = tgt_maps.float()
#
#         # 找出 tgt 中非全零行
#         row_sums = tgt.sum(dim=-1)  # [N]
#         valid_mask = row_sums > 0.0
#         valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
#
#         # 记录用于调试
#         if valid_idx.numel() > 0:
#             self.last_valid_idx = valid_idx.detach().cpu()
#         else:
#             self.last_valid_idx = torch.empty(0, dtype=torch.long)
#
#         if self.verbose:
#             total_rows = tgt.size(0)
#             valid_cnt = valid_idx.numel()
#             print(f"[TGALoss] total_rows={total_rows}, valid_rows={valid_cnt}")
#
#         # 若无有效行，返回 0（可求导）
#         if valid_idx.numel() == 0:
#             return torch.tensor(0.0, device=device, dtype=pred.dtype, requires_grad=True)
#
#         # 裁剪到有效行
#         pred_sel = pred.index_select(dim=0, index=valid_idx)  # [N_valid, L_img]
#         tgt_sel  = tgt.index_select(dim=0, index=valid_idx)   # [N_valid, L_img]
#
#         # 加 eps 并逐行归一化为概率分布（clamp + renormalize）
#         pred_sel = pred_sel.clamp_min(self.eps)
#         pred_sel = pred_sel / (pred_sel.sum(dim=-1, keepdim=True) + self.eps)
#
#         tgt_sel  = tgt_sel.clamp_min(self.eps)
#         tgt_sel  = tgt_sel / (tgt_sel.sum(dim=-1, keepdim=True) + self.eps)
#
#         # 计算 KL(P||Q) 每行，然后平均
#         kl_per_row = (tgt_sel * (tgt_sel.log() - pred_sel.log())).sum(dim=-1)  # [N_valid]
#         loss = kl_per_row.mean()
#
#         if self.verbose:
#             print(f"[TGALoss] mean KL per-row: {loss.item():.6e}")
#
#         return loss
#
# # =========================
# # 定位 <image> 与 ASSISTANT:
# # =========================
# def find_spans(input_ids_row: torch.Tensor, image_token_id: int, assistant_token_ids: list, img_len_fallback: int = 577):
#     """
#     返回：
#       image_pos: <image> token 在序列中位置
#       img_len: 图像 token 数
#       q_start, q_end: 问题 token 的半开区间 [q_start, q_end)
#     """
#     # --- 定位 <image> ---
#     image_pos_tensor = (input_ids_row == image_token_id).nonzero(as_tuple=False)
#     if image_pos_tensor.numel() == 0:
#         raise ValueError("未找到 <image> token")
#     image_pos = int(image_pos_tensor[0, 0].item())
#
#     # --- ASSISTANT: token 序列 ---
#     L = input_ids_row.size(0)
#     seq_len = len(assistant_token_ids)
#     start_assist = None
#
#     ###i从<image>开始
#     for i in range(image_pos  , L - seq_len + 1):
#
#         if input_ids_row[i:i + seq_len].tolist() == assistant_token_ids:
#             # 通过计算应该是i + 577 -1
#             start_assist = i + 576
#
#             break
#     if start_assist is None:
#         start_assist = L  # 没找到 ASSISTANT: 就用序列末尾
#
#     # 图像 token 长度固定或兜底
#     img_len = img_len_fallback
#
#     # 问题 token：从 <image> 后的图像 token末尾到 ASSISTANT: 开始
#     q_start = image_pos + img_len
#     q_end = start_assist
#
#     # 避免 q_end <= q_start
#     if q_end <= q_start:
#         q_end = q_start + 1  # 至少包含 1 个 token
#
#     return image_pos, img_len, q_start, q_end
#
# import numpy as np
# # =========================
# # forward_with_tga
# # =========================
# def forward_with_tga(model, batch, tokenizer, device="cuda", lambda_tga=0.1, tga_layers="last_4"):
#     input_ids = batch["input_ids"].to(device)
#     images = batch["image"].to(device, dtype=torch.float16, non_blocking=True)
#     attn_targets_list = batch["attention_distribution"]
#     important_idx_list = batch["important_q_idx"]
#
#     # LM labels: text token 为原值，其余 -100
#     labels = input_ids.clone()
#     labels[input_ids == IMAGE_TOKEN_INDEX] = -100
#     if tokenizer.pad_token_id is not None:
#         labels[input_ids == tokenizer.pad_token_id] = -100
#     else:
#         labels[input_ids == 0] = -100
#     #使用teacher forcing，问题和答案一起输入模型
#     outputs = model(
#         input_ids=input_ids,
#         images=images.to(device, dtype=torch.float16, non_blocking=True),
#         labels=labels,
#         output_attentions=True,
#         return_dict=True
#     )
#     lm_loss = outputs.loss
#     # print("lm_loss:", lm_loss)
#     attn_stack = torch.stack(outputs.attentions, dim=0)
#     attn_head_mean = attn_stack.mean(dim=2)
#     if tga_layers == "all":
#         attn_mean = attn_head_mean.mean(dim=0)
#     elif tga_layers.startswith("last_"):
#         k = int(tga_layers.split("_")[1])
#         attn_mean = attn_head_mean[-k:].mean(dim=0)
#     else:
#         attn_mean = attn_head_mean.mean(dim=0)
#
#     B, L, _ = attn_mean.shape
#     image_token_id = IMAGE_TOKEN_INDEX
#     # ASSISTANT: 分词结果
#     assistant_token_ids = [319, 1799, 9047, 13566, 29901]
#
#     tga_loss_fn = TGALoss()
#     per_sample_losses = []
#     for b in range(B):
#         ids_row = input_ids[b]
#
#         len_q = len(attn_targets_list[b])
#
#         image_pos, img_len, q_start, q_end = find_spans(ids_row, image_token_id, assistant_token_ids)
#
#         total_text_tokens = attn_mean.shape[1]  # 注意力矩阵的行数就是文本 token 数
#
#         qa = attn_mean[b, q_end - len_q:q_end, image_pos:image_pos + img_len]
#         #对qa归一化
#         qa = torch.softmax(qa.float(), dim=-1)
#
#         tgt = attn_targets_list[b].to(device)
#         imp_idx = important_idx_list[b]
#
#         # 调试打印
#         # print(f"[DEBUG] sample {b}")
#         # print(f"  image_pos={image_pos}, img_len={img_len}, q_end={q_end}, len_q={len_q}")
#         # print(f"  qa.shape={qa.shape}, tgt.shape={tgt.shape}")
#         # print(f"  qa.sum()={qa.sum().item():.6f}, tgt.sum()={tgt.sum().item():.6f}")
#         # print(f"  qa.min={qa.min().item():.6f}, qa.max={qa.max().item():.6f}")
#         # print(f"  tgt.min={tgt.min().item():.6f}, tgt.max={tgt.max().item():.6f}")
#         # if torch.isnan(qa).any() or torch.isnan(tgt).any():
#         #     print("[WARN] NaN detected in qa or tgt")
#         # if torch.all(qa == 0):
#         #     print("[WARN] qa 全 0")
#         # if torch.all(tgt == 0):
#         #     print("[WARN] tgt 全 0")
#
#         if imp_idx is not None:
#             imp_idx = imp_idx.to(device)
#             valid_mask = (imp_idx >= 0) & (imp_idx < qa.size(0))
#             imp_idx = imp_idx[valid_mask]
#             print(f"  imp_idx (after mask): {imp_idx.tolist()}")
#             if imp_idx.numel() == 0:
#                 print("[WARN] imp_idx 有效索引为空，跳过样本")
#                 per_sample_losses.append(torch.zeros((), device=device))
#                 continue
#             qa = qa.index_select(dim=0, index=imp_idx)
#             print(f"  qa.shape after imp_idx select: {qa.shape}")
#             if tgt.size(0) != qa.size(0):
#                 raise ValueError(f"目标行数与 gather 后不一致: tgt={tgt.size()} vs qa={qa.size()}")
#         else:
#             if tgt.size(0) != qa.size(0):
#                 raise ValueError(f"目标行数应与问题行数一致: tgt={tgt.size()} vs qa={qa.size()}")
#
#         # 计算 loss 前再检查一次
#         if torch.isnan(qa).any() or torch.isnan(tgt).any():
#             print("[ERROR] 计算 loss 前发现 NaN，跳过样本")
#             per_sample_losses.append(torch.zeros((), device=device))
#             continue
#
#         per_sample_losses.append(tga_loss_fn(qa, tgt))
#
#     tga_loss = torch.stack(per_sample_losses).mean()
#
#     total_loss = lm_loss + lambda_tga * tga_loss
#
#
#     print("total_loss:", total_loss)
#     return total_loss, lm_loss, tga_loss
#
#
# # =========================
# # 主程序
# # =========================
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     model_path = "/data2/gaodz/llava-v1.6-vicuna-7b"
#     tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_v1.6", device=device)
#
#     # LoRA
#     lora_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         r=16,
#         lora_alpha=32,
#         target_modules=["q_proj", "k_proj"],
#         lora_dropout=0.05,
#         bias="none",
#         inference_mode=False
#     )
#     model = get_peft_model(model, lora_config)
#     model.train().to(device)
#
#     dataset = MultimodalTGADataset(
#         json_file="/data2/gaodz/Re-Align/combined_data_token_filter_mark_t2_soft.json",
#         image_folder="/data2/gaodz/train2014",
#         tokenizer=tokenizer,
#         image_processor=image_processor,
#         max_length=512
#     )
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
#
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
#
#     num_epochs = 2
#     for epoch in range(num_epochs):
#         pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
#         for batch in pbar:
#             optimizer.zero_grad(set_to_none=True)
#             try:
#                 total_loss, lm_loss, tga_loss = forward_with_tga(model, batch, tokenizer, device=device, lambda_tga=0.1, tga_layers="last_8")
#             except Exception as e:
#                 print("[WARN] forward 出错：", str(e))
#                 continue
#             total_loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
#             optimizer.step()
#             pbar.set_postfix({"LM": f"{lm_loss.item():.4f}", "TGA": f"{tga_loss.item():.4f}"})
#
#     save_dir = "/data2/gaodz/tga_lora_2_layer4"
#     os.makedirs(save_dir, exist_ok=True)
#     model.save_pretrained(save_dir)
#     print(f"[OK] LoRA 已保存到 {save_dir}")
#


#
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType
import random

# ===== LLaVA 依赖 =====
parent_dir = os.path.abspath("./llava")  # 指向本机 llava 目录
sys.path.append(parent_dir)
from conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX


# =========================
# Dataset
class MultimodalTGADataset(Dataset):
    def __init__(self, json_file, image_folder, tokenizer, image_processor, max_length=512):
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

        self.data = []
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.conv_mode = "vicuna_v1"
        self.conv_template = conv_templates[self.conv_mode]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        qs = item["conversations"][0]["value"]  # 用户问题
        answer = item["conversations"][1]["value"]  # 模型回答

        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        image_path = os.path.join(self.image_folder, "COCO_train2014_" + item["image"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_processor(image, return_tensors="pt")["pixel_values"][0]
        que_token_list = item.get("question_tokens", [])

        important_idx = None
        if "important_q_idx" in item and item["important_q_idx"] is not None:
            important_idx = torch.tensor(item["important_q_idx"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "image": image_tensor,
            "question_tokens": que_token_list,
            "important_q_idx": important_idx,
        }


# =========================
# collate_fn
# =========================
def collate_fn(batch, pad_id=0):
    input_ids_list = [item["input_ids"].squeeze(0) for item in batch]
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence([torch.ones_like(x) for x in input_ids_list], batch_first=True,
                                                     padding_value=0)
    images = torch.stack([item["image"] for item in batch], dim=0)
    attn_targets = [item["question_tokens"] for item in batch]
    important_idx_list = [item["important_q_idx"] for item in batch]

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "image": images,
        "question_tokens": attn_targets,
        "important_q_idx": important_idx_list,
    }


# =========================
# Subword to token mapping
# =========================
def get_subword_to_token_mapping(question_tokens, original_tokens, max_extra_chars=5):
    mapping = [-1] * len(question_tokens)
    visited = [False] * len(question_tokens)
    qt_lower = [t.lower() for t in question_tokens]
    L_question = len(question_tokens)

    for i in range(L_question):
        if visited[i]:
            continue
        matched = False
        for ft_idx, ft in enumerate(original_tokens):
            ft_lower = ft.lower()
            max_span = min(L_question - i, len(ft_lower) + max_extra_chars)
            for span in range(1, max_span + 1):
                concat_tokens = "".join(qt_lower[i:i + span])
                if concat_tokens == ft_lower:
                    for j in range(i, i + span):
                        mapping[j] = ft_idx
                        visited[j] = True
                    matched = True
                    break
            if matched:
                break
    return mapping


# =========================
# find_spans
# =========================
def find_spans(input_ids_row: torch.Tensor, image_token_id: int, assistant_token_ids: list,
               img_len_fallback: int = 577):
    """
    返回：
      image_pos: <image> token 在序列中位置
      img_len: 图像 token 数
      q_start, q_end: 问题 token 的半开区间 [q_start, q_end)
    """
    # --- 定位 <image> ---
    image_pos_tensor = (input_ids_row == image_token_id).nonzero(as_tuple=False)
    if image_pos_tensor.numel() == 0:
        raise ValueError("未找到 <image> token")
    image_pos = int(image_pos_tensor[0, 0].item())

    # --- ASSISTANT: token 序列 ---
    L = input_ids_row.size(0)
    seq_len = len(assistant_token_ids)
    start_assist = None

    ###i从<image>开始
    for i in range(image_pos, L - seq_len + 1):

        if input_ids_row[i:i + seq_len].tolist() == assistant_token_ids:
            # 通过计算应该是i + 577 -1
            start_assist = i + 576

            break
    if start_assist is None:
        start_assist = L  # 没找到 ASSISTANT: 就用序列末尾

    # 图像 token 长度固定或兜底
    img_len = img_len_fallback

    # 问题 token：从 <image> 后的图像 token末尾到 ASSISTANT: 开始
    q_start = image_pos + img_len
    q_end = start_assist

    # 避免 q_end <= q_start
    if q_end <= q_start:
        q_end = q_start + 1  # 至少包含 1 个 token

    return image_pos, img_len, q_start, q_end


# =========================
# 注意力互信息计算
# =========================
def attention_mutual_information(attn_qv, attn_vq, eps=1e-8):
    """
    attn_qv: [T_q, T_v] 文到图注意力矩阵（已归一化行和为1，可直接softmax输出）
    attn_vq: [T_v, T_q] 图到文注意力矩阵（已归一化行和为1）
    返回：
        mi_val_q: 文本模态的互信息率
        mi_val_i: 图像模态的互信息率
    """
    T_q, T_v = attn_qv.shape

    # 构造联合分布近似
    joint = 0.5 * (attn_qv + attn_vq.T)  # [T_q, T_v]
    joint = joint / (joint.sum() + eps)  # 归一化为概率

    # 边缘概率
    p_q = joint.sum(dim=1, keepdim=True)  # [T_q,1]
    p_v = joint.sum(dim=0, keepdim=True)  # [1,T_v]

    # 避免 log(0)
    joint_safe = joint + eps
    p_q_safe = p_q + eps
    p_v_safe = p_v + eps

    # 互信息计算
    mi_matrix = joint * torch.log(joint / (p_q * p_v + eps) + eps)
    mi_val = mi_matrix.sum()

    # 熵
    H_q = -torch.sum(p_q * torch.log(p_q + eps))
    H_v = -torch.sum(p_v * torch.log(p_v + eps))

    # 互信息率（NMI）
    nmi_val_que = mi_val / ((H_q) + eps)  # 文本模态的互信息率
    nmi_val_img = mi_val / ((H_v) + eps)  # 图像模态的互信息率
    return nmi_val_que, nmi_val_img

#梯度平滑
def smooth_attention(attn_matrix, alpha=0.01):
    """添加均匀分布的平滑项，减少极端值影响"""
    T1, T2 = attn_matrix.shape
    # 添加一个很小的均匀分布（如alpha=0.01）
    smooth = torch.full_like(attn_matrix, alpha / T2)
    return attn_matrix + smooth
#梯度抬升
def stable_row_norm(attn_matrix, eps=1e-8, min_sum=0.1):
    """确保每行和不小于min_sum，避免分母过小"""
    row_sums = attn_matrix.sum(dim=1, keepdim=True)  # [T, 1]
    # 若行和小于min_sum，强制抬升至min_sum
    row_sums = torch.max(row_sums, torch.tensor(min_sum, device=row_sums.device))
    return attn_matrix / (row_sums + eps)


class AMSSRebalanceTrainer:
    def __init__(self, model, tokenizer, lambda_asa=0.1, tga_layers="last_4",
                 subnet_frac=0.6, mir_bins=8, mir_alpha=4.0, amss_a=1e-2, min_prob=1e-3, tau=0.5):
        """
        amss_a: AMSS+ 的平滑常数 a（用于避免 1/p 太大），论文建议小正数
        min_prob: 最小采样概率下限，防止未采中参数 p_j 太小
        """
        self.model = model
        self.tokenizer = tokenizer
        self.lambda_asa = lambda_asa
        self.tga_layers = tga_layers
        self.tau = tau

        # 按模态划分LoRA参数（假设参数名含"text"或"image"标识）
        self.lora_param_names = [n for n, p in model.named_parameters() if p.requires_grad and "lora" in n]
        self.lora_params_by_modal = {
            "text": [n for n in self.lora_param_names if "text" in n or "q_proj" in n or "k_proj" in n],
            "image": [n for n in self.lora_param_names if "image" in n or "vision" in n]
        }

        # 为每个模态维护Fisher估计
        self.fisher_estimates = {
            modal: {n: torch.zeros_like(dict(model.named_parameters())[n].detach(),
                                        device=dict(model.named_parameters())[n].device,
                                        dtype=torch.float32)
                    for n in params}
            for modal, params in self.lora_params_by_modal.items()
        }
        self._fisher_count = 0

        # 子网采样超参
        self.subnet_frac = float(subnet_frac)
        # MIR 超参
        self.mir_bins = int(mir_bins)
        self.mir_alpha = float(mir_alpha)

        # AMSS+ 超参
        self.amss_a = float(amss_a)
        self.min_prob = float(min_prob)

        # 参数映射
        self._param_map = {n: p for n, p in model.named_parameters()}

    def compute_token_saliency(self, attn_mean, important_idx):
        L_query = attn_mean.size(0)
        s_i = torch.zeros(L_query, device=attn_mean.device, dtype=attn_mean.dtype)
        if important_idx is not None and important_idx.numel() > 0:
            # proxy_score: 对每个 query token, 取其对重要区域的平均注意力（沿最后维度）
            proxy_score = attn_mean[important_idx].mean(dim=-1)  # [len(imp_idx)]
            norm = proxy_score.max().clamp(min=1e-6)
            s_i[important_idx] = proxy_score / norm

        return s_i

    def estimate_fisher(self):
        # 按模态累积Fisher信息
        for modal in ["text", "image"]:
            for n in self.lora_params_by_modal[modal]:
                p = self._param_map[n]
                if p.grad is not None:
                    self.fisher_estimates[modal][n] += (p.grad.detach().float() ** 2)
        self._fisher_count += 1

    def _sample_subnetworks(self, mir_dict):
        """
        为每个模态独立采样子网络
        mir_dict: {模态名: MIR值}，如 {"text": mir_mean_q, "img": mir_mean_i}
        返回：{模态名: (active_names, p_map)}
        """
        modal_active = {}
        K = len(mir_dict)  # 模态总数
        tau = self.tau

        # 计算所有模态的exp(u/τ)，用于分母求和
        exp_terms = {modal: torch.exp(torch.tensor(u / tau)) for modal, u in mir_dict.items()}
        sum_exp = sum(exp_terms.values())  # 分母：Σexp(u^(n)/τ)

        # 逐模态计算ρ和采样参数
        for modal, u in mir_dict.items():
            # 计算当前模态的更新比例ρ
            exp_k = exp_terms[modal]
            rho_k = 1.0 - (exp_k / sum_exp).item()
            rho_k = max(0.1, min(0.9, rho_k))  # 限制范围

            # 当前模态的参数列表
            modal_params = self.lora_params_by_modal[modal]
            total = len(modal_params)
            if total == 0:
                modal_active[modal] = (set(), {})
                continue

            # 计算需更新的参数数量
            k = max(1, int(rho_k * total))

            # Fisher重要性采样（仅针对当前模态的参数）
            importances = []
            for n in modal_params:
                f = self.fisher_estimates[modal].get(n)
                imp = float(f.mean().item()) if (f is not None and f.numel() > 0) else 1.0
                importances.append(max(imp, 0.0))
            imp_tensor = torch.tensor(importances, dtype=torch.float32)

            if imp_tensor.sum().item() <= 0:
                probs = torch.ones_like(imp_tensor) / float(total)
            else:
                probs = imp_tensor / (imp_tensor.sum() + 1e-12)
            probs = torch.clamp(probs, min=self.min_prob)
            probs = probs / probs.sum()

            # 无放回采样
            try:
                selected_idx = torch.multinomial(probs, num_samples=k, replacement=False)
            except:
                selected_idx = torch.multinomial(probs, num_samples=k, replacement=True)
                selected_idx = torch.unique(selected_idx)[:k]

            selected_names = [modal_params[i] for i in selected_idx.tolist()]
            selected_set = set(selected_names)
            p_map = {modal_params[i]: float(probs[i].item()) for i in range(total)}

            modal_active[modal] = (selected_set, p_map)

        return modal_active

    def forward_backward(self, batch):
        device = next(self.model.parameters()).device
        input_ids = batch["input_ids"].to(device)
        images = batch["image"].to(device, dtype=torch.float16)
        labels = input_ids.clone()
        labels[input_ids == IMAGE_TOKEN_INDEX] = -100
        pad_id = self.tokenizer.pad_token_id or 0
        labels[input_ids == pad_id] = -100
        attn_targets_list = batch["question_tokens"]

        # forward
        outputs = self.model(input_ids=input_ids, images=images, labels=labels,
                             output_attentions=True, return_dict=True)
        lm_loss = outputs.loss

        # --- 计算注意力均值 ---
        attn_stack = torch.stack(outputs.attentions, dim=0)  # [layers, B, heads, L, L]
        attn_head_mean = attn_stack.mean(dim=2)  # [layers, B, L, L]
        if self.tga_layers.startswith("last_"):
            k = int(self.tga_layers.split("_")[1])
            attn_mean = attn_head_mean[-k:].mean(dim=0)  # [B, L, L]
        else:
            attn_mean = attn_head_mean.mean(dim=0)

        # --- 计算每个样本的显著性和MIR ---
        B = input_ids.size(0)
        s_scalar_list = []
        mir_list_q = []
        mir_list_i = []
        for b in range(B):
            len_q = len(attn_targets_list[b])
            ids_row = input_ids[b]
            assistant_token_ids = [319, 1799, 9047, 13566, 29901]

            try:
                image_pos, img_len, q_start, q_end = find_spans(ids_row, IMAGE_TOKEN_INDEX, assistant_token_ids)
            except Exception:
                q_start, q_end = 0, ids_row.size(0)

            # 提取注意力矩阵
            qa_to_image = attn_mean[b, q_end - len_q:q_end, image_pos:image_pos + img_len]  # [q_len, img_len]
            image_to_qa = attn_mean[b, image_pos:image_pos + img_len, q_end - len_q:q_end]  # [img_len, q_len]

            #图像和文本token数差异过大，这是一个点。。。。。。。。

            qa_to_image = smooth_attention(qa_to_image)
            qa_to_image = stable_row_norm(qa_to_image)  # 结合上一步的稳定归一化


            image_to_qa = smooth_attention(image_to_qa)
            image_to_qa = stable_row_norm(image_to_qa)

            # 计算token显著性
            imp_idx = batch["important_q_idx"][b]
            s_i = self.compute_token_saliency(qa_to_image,
                                              imp_idx.to(qa_to_image.device) if imp_idx is not None else None)
            s_scalar = s_i.mean() if s_i.numel() > 0 else torch.tensor(0.0, device=qa_to_image.device,
                                                                       dtype=qa_to_image.dtype)
            s_scalar_list.append(s_scalar)

            # 计算互信息率
            mi_val_q, mi_val_i = attention_mutual_information(qa_to_image, image_to_qa)
            mir_list_q.append(mi_val_q.to(qa_to_image.device))
            mir_list_i.append(mi_val_i.to(qa_to_image.device))

        # 计算批次平均
        s_mean = torch.stack(s_scalar_list).float().mean() if len(s_scalar_list) > 0 else torch.tensor(0.0,
                                                                                                       device=device)
        mir_mean_q = torch.stack(mir_list_q).float().mean() if len(mir_list_q) > 0 else torch.tensor(0.0, device=device)
        mir_mean_i = torch.stack(mir_list_i).float().mean() if len(mir_list_i) > 0 else torch.tensor(0.0, device=device)

        # 构建MIR字典
        mir_dict = {
            "text": mir_mean_q.item(),
            "image": mir_mean_i.item()
        }

        # --- 反向传播和梯度处理 ---
        self.model.zero_grad()
        lm_loss.backward(retain_graph=True)
        self.estimate_fisher()  # 累积Fisher信息

        # Fisher归一化（平均）
        fisher_count = max(1, self._fisher_count)
        fisher_norm = {}
        for modal in ["text", "image"]:
            fisher_norm[modal] = {
                n: f / float(fisher_count)
                for n, f in self.fisher_estimates[modal].items()
            }

        # 为每个模态采样子网络
        modal_active = self._sample_subnetworks(mir_dict)

        # 计算每个模态的MIR缩放因子
        mir_scales = {}
        for modal in ["text", "image"]:
            if modal == "text":
                u = mir_mean_q
            else:
                u = mir_mean_i
            mir_norm = (u / (u + 1.0)).clamp(0.0, 1.0)
            mir_scales[modal] = 1.0 + self.mir_alpha * mir_norm

        # 梯度缩放和掩码
        with torch.no_grad():
            for modal in ["text", "image"]:
                active_names, p_map = modal_active[modal]
                for n in self.lora_params_by_modal[modal]:
                    p = self._param_map[n]
                    if p.grad is None:
                        continue

                    # 未被采样的参数梯度清零
                    if n not in active_names:
                        p.grad.zero_()
                        continue

                    # 获取采样概率p_j
                    p_j = p_map.get(n, self.min_prob)
                    p_j = max(p_j, self.min_prob)

                    # Fisher因子
                    fisher = fisher_norm[modal].get(n, torch.ones_like(p.data))
                    fisher = fisher.to(p.grad.device).to(p.grad.dtype)
                    fisher_scalar = float(fisher.mean().item()) if fisher.numel() > 0 else 1.0
                    fisher_scalar = max(fisher_scalar, 1e-12)
                    fisher_factor = (s_mean.to(p.grad.dtype) / (fisher_scalar + 1e-6))

                    # AMSS+无偏项
                    amss_unbias = 1.0 / (p_j + self.amss_a)

                    # 模态专属MIR缩放
                    mir_scale = mir_scales[modal].to(p.grad.dtype)

                    # 最终缩放梯度
                    scale = fisher_factor * float(amss_unbias) * float(mir_scale)
                    MAX_SCALE = 1e6
                    scale = min(scale, MAX_SCALE)
                    p.grad.mul_(scale)

        # 梯度裁剪
        trainable_params = []
        for modal in ["text", "image"]:
            trainable_params.extend([
                self._param_map[n] for n in self.lora_params_by_modal[modal]
                if self._param_map[n].requires_grad
            ])
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        return {
            "lm_loss": lm_loss.item(),
            "s_mean": s_mean.item() if isinstance(s_mean, torch.Tensor) else float(s_mean),
            "mir_mean_q": mir_mean_q.item() if isinstance(mir_mean_q, torch.Tensor) else float(mir_mean_q),
            "mir_mean_i": mir_mean_i.item() if isinstance(mir_mean_i, torch.Tensor) else float(mir_mean_i),
            "active_text_params": len(modal_active["text"][0]),
            "active_image_params": len(modal_active["image"][0]),
            "total_text_params": len(self.lora_params_by_modal["text"]),
            "total_image_params": len(self.lora_params_by_modal["image"])
        }


# =========================
# 主训练循环
# =========================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/data2/gaodz/llava-v1.6-vicuna-7b"
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_v1.6", device=device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj"],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False
    )
    model = get_peft_model(model, lora_config)

    vision_tower = model.get_vision_tower()

    # 冻结视觉塔原始参数
    for p in vision_tower.parameters():
        p.requires_grad = False

    # 冻结视觉塔的 LoRA 参数
    for n, p in vision_tower.named_parameters():
        if "lora" in n:
            p.requires_grad = False
    model.train().to(device)

    dataset = MultimodalTGADataset(
        json_file="/data2/gaodz/Re-Align/combined_data_token_filter_mark_t2_asa.json",
        image_folder="/data2/gaodz/train2014",
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=512
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # 使用新版训练器
    trainer = AMSSRebalanceTrainer(model, tokenizer, lambda_asa=0.1, tga_layers="last_4",
                                   subnet_frac=0.6, mir_bins=8, mir_alpha=4.0)
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n],
        lr=5e-6, weight_decay=0.01
    )
    num_epochs = 2
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            try:
                stats = trainer.forward_backward(batch)
            except Exception as e:
                print(f"[WARN] forward 出错 batch {i}: {str(e)}")
                continue

            optimizer.step()
            model.zero_grad()
            pbar.set_postfix({
                "LM": f"{stats['lm_loss']:.4f}",
                "s_mean": f"{stats['s_mean']:.4f}",
                "mir_q": f"{stats['mir_mean_q']:.4f}",
                "mir_i": f"{stats['mir_mean_i']:.4f}",
                "active_text": f"{stats['active_text_params']}/{stats['total_text_params']}",
                "active_image": f"{stats['active_image_params']}/{stats['total_image_params']}"
            })

    # 保存模型
    save_dir = "/data2/gaodz/tga_subnet_amss_qk_1"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print(f"[OK] LoRA 已保存到 {save_dir}")
