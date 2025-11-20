


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# import json
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, ConcatDataset
# from PIL import Image
# from tqdm import tqdm
# import warnings
# from transformers import CLIPModel, CLIPProcessor
#
# warnings.filterwarnings("ignore")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 配置（统一768维，匹配CLIP原生文本特征）
# SAVE_DIR = "/data2/gaodz/attr_attention_result_2"
# os.makedirs(SAVE_DIR, exist_ok=True)
# NUM_SUBSPACES = 6  # color/shape/material/object/scene/emotion
# FEAT_DIM = 768  # 核心：所有子空间特征维度（CLIP文本原生维度）
# BATCH_SIZE = 32
# #
# # STABILIZATION_EPOCHS = 1
# REQUIRED_ATTRS = ["color", "shape", "material", "object", "scene", "emotion"]
#
#
# ### 1. 数据加载（无修改）
# class AttrDataset(Dataset):
#     def __init__(self, json_path, image_root, clip_processor):
#         self.samples = []
#         with open(json_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 sample = json.loads(line.strip())
#                 if not all(k in sample for k in ["image", "long_text"]):
#                     continue
#                 # 补全空白detail
#                 if "detail" not in sample:
#                     sample["detail"] = {attr: [] for attr in REQUIRED_ATTRS}
#                 else:
#                     for attr in REQUIRED_ATTRS:
#                         if attr not in sample["detail"]:
#                             sample["detail"][attr] = []
#                 self.samples.append(sample)
#         self.image_root = image_root
#         self.clip_processor = clip_processor
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         img_path = os.path.join(self.image_root, sample["image"])
#         image = Image.open(img_path).convert("RGB")
#         return {"image": image, "text": sample["long_text"], "detail": sample["detail"]}
#
#
# def attr_collate_fn(batch):
#     images = [item["image"] for item in batch]
#     texts = [item["text"] for item in batch]
#     details = [item["detail"] for item in batch]
#     return {"image": images, "text": texts, "detail": details}
#
#
# ### 2. 语义探针与属性特征生成（跳过text_projection）
# def load_semantic_probes(probe_path, clip_model, clip_processor, device):
#     """加载探针（若为1024维，后续投影到768维）"""
#     with open(probe_path, "r", encoding="utf-8") as f:
#         probes = json.load(f)
#     probe_order = ["color", "shape", "material", "object", "scene", "emotion"]
#     probe_matrix = []
#     for key in probe_order:
#         if key not in probes:
#             raise ValueError(f"探针文件缺少{key}字段")
#         texts = [x for x in probes[key][:50]]
#         inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(device)
#         with torch.no_grad():
#             # 探针用全局特征（需text_projection，确保与视觉特征对齐）
#             feat = clip_model.get_text_features(** inputs)  # [50, 768]（CLIP-large输出）
#         probe_matrix.append(F.normalize(feat, dim=-1))
#     return torch.stack(probe_matrix, dim=0).to(device)  # [6, 50, 768]
#
#
# def generate_probe_subspaces(clip_text_feat, probe_matrix):
#     """生成768维z（无需额外投影，探针已为768维）"""
#     B = clip_text_feat.shape[0]
#     K, P, D = probe_matrix.shape  # K=6, P=50, D=768
#     z_list = []
#     for k in range(K):
#         sim = F.cosine_similarity(clip_text_feat.unsqueeze(1), probe_matrix[k].unsqueeze(0), dim=-1)
#         weights = F.softmax(sim, dim=1)
#         z_k = torch.bmm(weights.unsqueeze(1), probe_matrix[k].unsqueeze(0).repeat(B, 1, 1)).squeeze(1)
#         z_list.append(F.normalize(z_k, dim=-1))
#     return torch.stack(z_list, dim=1)  # [B, 6, 768]
#
#
# class AttrFeatureGenerator(nn.Module):
#     def __init__(self, clip_model, num_subspaces=6, feat_dim=768):
#         super().__init__()
#         self.clip_model = clip_model
#         self.num_subspaces = num_subspaces
#         self.feat_dim = feat_dim
#
#         # 1. 子空间query：移除多余的1维，简化维度（原[6,1,768]→[6,768]）
#         self.subspace_queries = nn.Parameter(torch.randn(num_subspaces, 1,feat_dim))  # [K, D]
#         for i in range(num_subspaces):
#             nn.init.orthogonal_(self.subspace_queries[i])  # 保证初始子空间正交
#         self.subspace_queries.data = F.normalize(self.subspace_queries.data, dim=-1)  # 初始归一化
#
#         # 2. 新增：子空间独立的可学习注意力模块（替换原有点积注意力）
#         self.attr_attn = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(feat_dim * 2, feat_dim // 2),  # 输入：属性词特征 + 子空间query（D*2）
#                 nn.GELU(),  # 非线性激活，增强表达
#                 nn.Linear(feat_dim // 2, 1)  # 输出每个属性词的注意力得分
#             ) for _ in range(num_subspaces)  # 每个子空间1个独立注意力
#         ])
#
#         # 3. 新增：自适应gate的MLP（替换原静态参数gate）
#         self.gate_mlp = nn.Sequential(
#             nn.Linear(feat_dim, feat_dim // 2),  # 输入：属性词加权特征（D）
#             nn.GELU(),
#             nn.Linear(feat_dim // 2, 1),  # 输出每个样本的融合比例（0~1）
#             nn.Sigmoid()
#         )
#
#         self.subspace_names = ["color", "shape", "material", "object", "scene", "emotion"]
#
#     def get_attr_word_feats(self, attr_words, clip_processor, device):
#         """属性词特征：移除torch.no_grad()，恢复梯度传递"""
#         if not attr_words:
#             return torch.zeros(0, self.feat_dim, device=device)  # [0, D]
#         inputs = clip_processor(
#             text=attr_words,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=32
#         ).to(device)
#         # 关键：移除torch.no_grad()，让梯度能回传至subspace_queries和gate_mlp
#         text_encoder_output = self.clip_model.text_model(** inputs)
#         cls_768 = text_encoder_output.last_hidden_state[:, 0, :]  # [N, D]（N=属性词总数）
#
#
#
#         return F.normalize(cls_768, dim=-1)  # 属性词特征归一化（保证尺度一致）
#
#     def forward(self, samples_detail, clip_processor, device):
#         B = len(samples_detail)
#         all_subspace_feats = []  # 存储每个子空间的特征
#         sample_valid_masks = []  # 存储每个样本的有效掩码
#
#         for subspace_idx in range(self.num_subspaces):
#             name = self.subspace_names[subspace_idx]
#             query = self.subspace_queries[subspace_idx].squeeze(0)  # [D]（当前子空间的query）
#             valid_mask = torch.zeros(B, dtype=torch.float32, device=device)  # [B]
#
#             # 1. 收集当前子空间的所有有效属性词（按样本分组）
#             batch_attrs = []  # 所有样本的有效属性词（平级存储）
#             attr_counts = []  # 每个样本的有效属性词数量（用于后续拆分）
#             for b in range(B):
#                 attrs = samples_detail[b].get(name, [])
#                 # 过滤空值/无效词（保留至少1个字符的词）
#                 valid_attrs = [str(a).strip().lower() for a in attrs if str(a).strip().lower() not in ["", "none", "unknown"]]
#                 attr_counts.append(len(valid_attrs))
#                 # 只要有1个有效属性词，就标记为有效样本（降低阈值，增加监督）
#                 if len(valid_attrs) >= 1:
#                     valid_mask[b] = 1.0
#                     batch_attrs.extend(valid_attrs)
#
#             # 2. 提取属性词特征（若无可跳过，用query兜底）
#             batch_raw_feat = torch.zeros(B, self.feat_dim, device=device)  # [B, D]（每个样本的属性词加权特征）
#             if len(batch_attrs) > 0:
#                 batch_attr_feats = self.get_attr_word_feats(batch_attrs, clip_processor, device)  # [N, D]
#                 attr_idx = 0  # 用于拆分不同样本的属性词特征
#
#                 # 3. 改进的注意力计算：子空间独立MLP注意力（结合query）
#                 for b in range(B):
#                     count = attr_counts[b]
#                     if count == 0:
#                         continue  # 无属性词，保持batch_raw_feat为0
#                     # 拆分当前样本的属性词特征
#                     sample_attr_feats = batch_attr_feats[attr_idx:attr_idx + count]  # [count, D]
#                     attr_idx += count
#
#                     # 注意力输入：拼接属性词特征和query（每个属性词都结合子空间语义）
#                     attn_input = torch.cat([
#                         sample_attr_feats,  # [count, D]
#                         query.unsqueeze(0).repeat(count, 1)  # [count, D]（复制query到每个属性词）
#                     ], dim=-1)  # [count, 2D]
#
#                     # 计算注意力得分（子空间独立MLP）
#                     attn_scores = self.attr_attn[subspace_idx](attn_input).squeeze(-1)  # [count]
#                     attn_weights = F.softmax(attn_scores, dim=-1)  # [count]（权重和为1）
#
#                     # 加权得到当前样本的属性词特征
#                     batch_raw_feat[b] = torch.matmul(attn_weights.unsqueeze(0), sample_attr_feats).squeeze(0)  # [D]
#
#             # 4. 改进的自适应融合：基于属性词特征动态调整gate
#             query_expand = query.unsqueeze(0).repeat(B, 1)  # [B, D]（query复制到每个样本）
#             # 计算每个样本的自适应gate（输入：当前样本的属性词特征）
#             gate = self.gate_mlp(batch_raw_feat)  # [B, 1]（每个样本一个gate值，0~1）
#             # 融合特征：不再立即归一化，留到后续对齐阶段（解决Attr-Orth不变问题）
#             fused_feat = gate * query_expand + (1 - gate) * batch_raw_feat  # [B, D]
#
#             # 5. 最终归一化（仅在特征融合完成后，保证后续对齐时尺度一致）
#             fused_feat = F.normalize(fused_feat, dim=-1)  # [B, D]
#
#             # 收集当前子空间的结果
#             all_subspace_feats.append(fused_feat)
#             sample_valid_masks.append(valid_mask)
#
#         # 拼接所有子空间的特征和掩码
#         a = torch.stack(all_subspace_feats, dim=1)  # [B, K, D]（K=6）
#         sample_valid_mask = torch.stack(sample_valid_masks, dim=1)  # [B, K]
#
#         return a, sample_valid_mask
#
#
# ### 3. 注意力融合模块（768维）
# class AttrAttentionFusion(nn.Module):
#     def __init__(self, dim=768):
#         super().__init__()
#         self.attn = nn.Sequential(
#             nn.Linear(dim * 2, dim // 2),  # 1536→384
#             nn.GELU(),
#             nn.Linear(dim // 2, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x1, x2):
#         concat = torch.cat([x1, x2], dim=-1)
#         attn_weights = self.attn(concat)
#         fused = x1 * (1 - attn_weights) + x2 * attn_weights
#         return F.normalize(fused, dim=-1)
#
#
# ### 4. 主模型（适配768维，无多余投影）
# class DynamicRoutingNetwork(nn.Module):
#     def __init__(self, num_subspaces=6, feat_dim=768):
#         super().__init__()
#         self.num_subspaces = num_subspaces
#         self.feat_dim = feat_dim
#
#         # 原有模块保留
#         self.attn_fusion = AttrAttentionFusion(dim=feat_dim)
#         self.route_mlp = nn.Sequential(
#             nn.Linear(feat_dim * 2, feat_dim // 2),
#             nn.GELU(),
#             nn.Linear(feat_dim // 2, 1)
#         )
#         self.route_temp = nn.Parameter(torch.tensor(1.0))
#         self.bridge = nn.Sequential(
#             nn.Linear(feat_dim, feat_dim),
#             nn.GELU(),
#             nn.Linear(feat_dim, feat_dim)
#         )
#
#         # === 可训练子空间注意力（修改：适配query融合） ===
#         self.vis_subspace_attn = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(feat_dim * 2, feat_dim // 2),  # 输入：patch特征 + 子空间query（768*2）
#                 nn.GELU(),
#                 nn.Linear(feat_dim // 2, 1)
#             ) for _ in range(num_subspaces)
#         ])
#
#         self.txt_subspace_attn = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(feat_dim * 2, feat_dim // 2),  # 输入：token特征 + 子空间query
#                 nn.GELU(),
#                 nn.Linear(feat_dim // 2, 1)
#             ) for _ in range(num_subspaces)
#         ])
#
#         self.vis_temp = nn.Parameter(torch.tensor(1.0))
#         self.txt_temp = nn.Parameter(torch.tensor(1.0))
#
#     def forward(self, clip_text_feat, z, a, patch_feats=None, clip_text_tokens=None):
#         B = clip_text_feat.shape[0]
#         K = self.num_subspaces
#         device = clip_text_feat.device
#
#         # === 文本子空间初始特征（保留） ===
#         base_subspaces = torch.zeros(B, K, self.feat_dim, device=device)
#         if clip_text_tokens is not None:
#             T = clip_text_tokens.shape[1]
#             queries = z.unsqueeze(2)  # [B,K,1,768]
#             keys = clip_text_tokens.unsqueeze(1)
#             attn = torch.matmul(queries, keys.transpose(-2, -1)) / (self.feat_dim ** 0.5)
#             w = F.softmax(attn, dim=-1)
#             base_subspaces = torch.matmul(w, clip_text_tokens.unsqueeze(1)).squeeze(2)
#             base_subspaces = F.normalize(base_subspaces, dim=-1)
#
#         # === 融合 & 路由（保留） ===
#         za_fused = self.attn_fusion(z, a)
#         attr_correction = self.attn_fusion(base_subspaces, za_fused)
#         final_subspaces = F.normalize(base_subspaces + attr_correction, dim=-1)  # [B,K,768]：子空间语义query
#
#         route_input = torch.cat([final_subspaces, a], dim=-1)
#         scores = self.route_mlp(route_input).squeeze(-1)
#         weights = F.softmax(scores / self.route_temp, dim=-1)
#
#         weighted_sub = final_subspaces * weights.unsqueeze(-1)
#         txt_subspace_agg = weighted_sub.sum(dim=1)
#         txt_global_feat = F.normalize(clip_text_feat + self.bridge(txt_subspace_agg), dim=-1)
#
#         # === CLIM 子空间（核心修改：融入子空间query） ===
#         clim_feats = None
#         if patch_feats is not None and clip_text_tokens is not None:
#             vis_sub, txt_sub = [], []
#             for k in range(K):
#                 # 子空间k的语义query（final_subspaces包含子空间的最终语义）
#                 subspace_query = final_subspaces[:, k, :]  # [B,768]
#
#                 # 1. 视觉子空间注意力（结合query）
#                 B_patch, P, D = patch_feats.shape
#                 # 将子空间query与每个patch特征拼接（每个patch都感知子空间语义）
#                 vis_input = torch.cat([
#                     patch_feats,  # [B,P,768]
#                     subspace_query.unsqueeze(1).repeat(1, P, 1)  # [B,P,768]（复制到每个patch）
#                 ], dim=-1)  # [B,P,1536]
#                 # 计算注意力权重（依赖patch特征+子空间query）
#                 vis_w = self.vis_subspace_attn[k](vis_input) / self.vis_temp  # [B,P,1]
#                 vis_w = F.softmax(vis_w, dim=1)
#                 vis_sub_k = (patch_feats * vis_w).sum(dim=1)  # [B,768]（聚合后特征）
#
#                 # 2. 文本子空间注意力（同理结合query）
#                 B_token, T, D = clip_text_tokens.shape
#                 txt_input = torch.cat([
#                     clip_text_tokens,  # [B,T,768]
#                     subspace_query.unsqueeze(1).repeat(1, T, 1)  # [B,T,768]（复制到每个token）
#                 ], dim=-1)  # [B,T,1536]
#                 txt_w = self.txt_subspace_attn[k](txt_input) / self.txt_temp  # [B,T,1]
#                 txt_w = F.softmax(txt_w, dim=1)
#                 txt_sub_k = (clip_text_tokens * txt_w).sum(dim=1)  # [B,768]
#
#                 vis_sub.append(vis_sub_k)
#                 txt_sub.append(txt_sub_k)
#
#             vis_sub = torch.stack(vis_sub, dim=1)  # [B,K,768]
#             txt_sub = torch.stack(txt_sub, dim=1)  # [B,K,768]
#             clim_feats = (vis_sub, txt_sub)
#
#         return final_subspaces, weights, txt_global_feat, clim_feats
#
#
# ### 5. 损失函数（适配768维）
# def compute_batch_hard_global_loss(txt_global_feat, img_global_feat, temperature=0.07):
#     """全局损失：图文特征均为768维"""
#     sim_matrix = torch.matmul(txt_global_feat, img_global_feat.T) / temperature  # [B,B]
#     pos_sim = sim_matrix.diag()
#     sim_matrix = sim_matrix - torch.eye(sim_matrix.size(0), device=device) * 1e9
#     neg_sim = sim_matrix.max(dim=1)[0]
#     return -F.logsigmoid(pos_sim - neg_sim).mean()
#
#
# class CLIMLoss(nn.Module):
#     def __init__(self, num_subspaces=6, temperature=0.1):
#         super().__init__()
#         self.K = num_subspaces
#         self.temp = temperature
#
#     def forward(self, vis_sub, txt_sub):
#         B, K, D = vis_sub.shape
#         total_loss = 0.0
#
#         for k in range(K):
#             vis = F.normalize(vis_sub[:, k], dim=-1)
#             txt = F.normalize(txt_sub[:, k], dim=-1)
#             sim = vis @ txt.T / self.temp  # [B,B]
#             labels = torch.arange(B, device=vis.device)
#             loss_i = F.cross_entropy(sim, labels)
#             loss_t = F.cross_entropy(sim.T, labels)
#             total_loss += (loss_i + loss_t) / 2
#
#         return total_loss / K
#
#
# def soft_attr_probe_loss(a, z, sample_valid_mask, temperature=0.2):
#     """a/z均为768维"""
#     B, K, D = a.shape  # D=768
#     loss_per_subspace = []
#     for k in range(K):
#         a_k = a[:, k, :]
#         z_k = z[:, k, :]
#         valid_mask_k = sample_valid_mask[:, k]
#         valid_indices = torch.where(valid_mask_k > 0)[0]
#         if len(valid_indices) < 2:
#             loss_per_subspace.append(torch.tensor(0.0, device=device))
#             continue
#
#         a_k_valid = a_k[valid_indices]
#         z_k_valid = z_k[valid_indices]
#         pos_sim = F.cosine_similarity(a_k_valid, z_k_valid, dim=-1)
#
#         z_k_neg = z_k_valid.unsqueeze(1).repeat(1, len(valid_indices), 1)
#         z_k_neg = z_k_neg[~torch.eye(len(valid_indices), dtype=torch.bool, device=device)]
#         z_k_neg = z_k_neg.reshape(len(valid_indices), -1, D)
#
#         neg_sim = F.cosine_similarity(a_k_valid.unsqueeze(1), z_k_neg, dim=-1)
#         logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1) / temperature
#         labels = torch.zeros(len(valid_indices), dtype=torch.long, device=device)
#         loss_k = F.cross_entropy(logits, labels, reduction='none')
#         loss_k = (loss_k * valid_mask_k[valid_indices]).mean()
#         loss_per_subspace.append(loss_k)
#     return torch.stack(loss_per_subspace).mean() * 0.5
#
#
# def subspace_ortho_loss(subspace_feats):
#     """子空间特征为768维"""
#     B, K, D = subspace_feats.shape
#     eye = torch.eye(K, device=device).unsqueeze(0).repeat(B, 1, 1)
#     gram_per_sample = torch.matmul(subspace_feats, subspace_feats.transpose(1, 2))
#     return ((gram_per_sample - eye) **2).mean()
#
#
# ### 6. 训练主函数（仅全局特征保留投影）
# def main():
#     # 加载CLIP-large（确认维度）
#     clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
#     clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
#     print("视觉投影层（1024→768）：", clip_model.visual_projection.weight.shape)
#     print("文本投影层（768→768）：", clip_model.text_projection.weight.shape)
#
#     # 核心：冻结CLIP所有参数（不更新），但保留梯度传递能力
#     for param in clip_model.parameters():
#         param.requires_grad = False  # 修正：CLIP参数不更新（原代码写反了）
#     clip_model.train()  # 设为train模式，允许梯度追踪
#
#     # 加载探针（已为768维）
#     probe_path = "/data2/gaodz/Re-Align/coco_semantic_probes.json"
#     probe_matrix = load_semantic_probes(probe_path, clip_model, clip_processor, device)
#     print(f"语义探针形状：{probe_matrix.shape}（768维）")
#
#     # 初始化模型（不变）
#     attr_generator = AttrFeatureGenerator(
#         clip_model=clip_model,
#         num_subspaces=NUM_SUBSPACES,
#         feat_dim=FEAT_DIM
#     ).to(device)
#     model = DynamicRoutingNetwork(
#         num_subspaces=NUM_SUBSPACES,
#         feat_dim=FEAT_DIM
#     ).to(device)
#     clim_loss_fn = CLIMLoss(num_subspaces=NUM_SUBSPACES).to(device)
#
#     # 优化器（不变，仅优化model和attr_generator）
#     optimizer = optim.AdamW(
#         list(model.parameters()) + list(attr_generator.parameters()),
#         lr=1e-4,
#         weight_decay=1e-4
#     )
#
#     # 加载数据（核心修改：拆分有效/空属性数据集）
#     data_config = {"datasets": [
#         {"data_json_path": "/data2/gaodz/sam_data/coco_with_detail_partial.json",
#          "image_root": "/data2/gaodz/train2014"},  # 有效属性：coco
#         {"data_json_path": "/data2/gaodz/sam_data/wikiart_with_detail_partial.json",
#          "image_root": "/data2/gaodz/WikiArt/OpenDataLab___WikiArt/raw/train_image/wikiart"},  # 有效属性：wikiart
#         {"data_json_path": "/data2/gaodz/sharegpt4v/sharegpt4v_coco.json",
#          "image_root": "/data2/gaodz/coco2017/PAI/COCO2017"},  # 空属性：10w条
#     ]}
#     all_datasets = []
#     for info in data_config["datasets"]:
#         ds = AttrDataset(info["data_json_path"], info["image_root"], clip_processor)
#         all_datasets.append(ds)
#         print(f"加载数据集 {info['data_json_path']}，样本数：{len(ds)}")
#
#     # 拆分阶段1（有效样本）和阶段2（混合样本）数据集
#     valid_datasets = all_datasets[:2]  # 前2个：4w有效属性样本
#     empty_dataset = all_datasets[2]    # 第3个：10w空属性样本
#     # 阶段2：有效样本与空样本按1:1混合（避免空样本过多）
#     valid_total = len(ConcatDataset(valid_datasets))
#     empty_sample_num = min(valid_total, len(empty_dataset))  # 空样本数≤有效样本数
#     from torch.utils.data import Subset
#     import random
#     empty_indices = random.sample(range(len(empty_dataset)), empty_sample_num)
#     balanced_empty_dataset = Subset(empty_dataset, empty_indices)
#     mixed_datasets = valid_datasets + [balanced_empty_dataset]  # 阶段2混合数据集
#
#     # 分阶段训练参数配置
#     EPOCHS = 10  # 总epoch：阶段1占2个，阶段2占4个
#     STAGE1_EPOCHS = 4  # 阶段1：固定查询向量
#     # 阶段1损失权重（聚焦a与z对齐）
#     stage1_weights = {
#         "lambda_global": 0.3,    # 弱全局损失
#         "lambda_clim": 0.0,      # 关闭CLIM
#         "lambda_orth": 0.0,      # 关闭正交损失
#         "lambda_attr_orth": 0.0, # 关闭属性正交损失
#         "probe_weight": 1.0      # 强属性对齐损失
#     }
#     # 阶段2损失权重（优化对齐，维持语义）
#     stage2_weights = {
#         "lambda_global": 1.0,    # 强全局损失
#         "lambda_clim": 0.5,      # 启用CLIM
#         "lambda_orth": 0.2,      # 启用正交损失
#         "lambda_attr_orth": 0.2, # 启用属性正交损失
#         "probe_weight": 0.5      # 维持属性对齐
#     }
#     STABILIZATION_EPOCHS = 2  # 原有第二阶段逻辑（嵌套在阶段2内）
#
#     # 训练循环（分阶段执行）
#     for epoch in range(EPOCHS):
#         # 1. 判断当前阶段并初始化数据集
#         is_stage1 = epoch < STAGE1_EPOCHS
#         is_stage2 = not is_stage1
#         current_datasets = valid_datasets if is_stage1 else mixed_datasets
#         dataloader = DataLoader(
#             ConcatDataset(current_datasets),
#             batch_size=BATCH_SIZE,
#             shuffle=True,
#             num_workers=2,
#             collate_fn=attr_collate_fn,
#             pin_memory=True
#         )
#         print(f"\n=== Epoch {epoch+1}/{EPOCHS} | {'阶段1（固定查询向量）' if is_stage1 else '阶段2（优化对齐）'} ===")
#         print(f"当前数据集：{'4w有效样本' if is_stage1 else f'4w有效+{empty_sample_num}空样本'} | Batch数：{len(dataloader)}")
#
#         # 2. 按阶段冻结参数
#         if is_stage1:
#             # 阶段1：只训练attr_generator（聚焦查询向量和属性注意力）
#             for param in model.parameters():
#                 param.requires_grad = False
#             for param in attr_generator.parameters():
#                 param.requires_grad = True
#         else:
#             # 阶段2：冻结attr_generator核心参数（避免语义退化），训练其他层
#             for name, param in attr_generator.named_parameters():
#                 if "subspace_queries" in name or "attr_attn" in name:  # 核心语义参数
#                     param.requires_grad = False
#                 else:  # 非核心层（如门控、残差）可微调
#                     param.requires_grad = True
#             for param in model.parameters():  # 训练动态路由网络
#                 param.requires_grad = True
#
#         # 3. 训练当前epoch
#         model.train()
#         attr_generator.train()
#         total_loss = 0.0
#         current_weights = stage1_weights if is_stage1 else stage2_weights
#
#         for batch_idx, batch in enumerate(tqdm(dataloader)):
#             optimizer.zero_grad()
#             images = batch["image"]
#             texts = [t.replace("<image>", "") for t in batch["text"]]
#             samples_detail = batch["detail"]
#
#             # 提取CLIP视觉特征（无需梯度）
#             img_inputs = clip_processor(images=images, return_tensors="pt").to(device)
#             with torch.no_grad():
#                 img_global_feat = clip_model.get_image_features(**img_inputs)
#                 img_global_feat = F.normalize(img_global_feat, dim=-1)
#                 vision_outputs = clip_model.vision_model(** img_inputs, output_hidden_states=True)
#                 patch_1024 = vision_outputs.hidden_states[-1][:, 1:, :]
#                 patch_feats = clip_model.visual_projection(patch_1024)
#                 patch_feats = F.normalize(patch_feats, dim=-1)
#
#             # 提取CLIP文本特征（保留梯度用于属性学习）
#             text_inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
#             text_outputs = clip_model.text_model(**text_inputs, output_hidden_states=True)
#             clip_text_feat = clip_model.get_text_features(** text_inputs).detach()  # 全局文本特征不更新CLIP
#             clip_text_feat = F.normalize(clip_text_feat, dim=-1)
#             clip_text_tokens = text_outputs.last_hidden_state[:, 1:, :]
#             clip_text_tokens = F.normalize(clip_text_tokens, dim=-1)
#
#             # 生成子空间特征
#             z = generate_probe_subspaces(clip_text_feat, probe_matrix)
#             a, sample_valid_mask = attr_generator(
#                 samples_detail=samples_detail,
#                 clip_processor=clip_processor,
#                 device=device
#             )
#
#             # 模型前向
#             final_subspaces, route_weights, txt_global_feat, clim_feats = model(
#                 clip_text_feat=clip_text_feat, z=z, a=a, patch_feats=patch_feats, clip_text_tokens=clip_text_tokens
#             )
#
#             # 计算损失（按阶段权重）
#             global_loss = compute_batch_hard_global_loss(txt_global_feat, img_global_feat) * current_weights["lambda_global"]
#             attr_probe_loss = soft_attr_probe_loss(a, z, sample_valid_mask) * current_weights["probe_weight"]
#             attr_orth_loss = subspace_ortho_loss(a) * current_weights["lambda_attr_orth"] if is_stage2 else torch.tensor(0.0).to(device)
#
#             clim_loss = torch.tensor(0.0).to(device)
#             orth_loss = torch.tensor(0.0).to(device)
#             if is_stage2 and (epoch >= STABILIZATION_EPOCHS) and (clim_feats is not None):
#                 clim_loss = clim_loss_fn(*clim_feats) * current_weights["lambda_clim"]
#                 orth_loss = subspace_ortho_loss(final_subspaces) * current_weights["lambda_orth"]
#
#             total_batch_loss = global_loss + clim_loss + orth_loss + attr_probe_loss + attr_orth_loss
#             total_batch_loss.backward()
#             optimizer.step()
#             total_loss += total_batch_loss.item()
#
#             # 日志打印
#             if (batch_idx + 1) % 50 == 0:
#                 avg_loss = total_loss / (batch_idx + 1)
#                 attr_probe_sim = F.cosine_similarity(a, z, dim=-1).mean().item()
#                 print(f"Batch {batch_idx + 1} | Avg Loss: {avg_loss:.4f} | "
#                       f"Global: {global_loss.item():.4f} | CLIM: {clim_loss.item():.4f} | "
#                       f"Probe Loss: {attr_probe_loss.item():.4f} | Attr-Probe Sim: {attr_probe_sim:.4f} | "
#                       f"Orth: {orth_loss.item():.4f} | Attr-Orth: {attr_orth_loss.item():.4f}")
#
#         # 保存模型
#         save_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch + 1}.pth")
#         torch.save({"model": model.state_dict(), "attr_generator": attr_generator.state_dict()}, save_path)
#         print(f"Epoch {epoch + 1} 模型保存至 {save_path}")
#
#
# if __name__ == "__main__":
#     main()




