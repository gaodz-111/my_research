import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Gumbel
from PIL import Image
from tqdm import tqdm
import sys
import json
import copy
# 补充缺失的PEFT依赖
from peft import LoraConfig, get_peft_model, TaskType
from peft.tuners.lora import LoraLayer


# ===== 1. 依赖导入（文档技术核心：ViR/ViCO + LLaVA基础）=====
# LLaVA依赖（需确保llava目录路径正确）
parent_dir = os.path.abspath("./llava")
sys.path.append(parent_dir)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX,IGNORE_INDEX
from llava.conversation import conv_templates


def log_debug_info(info_dict, filepath="debug_nan_log.jsonl"):
    """将调试信息写入文件"""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(info_dict, ensure_ascii=False) + "\n")

# 你提供的数据集与collate_fn
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

        # 生成包含<image>的input_ids，确保正确处理占位符
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        image_path = os.path.join(self.image_folder, "COCO_train2014_" + item["image"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_processor(image, return_tensors="pt")["pixel_values"][0]
        que_token_list = item.get("question_tokens", [])

        return {
            "input_ids": input_ids,
            "image": image_tensor,
            "question_tokens": que_token_list,
            # 新增原始问题用于后续注意力计算
            "original_question": qs
        }


def collate_fn(batch, pad_id=0):
    input_ids_list = [item["input_ids"].squeeze(0) for item in batch]
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.ones_like(x) for x in input_ids_list],
        batch_first=True,
        padding_value=0
    )
    images = torch.stack([item["image"] for item in batch], dim=0)
    attn_targets = [item["question_tokens"] for item in batch]
    original_questions = [item["original_question"] for item in batch]

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "image": images,
        "question_tokens": attn_targets,
        "original_questions": original_questions
    }


# ===== 2. ViR模块定义（修复压缩逻辑，确保输出M个patch）=====
class VisualResolutionRouter(nn.Module):
    """软混合特征输入的动态压缩模块（增强数值稳定性版本）"""

    def __init__(self, input_dim=1024, dtype=torch.float32):  # 修改：默认精度改为float32
        super().__init__()
        self.input_dim = input_dim
        self.compression_rates = {0: 4, 1: 16}  # 0:1/4压缩, 1:1/16压缩

        # 路由器分类器：预测每个patch应使用的压缩率
        self.router_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(input_dim, 2, dtype=dtype)  # 输出2类：0或1
        )

        # 压缩投影层（对应不同压缩率）
        self.proj_1_4 = nn.Linear(input_dim, input_dim, dtype=dtype)  # 1/4压缩投影
        self.proj_1_16 = nn.Linear(input_dim, input_dim, dtype=dtype)  # 1/16压缩投影

        # 初始化权重（关键修复：缩小初始权重防止数值爆炸）
        self._init_weights()

    def _init_weights(self):
        """重新初始化权重，使用更小的初始值"""
        for m in self.router_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.02)  # 修改：稍微增大增益
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in [self.proj_1_4, self.proj_1_16]:
            nn.init.xavier_normal_(m.weight, gain=0.02)  # 修改：稍微增大增益
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, visual_tokens, return_rate_pred=False, temperature=0.5):  # 修改：温度从0.3提高到0.5
        B, L_img, D = visual_tokens.shape
        target_device = visual_tokens.device
        target_dtype = visual_tokens.dtype

        # 1. 输入特征预处理（关键修复：限制输入范围）
        visual_tokens = torch.clamp(visual_tokens, min=-4.0, max=4.0)  # 修改：放宽范围
        print(f"\n[Router] 输入visual_tokens - 均值: {visual_tokens.mean().item():.4f}, "
              f"最大值: {visual_tokens.max().item():.4f}, "
              f"最小值: {visual_tokens.min().item():.4f}, "
              f"是否含nan: {torch.isnan(visual_tokens).any().item()}")

        # 2. 压缩率预测（关键修复：限制logits范围）
        rate_logits = self.router_classifier(visual_tokens)
        rate_logits = torch.clamp(rate_logits, min=-15.0, max=15.0)  # 修改：放宽范围
        print(f"[Router] rate_logits - 均值: {rate_logits.mean().item():.4f}, "
              f"最大值: {rate_logits.max().item():.4f}, "
              f"最小值: {rate_logits.min().item():.4f}, "
              f"是否含nan: {torch.isnan(rate_logits).any().item()}")

        if self.training:
            # 3. Gumbel-Softmax（关键修复：增加数值稳定性）
            try:
                # 安全的Gumbel采样（避免极端值）
                gumbel = torch.distributions.Gumbel(0, 1).sample(rate_logits.shape).to(target_device,
                                                                                       dtype=target_dtype)
                gumbel = torch.clamp(gumbel, min=-6.0, max=6.0)  # 修改：放宽范围

                # 计算概率分布（增加epsilon防止log(0)）
                rate_probs = F.softmax((rate_logits + gumbel) / temperature, dim=-1)
                rate_probs = torch.clamp(rate_probs, min=1e-7, max=1.0 - 1e-7)  # 修改：调整epsilon
                rate_pred = torch.argmax(rate_probs, dim=-1)

                print(f"[Router] rate_probs - 均值(0类): {rate_probs[..., 0].mean().item():.4f}, "
                      f"均值(1类): {rate_probs[..., 1].mean().item():.4f}, "
                      f"是否含nan: {torch.isnan(rate_probs).any().item()}")
            except Exception as e:
                print(f"[Router] Gumbel-Softmax计算错误: {str(e)}")
                # 异常时使用均匀分布作为 fallback
                rate_probs = torch.ones(B, L_img, 2, device=target_device) * 0.5
                rate_pred = torch.zeros(B, L_img, device=target_device, dtype=torch.long)

            # 4. 计算两种压缩特征（关键修复：增加中间结果检查）
            try:
                compressed_1_4 = self._compress_with_rate(visual_tokens, rate=4)
                compressed_1_4 = torch.clamp(compressed_1_4, min=-6.0, max=6.0)  # 修改：放宽范围
                print(f"[Router] 1/4压缩特征 - 均值: {compressed_1_4.mean().item():.4f}, "
                      f"最大值: {compressed_1_4.max().item():.4f}, "
                      f"是否含nan: {torch.isnan(compressed_1_4).any().item()}")
            except Exception as e:
                print(f"[Router] 1/4压缩计算错误: {str(e)}, 使用0矩阵替代")
                compressed_1_4 = torch.zeros(B, L_img // 4, D, device=target_device, dtype=target_dtype)

            try:
                compressed_1_16 = self._compress_with_rate(visual_tokens, rate=16)
                compressed_1_16 = torch.clamp(compressed_1_16, min=-6.0, max=6.0)  # 修改：放宽范围
                print(f"[Router] 1/16压缩特征 - 均值: {compressed_1_16.mean().item():.4f}, "
                      f"最大值: {compressed_1_16.max().item():.4f}, "
                      f"是否含nan: {torch.isnan(compressed_1_16).any().item()}")
            except Exception as e:
                print(f"[Router] 1/16压缩计算错误: {str(e)}, 使用0矩阵替代")
                compressed_1_16 = torch.zeros(B, L_img // 16, D, device=target_device, dtype=target_dtype)

            # 5. 对齐长度并混合（关键修复：权重归一化）
            max_len = max(compressed_1_4.shape[1], compressed_1_16.shape[1])
            compressed_1_4_padded = F.pad(compressed_1_4, (0, 0, 0, max_len - compressed_1_4.shape[1]))
            compressed_1_16_padded = F.pad(compressed_1_16, (0, 0, 0, max_len - compressed_1_16.shape[1]))

            # 归一化混合权重（防止权重和不为1）
            mix_weight_1_4 = rate_probs[..., 0].mean(dim=1, keepdim=True).unsqueeze(1)
            mix_weight_1_16 = rate_probs[..., 1].mean(dim=1, keepdim=True).unsqueeze(1)
            weight_sum = mix_weight_1_4 + mix_weight_1_16 + 1e-7  # 修改：调整epsilon
            mix_weight_1_4 = mix_weight_1_4 / weight_sum
            mix_weight_1_16 = mix_weight_1_16 / weight_sum

            # 混合特征并限制范围
            compressed_output = mix_weight_1_4 * compressed_1_4_padded + mix_weight_1_16 * compressed_1_16_padded
            compressed_output = torch.clamp(compressed_output, min=-6.0, max=6.0)  # 修改：放宽范围

            # 最终检查
            if torch.isnan(compressed_output).any() or torch.isinf(compressed_output).any():
                print(f"[Router] 混合后特征含nan/inf，使用1/4压缩特征替代")
                compressed_output = compressed_1_4_padded

            print(f"[Router] 混合后特征 - 均值: {compressed_output.mean().item():.4f}, "
                  f"最大值: {compressed_output.max().item():.4f}, "
                  f"是否含nan: {torch.isnan(compressed_output).any().item()}")
        else:
            # 推理阶段
            rate_probs = F.softmax(rate_logits, dim=-1)
            rate_pred = torch.argmax(rate_probs, dim=-1)
            compressed_output = self._compress_with_hard_selection(visual_tokens, rate_pred)

        if return_rate_pred:
            return compressed_output, rate_pred, rate_probs
        return compressed_output

    def _compress_with_rate(self, visual_tokens, rate):
        """按指定压缩率处理所有patch（增加异常处理）"""
        B, L_img, D = visual_tokens.shape
        compressed = []
        for b in range(B):
            tokens = visual_tokens[b]
            output = []
            idx = 0
            while idx < L_img:
                end_idx = min(idx + rate, L_img)
                group = tokens[idx:end_idx]

                # 防止空组导致的均值计算错误
                if group.numel() == 0:
                    output.append(torch.zeros(1, D, device=tokens.device, dtype=tokens.dtype))
                    idx = end_idx
                    continue

                # 投影并聚合
                if rate == 4:
                    proj = self.proj_1_4(group)
                else:  # 16
                    proj = self.proj_1_16(group)

                # 聚合前检查投影结果
                if torch.isnan(proj).any():
                    proj = torch.zeros_like(proj)
                aggregated = proj.mean(dim=0, keepdim=True)
                output.append(aggregated)
                idx = end_idx

            output = torch.cat(output, dim=0)
            compressed.append(output)

        # 对齐长度
        max_len = max(out.shape[0] for out in compressed)
        padded = [F.pad(out, (0, 0, 0, max_len - out.shape[0])) for out in compressed]
        return torch.stack(padded, dim=0)

    def _compress_with_hard_selection(self, visual_tokens, rate_pred):
        """推理时使用硬选择压缩（保持原始顺序）"""
        B, L_img, D = visual_tokens.shape
        compressed_outputs = []
        for b in range(B):
            sample_tokens = visual_tokens[b]
            sample_rate = rate_pred[b]
            sample_output = []
            idx = 0
            while idx < L_img:
                current_rate = self.compression_rates[sample_rate[idx].item()]
                end_idx = min(idx + current_rate, L_img)
                patch_group = sample_tokens[idx:end_idx]

                if current_rate == 4:
                    projected = self.proj_1_4(patch_group)
                else:  # 16
                    projected = self.proj_1_16(patch_group)

                # 检查投影结果
                if torch.isnan(projected).any():
                    projected = torch.zeros_like(projected)
                aggregated = projected.mean(dim=0, keepdim=True)
                sample_output.append(aggregated)
                idx = end_idx

            sample_output = torch.cat(sample_output, dim=0)
            compressed_outputs.append(sample_output)

        max_len = max(output.shape[0] for output in compressed_outputs)
        padded_outputs = [F.pad(output, (0, 0, 0, max_len - output.shape[0])) for output in compressed_outputs]
        return torch.stack(padded_outputs, dim=0)

class LLaVAViR_AMSS(nn.Module):
    """
    处理逻辑：完全对齐原模型prepare_inputs_labels_for_multimodal流程
    核心流程：拆分含<image>的文本序列 → 插入压缩后的图像特征 → 重组序列并处理掩码
    """

    def __init__(self, llava_base_model, vir_module, image_processor, tokenizer):
        super().__init__()
        self.llava = llava_base_model  # LLaVA基础模型（包含原处理逻辑）
        self.vir = vir_module  # ViR压缩模块（在原图像特征处理后插入）
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.vision_tower = self.llava.get_vision_tower()
        self.mm_projector = self.llava.model.mm_projector
        self.config = self.llava.config  # 复用原模型配置

    def encode_images(self, images):
        """在原模型图像编码后添加ViR压缩"""
        # 1. 原模型图像编码流程
        with torch.no_grad():
            image_features = self.vision_tower(images)  # [B, T, 1024]，T=576/577

        # 2. 应用ViR压缩（仅处理图像patch，保留CLS）
        if image_features.shape[1] == 577:  # 含CLS
            cls_emb = image_features[:, 0:1, :]  # [B, 1, 1024]
            patches = image_features[:, 1:, :]  # [B, 576, 1024]
            compressed_patches = self.vir(patches)  # [B, M, 1024]
            compressed_features = torch.cat([cls_emb, compressed_patches], dim=1)  # [B, M+1, 1024]
        else:  # 不含CLS（576维）
            compressed_patches = self.vir(image_features)  # [B, M, 1024]
            compressed_features = compressed_patches  # [B, M, 1024]

        # 3. 原模型投影流程
        return self.mm_projector(compressed_features)  # [B, M(+1), 4096]

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels,
            images, image_sizes=None
    ):
        """完全对齐原模型逻辑，仅在图像特征部分插入ViR压缩"""
        vision_tower = self.vision_tower
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # --------------------------
        # 1. 图像特征处理（替换原模型encode_images，加入ViR压缩）
        # --------------------------
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            # 关键修改：使用带ViR压缩的encode_images
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)

            # 处理图像拼接方式（复用原模型逻辑）
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                # 原模型空间拼接逻辑（保持不变）
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = vision_tower.num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
                        if image_aspect_ratio == 'anyres':
                            from llava.model.multimodal_encoder import get_anyres_image_grid_shape, unpad_image
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower.config.image_size
                            )
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.llava.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(
                                    image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.llava.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            # 关键修改：使用带ViR压缩的encode_images
            image_features = self.encode_images(images)

        # --------------------------
        # 2. 文本与图像特征融合（完全复用原模型逻辑）
        # --------------------------
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # 处理空值情况（保持原模型兼容）
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # 移除padding（保持原逻辑）
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 统计当前样本的<image>数量
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                # 无<image>标记时直接拼接空图像特征
                cur_image_features = image_features[cur_image_idx] if cur_image_idx < len(image_features) else None
                cur_input_embeds_1 = self.llava.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]],
                                             dim=0) if cur_image_features is not None else cur_input_embeds_1
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            # 定位<image>标记位置并拆分文本
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
                cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])

            # 生成文本嵌入并拆分
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.llava.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            # 拼接文本嵌入和图像特征
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    # 插入压缩后的图像特征（替换原图像特征）
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    # 图像位置标签设为IGNORE_INDEX（原模型逻辑）
                    cur_new_labels.append(torch.full(
                        (cur_image_features.shape[0],), IGNORE_INDEX,
                        device=cur_labels.device, dtype=cur_labels.dtype
                    ))

            # 处理设备一致性
            cur_new_input_embeds = [x.to(self.llava.device) for x in cur_new_input_embeds]
            # 合并序列
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # 截断过长序列（原模型逻辑）
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # 批次对齐（处理填充、掩码和位置ID）
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len), IGNORE_INDEX,
            dtype=new_labels[0].dtype, device=new_labels[0].device
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype,
            device=position_ids.device
        )

        # 根据padding方向处理（左对齐/右对齐）
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        # 恢复原模型输出格式
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def forward(self, input_ids, images=None, image_sizes=None, labels=None, **kwargs):
        """复用原模型forward流程，替换multimodal处理函数，确保传递所有必要参数"""
        # 提取必要参数（即使为None也显式传递）
        position_ids = kwargs.get('position_ids', None)
        attention_mask = kwargs.get('attention_mask', None)
        past_key_values = kwargs.get('past_key_values', None)

        # 调用自定义的prepare_inputs_labels_for_multimodal（显式传递所有必要参数）
        input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            images=images,
            image_sizes=image_sizes,
            labels=labels,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values
        )

        # 关键修改：根据是否有inputs_embeds决定是否传递input_ids
        # 当使用inputs_embeds时，不传递input_ids（避免冲突）
        llava_kwargs = {
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "labels": labels,** kwargs
        }

        if inputs_embeds is not None:
            # 使用预计算的嵌入向量时，只传递inputs_embeds
            llava_kwargs["inputs_embeds"] = inputs_embeds
        else:
            # 不使用嵌入向量时，传递input_ids
            llava_kwargs["input_ids"] = input_ids

        # 调用原模型forward（传递处理后的参数）
        outputs = self.llava(**llava_kwargs)
        return outputs

    def generate(self, input_ids, images=None, image_sizes=None, **generate_kwargs):
        """生成逻辑：复用原模型generate，确保图像处理一致"""
        # 预处理输入（与forward共享逻辑）
        _, position_ids, attention_mask, _, inputs_embeds, _ = self.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            images=images,
            image_sizes=image_sizes,
            labels=None,
            past_key_values=None
        )
        # 调用原模型生成
        return self.llava.generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,** generate_kwargs
        )


class PixelUnshuffleCompressor(nn.Module):
    """1/4压缩模块（用于参考模型固定压缩）"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv1d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        self.float()  # 修改：使用float32而不是half()

    def forward(self, x):
        B, N, C = x.shape
        if N % 4 != 0:
            raise ValueError(f"序列长度N={N}必须是4的倍数")

        x = x.permute(0, 2, 1)  # [B, C, N]

        # 两次2倍压缩
        x = x.reshape(B, C, 2, N // 2)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.reshape(B, C * 2, N // 2)

        x = x.reshape(B, C * 2, 2, (N // 2) // 2)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.reshape(B, C * 4, N // 4)

        x = self.conv(x)
        return x.permute(0, 2, 1)  # [B, N/4, out_channels]


class PixelUnshuffleCompressor16(nn.Module):
    """1/16压缩模块（用于参考模型固定压缩）"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv1d(in_channels * 16, out_channels, kernel_size=1, stride=1)
        self.float()  # 修改：使用float32而不是half()

    def forward(self, x):
        B, N, C = x.shape
        if N % 16 != 0:
            raise ValueError(f"序列长度N={N}必须是16的倍数")

        x = x.permute(0, 2, 1)  # [B, C, N]

        # 四次2倍压缩
        current_channels = C
        for _ in range(4):
            x = x.reshape(B, current_channels, 2, x.shape[2] // 2)
            x = x.permute(0, 1, 3, 2).contiguous()
            current_channels *= 2
            x = x.reshape(B, current_channels, x.shape[2])

        x = self.conv(x)
        return x.permute(0, 2, 1)  # [B, N/16, out_channels]


class ViCOTrainer:
    """修正版ViCO训练器（符合论文Section 2.3.3）"""

    def __init__(self, llava_vir_model, ref_llava_model, tokenizer, device="cuda"):
        self.device = device
        # 获取基础模型的数据类型
        self.dtype = llava_vir_model.dtype if hasattr(llava_vir_model, 'dtype') else torch.float16

        # 确保主模型和参考模型使用相同的数据类型
        self.model = llava_vir_model.to(device, dtype=self.dtype)
        self.ref_model = ref_llava_model.to(device, dtype=self.dtype)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 通道配置
        in_channels = self.model.vir.proj_1_4.in_features
        out_channels = self.model.vir.proj_1_4.out_features

        # 参考模型压缩模块（使用与模型一致的dtype）
        self.ref_pixel_shuffle_4 = PixelUnshuffleCompressor(in_channels, out_channels).to(device, dtype=self.dtype)
        self.ref_pixel_shuffle_16 = PixelUnshuffleCompressor16(in_channels, out_channels).to(device, dtype=self.dtype)

        # 投影层（确保与模型dtype一致）
        self.ref_mm_projector = copy.deepcopy(self.model.mm_projector).to(device, dtype=self.dtype)
        for param in self.ref_mm_projector.parameters():
            param.requires_grad = False

        self.model_mm_projector = copy.deepcopy(self.model.mm_projector).to(device, dtype=self.dtype)
        for param in self.model_mm_projector.parameters():
            param.requires_grad = True

        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
        for param in self.ref_pixel_shuffle_4.parameters():
            param.requires_grad = False
        for param in self.ref_pixel_shuffle_16.parameters():
            param.requires_grad = False

        # 可训练参数（包含ViR的router）
        self.trainable_params = [
            *self.model.vir.parameters(),  # 包含router_classifier
            *self.model_mm_projector.parameters()
        ]

        # 优化器设置（增强稳定性）
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=2e-7,  # 进一步降低学习率防止梯度爆炸
            weight_decay=0.01,
            eps=1e-8
        )

        # 添加梯度缩放器，用于混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()

        self.kl_loss = nn.KLDivLoss(reduction="none", log_target=True)
        self.IMAGE_TOKEN_INDEX = -200
        self.IGNORE_INDEX = -100

        # 新增：ViR模块关键参数监控列表
        self.vir_monitor_params = {
            'router_classifier': [p for n, p in self.model.vir.named_parameters() if 'router_classifier' in n],
            'proj_layers': [p for n, p in self.model.vir.named_parameters() if 'proj' in n]
        }

        # 新增：参数备份机制（用于ViR异常恢复）
        self.vir_param_backups = {
            'router_classifier': [p.data.clone() for p in self.vir_monitor_params['router_classifier']],
            'proj_layers': [p.data.clone() for p in self.vir_monitor_params['proj_layers']]
        }

    def _check_vir_parameters(self):
        """检查ViR模块参数是否异常"""
        has_nan = False

        # 检查路由器分类器参数
        for i, p in enumerate(self.vir_monitor_params['router_classifier']):
            if torch.isnan(p).any() or torch.isinf(p).any():
                print(f"[ViR参数异常] Router分类器参数{i}含nan/inf")
                has_nan = True

        # 检查投影层参数
        for i, p in enumerate(self.vir_monitor_params['proj_layers']):
            if torch.isnan(p).any() or torch.isinf(p).any():
                print(f"[ViR参数异常] 投影层参数{i}含nan/inf")
                has_nan = True

        return has_nan

    def _get_text_embeddings(self, model, input_ids):
        """获取过滤<image>token后的文本嵌入"""
        B, T = input_ids.shape
        embeds_list = []

        # 区分模型类型获取基础模型
        if hasattr(model, 'llava'):
            base_model = model.llava.get_model()
        else:
            base_model = model.get_model()

        embed_layer = base_model.embed_tokens
        vocab_size = embed_layer.weight.shape[0]

        for b in range(B):
            cur_input_ids = input_ids[b]
            is_image = (cur_input_ids == self.IMAGE_TOKEN_INDEX)
            is_invalid = (cur_input_ids < 0) | (cur_input_ids >= vocab_size)
            valid_mask = ~is_image & ~is_invalid

            cur_valid_ids = cur_input_ids[valid_mask]
            if cur_valid_ids.numel() == 0:
                cur_embeds = torch.empty(
                    (0, model.config.hidden_size),
                    device=cur_input_ids.device,
                    dtype=embed_layer.weight.dtype
                )
            else:
                cur_embeds = embed_layer(cur_valid_ids)
            embeds_list.append(cur_embeds)

        # 打印文本嵌入统计
        if embeds_list:
            embed_shapes = [emb.shape for emb in embeds_list]
            print(
                f"[文本嵌入] 样本形状: {embed_shapes[0]} (首样本), 有效token占比: {sum(valid_mask) / len(cur_input_ids):.2f}")

        return embeds_list

    def _merge_visual_text_embeds(self, llm_embeds_list, mm_hidden, input_ids, labels=None):
        """融合文本和视觉特征（保持对齐）"""
        B = input_ids.shape[0]
        device = input_ids.device
        new_input_embeds = []
        new_labels = [] if labels is not None else None

        for batch_idx in range(B):
            cur_input_ids = input_ids[batch_idx]
            cur_llm_embeds = llm_embeds_list[batch_idx]
            cur_labels = labels[batch_idx] if labels is not None else None

            # 定位<image>token位置
            image_positions = torch.where(cur_input_ids == self.IMAGE_TOKEN_INDEX)[0].tolist()
            num_images = len(image_positions)

            if num_images == 0:
                new_input_embeds.append(cur_llm_embeds)
                if labels is not None:
                    new_labels.append(cur_labels)
                continue

            # 拆分文本片段
            split_markers = [-1] + image_positions + [cur_input_ids.shape[0]]
            seg_lengths = [split_markers[i + 1] - split_markers[i] - 1 for i in range(num_images + 1)]
            split_embeds = torch.split(cur_llm_embeds, seg_lengths, dim=0)

            # 拼接文本和图像特征
            cur_new_embeds = []
            cur_new_labels = [] if labels is not None else None

            for i in range(num_images + 1):
                cur_new_embeds.append(split_embeds[i])
                if labels is not None:
                    start, end = split_markers[i] + 1, split_markers[i + 1]
                    cur_new_labels.append(cur_labels[start:end])

                if i < num_images:
                    img_feat = mm_hidden[batch_idx]
                    cur_new_embeds.append(img_feat)
                    if labels is not None:
                        img_label = torch.full(
                            (img_feat.shape[0],), self.IGNORE_INDEX,
                            device=device, dtype=cur_labels.dtype
                        )
                        cur_new_labels.append(img_label)

            # 合并片段
            final_embeds = torch.cat(cur_new_embeds, dim=0)
            new_input_embeds.append(final_embeds)

            if labels is not None:
                final_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(final_labels)

        # 批次对齐
        max_len = max(emb.shape[0] for emb in new_input_embeds) if new_input_embeds else 0
        batch_size = len(new_input_embeds)
        D = new_input_embeds[0].shape[1] if batch_size > 0 else 0

        padded_embeds = torch.zeros((batch_size, max_len, D), device=device,
                                    dtype=new_input_embeds[0].dtype if batch_size > 0 else torch.float16)
        attention_mask = torch.zeros((batch_size, max_len), device=device, dtype=torch.bool)
        position_ids = torch.zeros((batch_size, max_len), device=device, dtype=torch.long)
        padded_labels = None

        if labels is not None and new_labels:
            padded_labels = torch.full(
                (batch_size, max_len), self.IGNORE_INDEX,
                device=device, dtype=new_labels[0].dtype
            )

        padding_side = getattr(self.model.config, 'tokenizer_padding_side', 'right')
        for i in range(batch_size):
            cur_emb = new_input_embeds[i]
            cur_len = cur_emb.shape[0]

            if padding_side == 'right':
                padded_embeds[i, :cur_len] = cur_emb
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(cur_len, device=device)
                if labels is not None:
                    padded_labels[i, :cur_len] = new_labels[i]
            else:
                padded_embeds[i, max_len - cur_len:] = cur_emb
                attention_mask[i, max_len - cur_len:] = True
                position_ids[i, max_len - cur_len:] = torch.arange(cur_len, device=device)
                if labels is not None:
                    padded_labels[i, max_len - cur_len:] = new_labels[i]

        # 打印融合后特征信息
        print(
            f"[特征融合] 融合后形状: {padded_embeds.shape}, 有效注意力占比: {attention_mask.float().mean().item():.4f}")
        return {
            'input_embeds': padded_embeds,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': padded_labels
        }

    def _get_compressed_features(self, visual_embeds, compression_rate=None, use_vir=False):
        """获取压缩特征（支持动态/固定压缩）"""
        B, N, C = visual_embeds.shape
        print(f"[压缩前] 视觉特征形状: {visual_embeds.shape}, 均值: {visual_embeds.mean().item():.4f}")

        if use_vir:
            # 使用ViR动态压缩（返回动态长度特征）
            compressed = self.model.vir(visual_embeds)
            print(f"[动态压缩] ViR输出形状: {compressed.shape}, 压缩率: 动态")
            return compressed
        elif compression_rate == 1 / 4:
            compressed = self.ref_pixel_shuffle_4(visual_embeds)
            print(f"[固定压缩] 1/4压缩后形状: {compressed.shape}")
            return compressed
        elif compression_rate == 1 / 16:
            compressed = self.ref_pixel_shuffle_16(visual_embeds)
            print(f"[固定压缩] 1/16压缩后形状: {compressed.shape}")
            return compressed
        else:
            raise ValueError("压缩率必须为1/4、1/16或使用ViR动态压缩")

    def _restore_vir_parameters(self):
        """恢复ViR模块参数至最近备份"""
        print("[ViR参数恢复] 开始恢复上一轮有效参数")

        # 恢复路由器分类器
        for p, backup in zip(self.vir_monitor_params['router_classifier'], self.vir_param_backups['router_classifier']):
            p.data.copy_(backup)

        # 恢复投影层
        for p, backup in zip(self.vir_monitor_params['proj_layers'], self.vir_param_backups['proj_layers']):
            p.data.copy_(backup)

    def _backup_vir_parameters(self):
        """备份当前ViR模块参数"""
        self.vir_param_backups['router_classifier'] = [
            p.data.clone() for p in self.vir_monitor_params['router_classifier']
        ]
        self.vir_param_backups['proj_layers'] = [
            p.data.clone() for p in self.vir_monitor_params['proj_layers']
        ]

    def _get_compressed_features(self, visual_embeds, compression_rate=None, use_vir=False):
        """增强版：适配ViR模块的压缩特征获取"""
        B, N, C = visual_embeds.shape
        visual_embeds = torch.clamp(visual_embeds, min=-4.0, max=4.0)  # 修改：调整范围
        print(f"[压缩前] 视觉特征形状: {visual_embeds.shape}, 均值: {visual_embeds.mean().item():.4f}, "
              f"是否含nan: {torch.isnan(visual_embeds).any().item()}")

        if use_vir:
            # 调用修复后的ViR模块
            try:
                # 启用ViR的详细日志输出
                compressed = self.model.vir(visual_embeds)

                # 检查ViR输出
                if torch.isnan(compressed).any() or torch.isinf(compressed).any():
                    print(f"[紧急] ViR输出仍含nan/inf，使用1/4压缩 fallback")
                    return self.ref_pixel_shuffle_4(visual_embeds)  # 降级方案

                return compressed
            except Exception as e:
                print(f"[ViR调用错误] {str(e)}，使用1/4压缩 fallback")
                return self.ref_pixel_shuffle_4(visual_embeds)  # 降级方案

        elif compression_rate == 1 / 4:
            compressed = self.ref_pixel_shuffle_4(visual_embeds)
            return compressed
        elif compression_rate == 1 / 16:
            compressed = self.ref_pixel_shuffle_16(visual_embeds)
            return compressed
        else:
            raise ValueError("压缩率必须为1/4、1/16或使用ViR动态压缩")

    def train_consistency(self, dataloader, epochs=3):
        """增强版：适配ViR模块的一致性训练"""
        self.model.train()
        print(f"\n[训练开始] 启动一致性训练（适配增强版ViR），共{epochs}个epoch")

        for epoch in range(epochs):
            print(f"\n===== Epoch {epoch + 1}/{epochs} 开始 =====")

            # 每个epoch开始时检查并备份参数
            if self._check_vir_parameters():
                print("[警告] 发现上一轮遗留的异常参数，尝试恢复")
                self._restore_vir_parameters()
            self._backup_vir_parameters()  # 备份当前有效参数

            pbar = tqdm(dataloader, desc=f"Consistency Epoch {epoch + 1}/{epochs}")
            total_loss = 0.0
            nan_batch_count = 0  # 统计异常批次

            for batch_idx, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)
                images = batch["image"].to(self.device, dtype=torch.float16)  # 修改：使用float32
                B = input_ids.shape[0]

                # 1. 参考模型输出（固定1/4压缩）
                with torch.no_grad():
                    ref_visual_embeds = self.ref_model.get_vision_tower()(images)
                    if ref_visual_embeds.shape[1] == 577:
                        ref_visual_embeds = ref_visual_embeds[:, 1:577, :]
                    ref_compressed = self._get_compressed_features(ref_visual_embeds, 1 / 4)
                    ref_mm_hidden = self.ref_mm_projector(ref_compressed)
                    ref_llm_embeds_list = self._get_text_embeddings(self.ref_model, input_ids)
                    ref_merge_output = self._merge_visual_text_embeds(
                        ref_llm_embeds_list, ref_mm_hidden, input_ids
                    )
                    ref_outputs = self.ref_model(
                        input_ids=None,
                        inputs_embeds=ref_merge_output['input_embeds'],
                        attention_mask=ref_merge_output['attention_mask'],
                        position_ids=ref_merge_output['position_ids'],
                        return_dict=True
                    )
                    ref_logits = F.log_softmax(ref_outputs.logits, dim=-1)
                    ref_logits = torch.clamp(ref_logits, min=-150.0, max=0.0)  # 修改：调整范围

                # 2. 策略模型输出（使用修复后的ViR）
                xi_mask = torch.rand(B, device=self.device) < 0.5
                model_visual_embeds = self.model.llava.get_vision_tower()(images)
                if model_visual_embeds.shape[1] == 577:
                    model_visual_embeds = model_visual_embeds[:, 1:577, :]

                model_compressed_list = []
                max_length = 0
                for b in range(B):
                    if xi_mask[b]:
                        comp = self._get_compressed_features(model_visual_embeds[b:b + 1], 1 / 4)
                    else:
                        # 关键修改：使用增强版ViR动态压缩
                        comp = self._get_compressed_features(model_visual_embeds[b:b + 1], use_vir=True)
                    model_compressed_list.append(comp)
                    max_length = max(max_length, comp.shape[1])

                # 填充并拼接
                padded_compressed = [
                    F.pad(comp, (0, 0, 0, max_length - comp.shape[1]), mode='constant', value=0)
                    for comp in model_compressed_list
                ]
                model_compressed = torch.cat(padded_compressed, dim=0)

                # 检查压缩特征是否异常
                if torch.isnan(model_compressed).any():
                    print(f"[批次{batch_idx}] 压缩特征含nan，跳过当前批次")
                    nan_batch_count += 1
                    continue

                # 投影与融合
                model_mm_hidden = self.model_mm_projector(model_compressed)
                model_llm_embeds_list = self._get_text_embeddings(self.model, input_ids)
                model_merge_output = self._merge_visual_text_embeds(
                    model_llm_embeds_list, model_mm_hidden, input_ids
                )

                # 策略模型输出
                model_outputs = self.model(
                    input_ids=None,
                    inputs_embeds=model_merge_output['input_embeds'],
                    attention_mask=model_merge_output['attention_mask'],
                    position_ids=model_merge_output['position_ids'],
                    return_dict=True
                )
                model_logits = F.log_softmax(model_outputs.logits, dim=-1)
                model_logits = torch.clamp(model_logits, min=-150.0, max=0.0)  # 修改：调整范围

                # 检查logits是否异常
                if torch.isnan(model_logits).any() or torch.isnan(ref_logits).any():
                    print(f"[批次{batch_idx}] logits含nan，跳过当前批次")
                    nan_batch_count += 1
                    continue

                # 3. 计算KL损失
                min_seq_len = min(model_logits.shape[1], ref_logits.shape[1])
                model_logits_aligned = model_logits[:, :min_seq_len]
                ref_logits_aligned = ref_logits[:, :min_seq_len]
                attention_mask = ref_merge_output['attention_mask'][:, :min_seq_len]

                kl_per_token = self.kl_loss(model_logits_aligned + 1e-7, ref_logits_aligned + 1e-7)  # 修改：调整epsilon
                kl_per_token = kl_per_token * attention_mask.unsqueeze(-1)
                valid_token_count = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                kl_per_sample = kl_per_token.sum(dim=[1, 2]) / valid_token_count.squeeze(1)
                loss = kl_per_sample.mean()

                # 4. 梯度更新（适配ViR的特殊处理）
                self.optimizer.zero_grad()
                loss.backward()

                # 关键修改：针对ViR参数的梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.vir_monitor_params['router_classifier'], max_norm=0.2)  # 修改：调整梯度裁剪值
                torch.nn.utils.clip_grad_norm_(self.vir_monitor_params['proj_layers'], max_norm=0.2)  # 修改：调整梯度裁剪值
                torch.nn.utils.clip_grad_norm_(self.model_mm_projector.parameters(), max_norm=0.3)  # 修改：调整梯度裁剪值

                # 梯度清洗：移除NaN/inf梯度
                for param in self.trainable_params:
                    if param.grad is not None:
                        param.grad = torch.where(
                            torch.isnan(param.grad) | torch.isinf(param.grad),
                            torch.zeros_like(param.grad),
                            param.grad
                        )

                self.optimizer.step()

                # 检查参数是否异常，异常则恢复
                if self._check_vir_parameters():
                    print(f"[批次{batch_idx}] 参数更新后出现异常，恢复备份参数")
                    self._restore_vir_parameters()

                # 正常更新时备份参数
                self._backup_vir_parameters()

                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1 - nan_batch_count)
                pbar.set_postfix({
                    "Consistency Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{avg_loss:.4f}",
                    "NaN Batches": nan_batch_count
                })

            print(f"===== Epoch {epoch + 1}/{epochs} 结束 =====")
            print(f"[统计] 总异常批次: {nan_batch_count}/{len(dataloader)}")

    def train_router(self, dataloader, epochs=2):
        """阶段2：路由器训练（指导1/4与1/16压缩的选择）"""
        self.model.train()
        self.model_pixel_shuffle_16.eval()  # 固定压缩模块参数，仅训练路由器
        print(f"\n[训练开始] 启动路由器训练，共{epochs}个epoch，数据集大小: {len(dataloader)}批次")

        for epoch in range(epochs):
            print(f"\n===== Router Epoch {epoch + 1}/{epochs} 开始 =====")
            pbar = tqdm(dataloader, desc=f"ViCO Router Epoch {epoch + 1}/{epochs}")
            total_loss = 0.0
            for batch_idx, batch in enumerate(pbar):
                print(f"\n----- 批次 {batch_idx + 1}/{len(dataloader)} 处理 -----")
                input_ids = batch["input_ids"].to(self.device)
                images = batch["image"].to(self.device, dtype=torch.float16)
                print(f"[批次信息] 样本数量: {input_ids.shape[0]}, 图像形状: {images.shape}")

                # 生成标签
                rate_labels = self._compute_patch_level_labels(batch).view(-1)
                print(
                    f"[标签信息] 压缩率标签分布 - 1/4: {torch.sum(rate_labels == 0).item()}, 1/16: {torch.sum(rate_labels == 1).item()}")

                # 预测压缩率
                visual_embeds = self.model.vision_tower(images)
                if visual_embeds.shape[1] == 577:
                    visual_embeds = visual_embeds[:, 1:577, :]  # 移除CLS
                print(f"[视觉特征] 形状: {visual_embeds.shape}, 均值: {visual_embeds.mean().item():.4f}")

                rate_logits, _ = self.model.vir(visual_embeds, return_rate_pred=True)
                rate_logits = rate_logits.view(-1, 2)
                print(f"[Router输出] logits形状: {rate_logits.shape}, 均值: {rate_logits.mean().item():.4f}")

                # 计算损失
                loss = self.ce_loss(rate_logits, rate_labels)
                print(f"[损失计算] CE损失: {loss.item():.4f}")

                # 优化
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度统计
                grad_norms = []
                for param in self.model.vir.parameters():
                    if param.grad is not None:
                        grad_norms.append(torch.norm(param.grad).item())
                print(f"[梯度统计] Router平均梯度范数: {sum(grad_norms) / len(grad_norms):.6f}")

                torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=0.5)
                self.optimizer.step()

                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({"Router CE Loss": f"{loss.item():.4f}", "Avg Loss": f"{avg_loss:.4f}"})
                print(f"[批次结束] 累计平均损失: {avg_loss:.4f}")

            print(f"===== Router Epoch {epoch + 1}/{epochs} 结束，平均损失: {total_loss / len(dataloader):.4f} =====")

    def train(self, dataloader):
        """完整训练流程：先优化压缩模块，再训练路由器"""
        print("[完整训练] 开始执行两阶段训练流程")
        self.train_consistency(dataloader)
        self.train_router(dataloader)
        print("[完整训练] 所有训练阶段完成")
        return self.model


# AMSS核心工具函数（适配ViR压缩后维度）
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


def find_spans(input_ids_row: torch.Tensor, image_token_id: int, assistant_token_ids: list,
               img_len_fallback: int = 256):
    """适配ViR：用压缩后的img_len替换原577"""
    # 定位<image> token
    image_pos_tensor = (input_ids_row == image_token_id).nonzero(as_tuple=False)
    if image_pos_tensor.numel() == 0:
        raise ValueError("未找到 <image> token")
    image_pos = int(image_pos_tensor[0, 0].item())

    # 定位ASSISTANT: token
    L = input_ids_row.size(0)
    seq_len = len(assistant_token_ids)
    start_assist = None
    for i in range(image_pos, L - seq_len + 1):
        if input_ids_row[i:i + seq_len].tolist() == assistant_token_ids:
            start_assist = i + img_len_fallback - 1  # 适配压缩后长度
            break
    if start_assist is None:
        start_assist = L

    # 问题token区间（压缩后视觉token末尾到ASSISTANT:）
    q_start = image_pos + img_len_fallback
    q_end = start_assist
    if q_end <= q_start:
        q_end = q_start + 1

    return image_pos, img_len_fallback, q_start, q_end


def attention_mutual_information(attn_qv, attn_vq, eps=1e-8):
    """AMSS核心：计算模态互信息率"""
    T_q, T_v = attn_qv.shape
    # 构造联合分布
    joint = 0.5 * (attn_qv + attn_vq.T)
    joint = joint / (joint.sum() + eps)
    # 边缘概率
    p_q = joint.sum(dim=1, keepdim=True)
    p_v = joint.sum(dim=0, keepdim=True)
    # 避免log(0)
    joint_safe = joint + eps
    p_q_safe = p_q + eps
    p_v_safe = p_v + eps
    # 互信息与熵
    mi_matrix = joint * torch.log(joint / (p_q * p_v + eps) + eps)
    mi_val = mi_matrix.sum()
    H_q = -torch.sum(p_q * torch.log(p_q + eps))
    H_v = -torch.sum(p_v * torch.log(p_v + eps))
    # 互信息率（NMI）
    nmi_val_que = mi_val / ((H_q) + eps) if H_q > eps else torch.tensor(0.0, device=mi_val.device)
    nmi_val_img = mi_val / ((H_v) + eps) if H_v > eps else torch.tensor(0.0, device=mi_val.device)
    return nmi_val_que, nmi_val_img


def smooth_attention(attn_matrix, alpha=0.01):
    """注意力平滑（减少压缩后极端值）"""
    T1, T2 = attn_matrix.shape
    smooth = torch.full_like(attn_matrix, alpha / T2)
    return attn_matrix + smooth


def stable_row_norm(attn_matrix, eps=1e-8, min_sum=0.1):
    """稳定归一化（避免压缩后行和过小）"""
    row_sums = attn_matrix.sum(dim=1, keepdim=True)
    row_sums = torch.max(row_sums, torch.tensor(min_sum, device=row_sums.device))
    return attn_matrix / (row_sums + eps)


class AMSSRebalanceTrainer:
    """AMSS模态平衡训练器（适配ViR压缩后输入）"""

    def __init__(self, model, tokenizer, lambda_asa=0.1, tga_layers="last_4",
                 subnet_frac=0.6, mir_bins=8, mir_alpha=4.0, amss_a=1e-2, min_prob=1e-3, tau=0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.lambda_asa = lambda_asa
        self.tga_layers = tga_layers
        self.tau = tau

        # 按模态划分LoRA参数
        self.lora_param_names = [n for n, p in model.named_parameters() if p.requires_grad and "lora" in n]
        self.lora_params_by_modal = {
            "text": [n for n in self.lora_param_names if "q_proj" in n or "k_proj" in n],  # 文本模态
            "image": [n for n in self.lora_param_names if "vision" in n or "mm_projector" in n]  # 图像模态
        }

        # Fisher信息估计
        self.fisher_estimates = {
            modal: {n: torch.zeros_like(dict(model.named_parameters())[n].detach(),
                                        device=dict(model.named_parameters())[n].device,
                                        dtype=torch.float16)
                    for n in params}
            for modal, params in self.lora_params_by_modal.items()
        }
        self._fisher_count = 0

        # 超参
        self.subnet_frac = subnet_frac
        self.mir_bins = mir_bins
        self.mir_alpha = mir_alpha
        self.amss_a = amss_a
        self.min_prob = min_prob
        self._param_map = {n: p for n, p in model.named_parameters()}

    def compute_token_saliency(self, attn_mean, important_idx):
        """计算token显著性（适配压缩后注意力矩阵）"""
        L_query = attn_mean.size(0)
        s_i = torch.zeros(L_query, device=attn_mean.device, dtype=attn_mean.dtype)
        if important_idx is not None and important_idx.numel() > 0:
            proxy_score = attn_mean[important_idx].mean(dim=-1)
            norm = proxy_score.max().clamp(min=1e-6)
            s_i[important_idx] = proxy_score / norm
        return s_i

    def estimate_fisher(self):
        """累积Fisher信息"""
        for modal in ["text", "image"]:
            for n in self.lora_params_by_modal[modal]:
                p = self._param_map[n]
                if p.grad is not None:
                    self.fisher_estimates[modal][n] += (p.grad.detach().float() ** 2)
        self._fisher_count += 1

    def _sample_subnetworks(self, mir_dict):
        """子网采样（模态平衡优化）"""
        modal_active = {}
        exp_terms = {modal: torch.exp(torch.tensor(u / self.tau)) for modal, u in mir_dict.items()}
        sum_exp = sum(exp_terms.values())

        for modal, u in mir_dict.items():
            # 计算更新比例ρ
            exp_k = exp_terms[modal]
            rho_k = 1.0 - (exp_k / sum_exp).item()
            rho_k = max(0.1, min(0.9, rho_k))
            modal_params = self.lora_params_by_modal[modal]
            total = len(modal_params)
            if total == 0:
                modal_active[modal] = (set(), {})
                continue

            # Fisher重要性采样
            k = max(1, int(rho_k * total))
            importances = []
            for n in modal_params:
                f = self.fisher_estimates[modal].get(n)
                imp = float(f.mean().item()) if (f is not None and f.numel() > 0) else 1.0
                importances.append(max(imp, 0.0))
            imp_tensor = torch.tensor(importances, dtype=torch.float16)
            probs = imp_tensor / (imp_tensor.sum() + 1e-12) if imp_tensor.sum() > 0 else torch.ones_like(
                imp_tensor) / total
            probs = torch.clamp(probs, min=self.min_prob)
            probs = probs / probs.sum()

            # 采样
            try:
                selected_idx = torch.multinomial(probs, num_samples=k, replacement=False)
            except:
                selected_idx = torch.multinomial(probs, num_samples=k, replacement=True)
                selected_idx = torch.unique(selected_idx)[:k]
            selected_names = [modal_params[i] for i in selected_idx.tolist()]
            modal_active[modal] = (set(selected_names), {modal_params[i]: float(probs[i].item()) for i in range(total)})

        return modal_active

    def forward_backward(self, batch):
        """前向+反向（适配ViR压缩后输入）"""
        device = next(self.model.parameters()).device
        input_ids = batch["input_ids"].to(device)
        images = batch["image"].to(device, dtype=torch.float16)
        labels = input_ids.clone()
        labels[input_ids == IMAGE_TOKEN_INDEX] = -100
        pad_id = self.tokenizer.pad_token_id or 0
        labels[input_ids == pad_id] = -100
        attn_targets_list = batch["question_tokens"]

        # 1. 前向传播（获取压缩后注意力矩阵）
        outputs = self.model(input_ids=input_ids, images=images, labels=labels,
                             output_attentions=True, return_dict=True)
        lm_loss = outputs.loss
        compressed_img_len = outputs.compressed_img_len  # 从输出获取ViR压缩后长度

        # 2. 计算注意力均值（适配tga_layers）
        attn_stack = torch.stack(outputs.attentions, dim=0)  # [layers, B, heads, L, L]
        attn_head_mean = attn_stack.mean(dim=2)  # [layers, B, L, L]
        if self.tga_layers.startswith("last_"):
            k = int(self.tga_layers.split("_")[1])
            attn_mean = attn_head_mean[-k:].mean(dim=0)  # [B, L, L]
        else:
            attn_mean = attn_head_mean.mean(dim=0)

        # 3. 计算AMSS关键指标（互信息率、显著性）
        B = input_ids.size(0)
        s_scalar_list = []
        mir_list_q = []
        mir_list_i = []
        # 适配LLaVA的ASSISTANT token序列（"ASSISTANT:"的token ID）
        assistant_token_ids = self.tokenizer.encode("ASSISTANT:", add_special_tokens=False)

        for b in range(B):
            len_q = len(attn_targets_list[b])
            ids_row = input_ids[b]
            # 使用压缩后的图像长度提取注意力
            image_pos, img_len, q_start, q_end = find_spans(
                ids_row, IMAGE_TOKEN_INDEX, assistant_token_ids,
                img_len_fallback=compressed_img_len
            )

            # 提取压缩后的注意力矩阵（问题到图像 & 图像到问题）
            qa_to_image = attn_mean[b, q_end - len_q:q_end, image_pos:image_pos + img_len]  # [len_q, img_len]
            image_to_qa = attn_mean[b, image_pos:image_pos + img_len, q_end - len_q:q_end]  # [img_len, len_q]

            # 注意力处理（平滑+归一化）
            qa_to_image = smooth_attention(qa_to_image)
            qa_to_image = stable_row_norm(qa_to_image)
            image_to_qa = smooth_attention(image_to_qa)
            image_to_qa = stable_row_norm(image_to_qa)

            # 计算显著性（简单用前10%作为重要token）
            if len_q > 0:
                important_idx = torch.topk(qa_to_image.mean(dim=1), k=max(1, len_q // 10)).indices
            else:
                important_idx = None
            s_i = self.compute_token_saliency(qa_to_image, important_idx)
            s_scalar_list.append(s_i.mean() if s_i.numel() > 0 else torch.tensor(0.0, device=device))

            # 计算互信息率
            mi_q, mi_i = attention_mutual_information(qa_to_image, image_to_qa)
            mir_list_q.append(mi_q)
            mir_list_i.append(mi_i)

        # 4. 批次指标平均
        s_mean = torch.stack(s_scalar_list).float().mean() if s_scalar_list else torch.tensor(0.0, device=device)
        mir_mean_q = torch.stack(mir_list_q).float().mean() if mir_list_q else torch.tensor(0.0, device=device)
        mir_mean_i = torch.stack(mir_list_i).float().mean() if mir_list_i else torch.tensor(0.0, device=device)
        mir_dict = {"text": mir_mean_q.item(), "image": mir_mean_i.item()}

        # 5. 反向传播与梯度处理
        self.model.zero_grad()

        if torch.isnan(lm_loss):
            debug_data = {
                "stage": "AMSS_forward_backward",
                "lm_loss": None,
                "mir_mean_q": float(mir_mean_q.item()),
                "mir_mean_i": float(mir_mean_i.item()),
                "s_mean": float(s_mean.item()),
            }
            log_debug_info(debug_data)
            return  # 直接返回，避免崩溃
        else:
            debug_data = {
                "stage": "AMSS_forward_backward",
                "lm_loss": float(lm_loss.item()),
                "mir_mean_q": float(mir_mean_q.item()),
                "mir_mean_i": float(mir_mean_i.item()),
                "s_mean": float(s_mean.item()),
            }
            log_debug_info(debug_data)

        lm_loss.backward(retain_graph=True)
        self.estimate_fisher()  # 累积Fisher信息

        # Fisher归一化
        fisher_count = max(1, self._fisher_count)
        fisher_norm = {modal: {n: f / fisher_count for n, f in self.fisher_estimates[modal].items()} for modal in
                       ["text", "image"]}

        # 子网采样
        modal_active = self._sample_subnetworks(mir_dict)

        # 模态MIR缩放因子
        mir_scales = {}
        for modal in ["text", "image"]:
            u = mir_mean_q if modal == "text" else mir_mean_i
            mir_norm = (u / (u + 1.0)).clamp(0.0, 1.0)
            mir_scales[modal] = 1.0 + self.mir_alpha * mir_norm

        # 梯度缩放与掩码
        with torch.no_grad():
            for modal in ["text", "image"]:
                active_names, p_map = modal_active[modal]
                for n in self.lora_params_by_modal[modal]:
                    p = self._param_map[n]
                    if p.grad is None:
                        continue
                    # 未采样参数梯度清零
                    if n not in active_names:
                        p.grad.zero_()
                        continue
                    # 采样概率与Fisher因子
                    p_j = max(p_map.get(n, self.min_prob), self.min_prob)
                    fisher = fisher_norm[modal].get(n, torch.ones_like(p.data)).to(p.grad.device).to(p.grad.dtype)
                    fisher_scalar = max(float(fisher.mean().item()), 1e-12)
                    fisher_factor = s_mean.to(p.grad.dtype) / (fisher_scalar + 1e-6)
                    # AMSS无偏项与MIR缩放
                    amss_unbias = 1.0 / (p_j + self.amss_a)
                    mir_scale = mir_scales[modal].to(p.grad.dtype)
                    # 最终梯度缩放
                    scale = fisher_factor * float(amss_unbias) * float(mir_scale)
                    scale = min(scale, 1e6)
                    p.grad.mul_(scale)

        # 梯度裁剪
        trainable_params = [self._param_map[n] for modal in ["text", "image"] for n in self.lora_params_by_modal[modal]
                            if self._param_map[n].requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        # 返回训练统计
        return {
            "lm_loss": lm_loss.item(),
            "s_mean": s_mean.item(),
            "mir_mean_q": mir_mean_q.item(),
            "mir_mean_i": mir_mean_i.item(),
            "active_text_params": len(modal_active["text"][0]),
            "active_image_params": len(modal_active["image"][0]),
            "total_text_params": len(self.lora_params_by_modal["text"]),
            "total_image_params": len(self.lora_params_by_modal["image"])
        }


if __name__ == "__main__":
    # 环境配置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载原始LLaVA-v1.6基座
    model_path = "/data2/gaodz/llava-v1.6-vicuna-7b"
    tokenizer, base_llava, image_processor, _ = load_pretrained_model(
        model_path, None, "llava_v1.6", device=device
    )

    # 2. 初始化ViR模块
    vir_module = VisualResolutionRouter(input_dim=1024, dtype=torch.float16)

    # 3. 集成LLaVA+ViR
    llava_vir_model = LLaVAViR_AMSS(
        llava_base_model=base_llava,
        tokenizer=tokenizer,
        vir_module=vir_module,
        image_processor=image_processor,
    ).to(device)

    # 4. 加载数据集
    dataset = MultimodalTGADataset(
        json_file="/data2/gaodz/Re-Align/llava_instruct_11k.json",
        image_folder="/data2/gaodz/train2014",
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=512
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 5. 第一阶段：ViCO训练（仅训ViR）
    vico_trainer = ViCOTrainer(
        llava_vir_model=llava_vir_model,
        ref_llava_model=base_llava,
        tokenizer=tokenizer,
        device=device
    )
    llava_vir_model = vico_trainer.train(dataloader)
    print("[ViCO训练完成] ViR模块已适配LLaVA")

    # 6. 第二阶段：AMSS LoRA训练（训模态平衡）
    # 6.1 初始化LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj"],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False
    )
    model = get_peft_model(llava_vir_model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数比例

    # 6.2 冻结ViR（仅训LoRA）
    for p in model.vir.parameters():
        p.requires_grad = False

    # 6.3 初始化AMSS训练器
    amss_trainer = AMSSRebalanceTrainer(
        model=model,
        tokenizer=tokenizer,
        lambda_asa=0.1,
        tga_layers="last_4",
        subnet_frac=0.6,
        mir_bins=8,
        mir_alpha=4.0
    )

    # 6.4 优化器
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n],
        lr=1e-6,
        weight_decay=0.01
    )

    # 6.5 AMSS训练循环
    num_epochs = 2
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"AMSS Epoch {epoch}")
        model.train()
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            try:
                stats = amss_trainer.forward_backward(batch)
            except Exception as e:
                print(f"[WARN] batch {i} 出错: {str(e)}")
                continue

            optimizer.step()
            model.zero_grad()
            # 进度条显示
            pbar.set_postfix({
                "LM": f"{stats['lm_loss']:.4f}",
                "Saliency": f"{stats['s_mean']:.4f}",
                "MIR(Text)": f"{stats['mir_mean_q']:.4f}",
                "MIR(Image)": f"{stats['mir_mean_i']:.4f}",
                "Text-Active": f"{stats['active_text_params']}/{stats['total_text_params']}",
                "Image-Active": f"{stats['active_image_params']}/{stats['total_image_params']}"
            })

    # 7. 保存模型
    save_dir = "/data2/gaodz/llava_v1.6_vir_amss"
    os.makedirs(save_dir, exist_ok=True)
    # 保存LoRA权重
    model.save_pretrained(save_dir)
    # 保存ViR权重
    torch.save(llava_vir_model.vir.state_dict(), os.path.join(save_dir, "vir_state_dict.pth"))
    print(f"[模型保存完成] 路径: {save_dir}")
