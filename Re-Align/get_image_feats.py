import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from lang_sam import LangSAM
from sklearn.cluster import KMeans
from transformers import CLIPModel, CLIPProcessor

# --------------------------
# 配置参数（完整保留所有必要配置）
# --------------------------
INPUT_JSONS = [
    "/data2/gaodz/sam_data/coco_with_detail_partial.json",
    "/data2/gaodz/sam_data/wikiart_with_detail_partial.json"
]
OUTPUT_JSON = "/data2/gaodz/sam_data/preprocessed_combined_dataset.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 子空间配置（保持原维度定义，确保特征维度匹配）
SUBSPACE_CONFIG = {
    "color": {"feat_dim": 768, "win_size": 150, "step": 50, "multi_word": True},
    "shape": {"feat_dim": 768, "win_size": 200, "step": 50, "multi_word": False},
    "material": {"feat_dim": 768, "win_size": 100, "step": 30, "multi_word": True},
    "object": {"feat_dim": 768, "win_size": 200, "step": 50, "multi_word": True},
    "scene": {"feat_dim": 768, "win_size": -1, "step": -1, "multi_word": False},  # 全局特征
    "emotion": {"feat_dim": 1536, "win_size": 150, "step": 50, "multi_word": True}  # 局部+全局拼接
}
IOU_THRESHOLD = 0.7
DEFAULT_MASK_NUM = 3  # 每个子空间保留3个掩码区域
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"  # CLIP模型（输出768维特征）
# 可选：若需使用自定义缓存目录（如之前的sam2_model），可添加以下配置
# CUSTOM_CACHE_DIR = "/data2/gaodz/sam2_model"
# os.environ["HF_HOME"] = CUSTOM_CACHE_DIR
# os.environ["TORCH_HOME"] = CUSTOM_CACHE_DIR

# --------------------------
# 初始化模型（完整加载CLIP和LangSAM）
# --------------------------
# 加载CLIP模型（添加类型转换，避免BFloat16问题）
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
# 关键：将CLIP模型参数转换为float32，解决BFloat16不支持问题
clip_model = clip_model.float()
clip_model.eval()  # 推理模式，禁用梯度计算

# 初始化LangSAM（用于生成文本引导的掩码）
lang_sam = LangSAM(device=DEVICE)


# --------------------------
# 核心工具函数（补充缺失函数，修复类型问题）
# --------------------------
def safe_str_convert(value):
    """安全转换任意值为字符串，避免None/特殊类型报错"""
    if value is None:
        return ""
    try:
        # 处理列表/元组（如多个属性词）
        if isinstance(value, (list, tuple)):
            return [safe_str_convert(item) for item in value]
        # 处理普通类型
        return str(value).strip()
    except Exception:
        return "unknown"  # 极端情况兜底


def calculate_iou(mask1, mask2):
    """计算两个二进制掩码的交并比（IoU），用于掩码去重"""
    # 确保掩码为布尔型（避免数值类型偏差）
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    # 计算交集和并集
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    # 避免除零错误
    return intersection / union if union > 0 else 0.0


def get_clip_feature(image, region=None):
    """提取CLIP图像特征（支持全局/局部区域，修复BFloat16问题）"""
    if region is not None:
        # 裁剪局部区域（确保坐标为整数，避免PIL报错）
        x1, y1, x2, y2 = [int(coord) for coord in region]
        # 边界校验：避免裁剪坐标超出图像范围
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.width, x2)
        y2 = min(image.height, y2)
        region_img = image.crop((x1, y1, x2, y2))
    else:
        # 全局特征：使用原图
        region_img = image

    # CLIP预处理（保持与模型输入要求一致）
    inputs = clip_processor(
        images=region_img,
        return_tensors="pt",
        padding=True,
        truncation=False
    ).to(DEVICE)

    # 提取特征（禁用梯度，加速推理）
    with torch.no_grad():
        # 强制输出为float32，解决下游不支持BFloat16问题
        feat = clip_model.get_image_features(**inputs).squeeze().float()
        # 转换为numpy数组，便于后续处理
        feat = feat.cpu().numpy().astype(np.float32)

    return feat


def get_multi_word_masks_by_lang_sam(image, attr_words, subspace_name):
    """用LangSAM生成文本引导的掩码+边界框，支持多属性词"""
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    all_masks = []
    all_scores = []
    all_labels = []
    all_bboxes = []

    # 1. 清洗属性词（过滤无效值）
    valid_words = []
    # 安全转换属性词（处理列表/单个值）
    attr_words = safe_str_convert(attr_words)
    # 若为列表，展开处理；若为单个词，转为列表
    if isinstance(attr_words, list):
        raw_words = attr_words
    else:
        raw_words = [attr_words]

    # 过滤空值/无效词
    for word in raw_words:
        if word not in ["", "none", "unknown", []]:
            valid_words.append(word)

    # 若无有效词，使用子空间默认词
    if not valid_words:
        default_words = {
            "color": "colored region",
            "shape": "shaped object",
            "material": "textured surface",
            "object": "main object",
            "scene": "whole scene",
            "emotion": "emotional region"
        }
        valid_words = [default_words[subspace_name]]

    # 2. LangSAM生成掩码（逐词生成，避免多词干扰）
    for word in valid_words:
        try:
            # LangSAM预测（单词输入，提高掩码准确性）
            results = lang_sam.predict([image], [word])[0]
            # 处理每个预测结果
            for mask, score, label in zip(results["masks"], results["scores"], results["labels"]):
                # 过滤过小掩码（避免噪声）
                if mask.sum() < 100:  # 掩码像素数阈值，可调整
                    continue
                # 计算掩码边界框（x1,y1:左上角；x2,y2:右下角）
                y_idx, x_idx = np.where(mask)
                bbox = (x_idx.min(), y_idx.min(), x_idx.max(), y_idx.max())
                # 存储结果
                all_masks.append(mask)
                all_scores.append(score.item())  # 转为Python数值
                all_labels.append(safe_str_convert(label))
                all_bboxes.append(bbox)
        except Exception as e:
            # 单词失败时，用全图掩码兜底
            print(f"警告：{subspace_name}子空间（词：{word}）生成失败：{str(e)}")
            default_mask = np.ones((h, w), dtype=bool)
            all_masks.append(default_mask)
            all_scores.append(0.0)
            all_labels.append(f"default_{subspace_name}")
            all_bboxes.append((0, 0, w, h))

    # 3. 掩码去重（基于IoU阈值）
    unique_masks = []
    unique_scores = []
    unique_labels = []
    unique_bboxes = []
    for idx in range(len(all_masks)):
        current_mask = all_masks[idx]
        current_score = all_scores[idx]
        current_label = all_labels[idx]
        current_bbox = all_bboxes[idx]
        # 检查是否与已保留掩码重复
        is_duplicate = False
        for saved_idx in range(len(unique_masks)):
            iou = calculate_iou(current_mask, unique_masks[saved_idx])
            if iou > IOU_THRESHOLD:
                is_duplicate = True
                # 保留分数更高的掩码
                if current_score > unique_scores[saved_idx]:
                    unique_masks[saved_idx] = current_mask
                    unique_scores[saved_idx] = current_score
                    unique_labels[saved_idx] = current_label
                    unique_bboxes[saved_idx] = current_bbox
                break
        if not is_duplicate:
            unique_masks.append(current_mask)
            unique_scores.append(current_score)
            unique_labels.append(current_label)
            unique_bboxes.append(current_bbox)

    # 4. 补齐掩码数量（确保至少有DEFAULT_MASK_NUM个）
    while len(unique_masks) < DEFAULT_MASK_NUM:
        unique_masks.append(np.ones((h, w), dtype=bool))
        unique_scores.append(0.0)
        unique_labels.append(f"fill_{subspace_name}")
        unique_bboxes.append((0, 0, w, h))

    # 5. 按置信度排序，取前DEFAULT_MASK_NUM个
    sorted_indices = np.argsort(unique_scores)[::-1]  # 降序排序
    final_masks = [unique_masks[i] for i in sorted_indices[:DEFAULT_MASK_NUM]]
    final_scores = [unique_scores[i] for i in sorted_indices[:DEFAULT_MASK_NUM]]
    final_labels = [unique_labels[i] for i in sorted_indices[:DEFAULT_MASK_NUM]]
    final_bboxes = [unique_bboxes[i] for i in sorted_indices[:DEFAULT_MASK_NUM]]

    # 日志输出（便于调试）
    print(f"[{subspace_name}] 词：{valid_words} → 生成{len(final_masks)}个掩码（标签：{final_labels}）")
    return final_masks, final_scores, final_labels, final_bboxes


# --------------------------
# 子空间特征提取函数（完整实现，匹配配置维度）
# --------------------------
def extract_color_feats(image, masks, bboxes):
    """颜色子空间特征：取前2个掩码区域的CLIP特征平均值（768维）"""
    feats = []
    # 取前2个高置信度区域
    for bbox in bboxes[:2]:
        feat = get_clip_feature(image, region=bbox)
        feats.append(feat)
    # 若不足2个区域，用全局特征补充
    while len(feats) < 1:
        feats.append(get_clip_feature(image))
    # 平均特征（保持768维）
    return np.mean(feats, axis=0).tolist()


def extract_shape_feats(image, masks, bboxes):
    """形状子空间特征：取面积最大的掩码区域CLIP特征（768维）"""
    # 计算每个掩码的面积
    mask_areas = [mask.sum() for mask in masks]
    # 取面积最大的掩码对应边界框
    max_area_idx = np.argmax(mask_areas)
    feat = get_clip_feature(image, region=bboxes[max_area_idx])
    return feat.tolist()


def extract_material_feats(image, masks, bboxes):
    """材质子空间特征：取置信度最高的掩码区域CLIP特征（768维）"""
    # 第1个掩码是置信度最高的（已排序）
    feat = get_clip_feature(image, region=bboxes[0])
    return feat.tolist()


def extract_object_feats(image, masks, bboxes):
    """物体子空间特征：前3个区域CLIP特征平均值（768维）"""
    feats = [get_clip_feature(image, region=bbox) for bbox in bboxes[:3]]
    # 平均后保持768维
    return np.mean(feats, axis=0).tolist()


def extract_scene_feats(image):
    """场景子空间特征：全图CLIP特征（768维）"""
    return get_clip_feature(image).tolist()


def extract_emotion_feats(image, masks, bboxes):
    """情感子空间特征：局部+全局CLIP特征拼接（768*2=1536维）"""
    # 局部特征：置信度最高的区域
    local_feat = get_clip_feature(image, region=bboxes[0])
    # 全局特征：全图
    global_feat = get_clip_feature(image)
    # 拼接为1536维
    return np.concatenate([local_feat, global_feat]).tolist()


# --------------------------
# 主预处理函数（完整逻辑，含错误处理）
# --------------------------
def preprocess_dataset():
    # 创建输出目录（若不存在）
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    # 第一步：统计总样本数（用于进度条）
    total_samples = 0
    for json_path in INPUT_JSONS:
        if not os.path.exists(json_path):
            print(f"警告：文件不存在，跳过统计：{json_path}")
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                # 统计有效JSON行（排除空行和非字典数据）
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if isinstance(data, dict):
                        total_samples += 1
        except Exception as e:
            print(f"警告：统计文件{json_path}时出错：{str(e)}，跳过该文件")

    # 第二步：处理并保存样本
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f_out, \
            tqdm(total=total_samples, desc="LangSAM+CLIP预处理") as pbar:

        for json_idx, json_path in enumerate(INPUT_JSONS):
            if not os.path.exists(json_path):
                print(f"\n跳过不存在的文件：{json_path}")
                continue
            print(f"\n开始处理第{json_idx + 1}/{len(INPUT_JSONS)}个文件：{json_path}")

            try:
                with open(json_path, "r", encoding="utf-8") as f_in:
                    for line_idx, line in enumerate(f_in, 1):
                        line = line.strip()
                        if not line:
                            continue  # 跳过空行

                        # 1. 解析JSON样本
                        try:
                            sample = json.loads(line)
                            if not isinstance(sample, dict):
                                print(f"文件{json_path}第{line_idx}行：非字典数据，跳过")
                                continue
                        except json.JSONDecodeError as e:
                            print(f"文件{json_path}第{line_idx}行：JSON解析失败（{str(e)}），跳过")
                            continue

                        # 2. 提取detail字段（确保存在，避免后续报错）
                        if "detail" not in sample:
                            sample["detail"] = {}
                        detail = sample["detail"]
                        if not isinstance(detail, dict):
                            detail = {}  # 强制转为字典

                        # 3. 加载图像（核心步骤，含完整错误处理）
                        try:
                            # 拼接图像路径
                            image_root = sample.get("image_root", "")
                            image_rel_path = sample.get("image", "")
                            if not image_root or not image_rel_path:
                                raise ValueError("样本缺少image_root或image字段")
                            image_path = os.path.join(image_root, image_rel_path)

                            # 检查路径有效性
                            if not os.path.exists(image_path):
                                raise FileNotFoundError(f"图像路径不存在：{image_path}")
                            if not os.path.isfile(image_path):
                                raise ValueError(f"路径不是文件：{image_path}")

                            # 加载并预处理图像（转为RGB，避免通道问题）
                            image = Image.open(image_path).convert("RGB")
                            if image.size[0] == 0 or image.size[1] == 0:
                                raise ValueError("加载的图像为空（宽/高为0）")
                        except Exception as e:
                            print(f"文件{json_path}第{line_idx}行：图像加载失败（{str(e)}），跳过")
                            pbar.update(1)
                            continue

                        # 4. 提取所有子空间特征
                        image_feats = {}
                        # 存储当前样本的掩码标签（用于后续分析）
                        sample_mask_labels = {}
                        for subspace in SUBSPACE_CONFIG.keys():
                            # 获取当前子空间的属性词
                            attr_words = detail.get(subspace, [])
                            # 生成掩码和边界框
                            try:
                                masks, scores, labels, bboxes = get_multi_word_masks_by_lang_sam(
                                    image=image,
                                    attr_words=attr_words,
                                    subspace_name=subspace
                                )
                                sample_mask_labels[subspace] = labels  # 保存标签
                            except Exception as e:
                                print(f"文件{json_path}第{line_idx}行：{subspace}掩码生成失败（{str(e)}），用兜底值")
                                # 兜底掩码（全图）
                                h, w = image.size[1], image.size[0]  # PIL.size是(w,h)，此处转为(h,w)
                                masks = [np.ones((h, w), dtype=bool) for _ in range(DEFAULT_MASK_NUM)]
                                bboxes = [(0, 0, w, h) for _ in range(DEFAULT_MASK_NUM)]
                                labels = [f"error_{subspace}"] * DEFAULT_MASK_NUM
                                sample_mask_labels[subspace] = labels

                            # 提取子空间特征
                            try:
                                if subspace == "color":
                                    feat = extract_color_feats(image, masks, bboxes)
                                elif subspace == "shape":
                                    feat = extract_shape_feats(image, masks, bboxes)
                                elif subspace == "material":
                                    feat = extract_material_feats(image, masks, bboxes)
                                elif subspace == "object":
                                    feat = extract_object_feats(image, masks, bboxes)
                                elif subspace == "scene":
                                    feat = extract_scene_feats(image)
                                elif subspace == "emotion":
                                    feat = extract_emotion_feats(image, masks, bboxes)
                                else:
                                    feat = [0.0] * SUBSPACE_CONFIG[subspace]["feat_dim"]

                                # 维度校验（确保与配置一致）
                                expected_dim = SUBSPACE_CONFIG[subspace]["feat_dim"]
                                if len(feat) != expected_dim:
                                    raise ValueError(f"特征维度不匹配（预期{expected_dim}，实际{len(feat)}）")
                                image_feats[subspace] = feat
                            except Exception as e:
                                # 特征提取失败，用零向量兜底
                                expected_dim = SUBSPACE_CONFIG[subspace]["feat_dim"]
                                image_feats[subspace] = [0.0] * expected_dim
                                print(f"文件{json_path}第{line_idx}行：{subspace}特征提取失败（{str(e)}），用零向量")

                        # 5. 更新样本数据（添加特征和掩码标签）
                        sample["detail"]["image_feats"] = image_feats
                        sample["detail"]["mask_labels"] = sample_mask_labels

                        # 6. 保存样本（JSON格式，确保中文正常显示）
                        try:
                            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        except Exception as e:
                            print(f"文件{json_path}第{line_idx}行：样本保存失败（{str(e)}），跳过")

                        # 更新进度条
                        pbar.update(1)

            except Exception as e:
                print(f"处理文件{json_path}时发生严重错误（{str(e)}），跳过该文件")
                continue

    # 处理完成提示
    print(f"\n预处理完成！输出文件：{OUTPUT_JSON}")
    print(f"预期处理样本数：{total_samples}，实际处理样本数：{pbar.n}")


# --------------------------
# 执行主函数
# --------------------------
if __name__ == "__main__":
    preprocess_dataset()