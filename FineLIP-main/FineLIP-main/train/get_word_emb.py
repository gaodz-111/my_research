import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import json
import torch
from textblob import TextBlob
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'model'))

from tqdm import tqdm
from model import finelip
from model.simple_tokenizer import SimpleTokenizer
# --------------------------
# 1. 配置参数（和训练脚本一致）
# --------------------------
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
save_path = "/data2/gaodz/FineLIP-main/word_feat_cache.pt"  # 保存词向量的路径
base_model = "ViT-L/14"  # 和训练用的模型一致
batch_size = 32  # GPU 批量编码大小（根据显存调整，越大越快）


# ========== 初始化 tokenizer ==========
simple_tokenizer = SimpleTokenizer()

# ========== 提取唯一词（和训练一致的分词逻辑） ==========
def extract_all_unique_words(datasets_config):
    all_unique_words = set()

    for cfg in datasets_config:
        data_json_path = cfg["data_json_path"]
        print(f"处理 {data_json_path} ...")

        with open(data_json_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    item = json.loads(line)
                    text = item.get("long_text", item.get("text", "")).strip()
                    if not text:
                        continue

                    # 完全一致的分词流程
                    input_ids = finelip.tokenize([text], truncate=True)[0]  # tensor -> list
                    tokens = [t.replace("</w>", "") for t in simple_tokenizer.decode(input_ids.tolist()).split()]

                    # 过滤非字母词
                    words = [word for word in tokens if word.isalpha()]
                    all_unique_words.update(words)

                except Exception as e:
                    print(f"跳过错误行: {e}")
                    continue

    print(f"共提取到 {len(all_unique_words)} 个唯一词")
    return list(all_unique_words)

# ========== GPU 批量编码 ==========
def gpu_precompute_word_feats(unique_words, save_path, base_model, batch_size):
    clip_model, _ = finelip.load_from_clip(base_model, device='cuda:0', run_finelip=False)
    clip_model.eval()

    word_feat_dict = {}
    total_batches = (len(unique_words) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(total_batches), desc="GPU 编码词向量"):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(unique_words))
            batch_words = unique_words[start:end]

            # Tokenize（批量）
            batch_tokens = finelip.tokenize(batch_words, truncate=True).cuda()

            # 编码
            batch_feats = clip_model.encode_text_full(batch_tokens).mean(dim=1)  # [B, D]

            # 转CPU存字典
            for word, feat in zip(batch_words, batch_feats.cpu()):
                word_feat_dict[word] = feat

    torch.save(word_feat_dict, save_path)
    print(f"词向量已保存到 {save_path}")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    unique_words = extract_all_unique_words(datasets_config)
    gpu_precompute_word_feats(unique_words, save_path, base_model, batch_size)