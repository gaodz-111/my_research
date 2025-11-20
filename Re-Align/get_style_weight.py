import os
os.environ["CUDA_VISIBLE_DEVICES"]='7'
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
import torch
from collections import defaultdict
from tqdm import tqdm

import json
from pathlib import Path
from collections import defaultdict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TokenSimilarityMatcher:
    def __init__(self, anchor_path, related_path, model_name="/data2/gaodz/bge_m3"):
        self.model = SentenceTransformer(model_name)
        # 加载预计算的向量表（格式：{token: 向量}）
        self.anchor_tokens, self.anchor_vecs = self._load_vectors(anchor_path)  # 锚点词列表 + 向量矩阵
        self.related_tokens, self.related_vecs = self._load_vectors(related_path)  # 描述词列表 + 向量矩阵

    def _load_vectors(self, path):
        """加载向量表，返回token列表和向量矩阵（便于批量计算相似度）"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tokens = list(data.keys())
        # 将向量列表转为矩阵（shape: [num_tokens, embed_dim]）
        vecs = torch.tensor([data[token][0] for token in tokens], dtype=torch.float32).to(device)
        return tokens, vecs

    @staticmethod
    def precompute_vectors(token_list, model, save_path):
        """预计算token向量（单个token对应一个向量）"""
        token_vecs = {}
        for token in token_list:
            # 对单个token编码
            vec = model.encode(token, convert_to_tensor=True, show_progress_bar=False)
            token_vecs[token] = [vec.tolist()]  # 保持列表格式，便于后续扩展
        # 保存
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(token_vecs, f, ensure_ascii=False, indent=2)
        print(f"向量表已保存至: {save_path}，共{len(token_list)}个token")
        return token_vecs

    def calculate_similarity(self, new_token):
        """
        计算新token与所有锚点词、描述词的相似度（不依赖字符串匹配）
        返回最大相似度及最终权重
        """
        # 1. 编码新token
        new_vec = self.model.encode(new_token, convert_to_tensor=True, show_progress_bar=False)
        new_vec = new_vec.unsqueeze(0)  # 转为[1, embed_dim]，便于批量计算

        # 2. 与锚点词计算相似度（批量计算，取最大值）
        if len(self.anchor_vecs) > 0:
            anchor_sims = util.cos_sim(new_vec, self.anchor_vecs).squeeze()  # [num_anchor_tokens]
            max_anchor_sim = anchor_sims.max().item()
            # 找到最相似的锚点词（用于调试）
            best_anchor = self.anchor_tokens[anchor_sims.argmax().item()] if len(anchor_sims) > 0 else None
        else:
            max_anchor_sim = 0.0
            best_anchor = None

        # 规则1：锚点词最大相似度>0.8 → 返回1.0
        if max_anchor_sim > 0.8:
            return {
                "token": new_token,
                "max_anchor_sim": max_anchor_sim,
                "best_anchor": best_anchor,
                "max_related_sim": 0.0,
                "best_related": None,
                "final_weight": 1.0
            }

        # 3. 与描述词计算相似度（批量计算，取最大值）
        if len(self.related_vecs) > 0:
            related_sims = util.cos_sim(new_vec, self.related_vecs).squeeze()  # [num_related_tokens]
            max_related_sim = related_sims.max().item()
            # 找到最相似的描述词（用于调试）
            best_related = self.related_tokens[related_sims.argmax().item()] if len(related_sims) > 0 else None
        else:
            max_related_sim = 0.0
            best_related = None

        # 规则2：描述词最大相似度>0.8 → 返回相似度×0.7
        if max_related_sim > 0.8:
            final_weight = max_related_sim * 0.7
        # 规则3：其他情况 → 返回最大相似度
        else:
            final_weight = max(max_anchor_sim, max_related_sim)

        return {
            "token": new_token,
            "max_anchor_sim": max_anchor_sim,
            "best_anchor": best_anchor,  # 最相似的锚点词（便于调试）
            "max_related_sim": max_related_sim,
            "best_related": best_related,  # 最相似的描述词（便于调试）
            "final_weight": final_weight
        }

# 使用示例
if __name__ == "__main__":
    # 1. 选择目标风格（以印象派为例）
    model = SentenceTransformer("/data2/gaodz/bge_m3")
    tokenizer = AutoTokenizer.from_pretrained("/data2/gaodz/bge_m3")
    anchor_tokens = ["Impressionism", "loose brushstrokes", "light shifts", "plein air", "color juxtaposition", "momentary effect",
                    "Realism", "accurate detail", "everyday life", "natural color", "objective representation", "fine brushwork",
                    "Cubism", "geometric fragmentation", "multiple perspectives", "overlapping planes", "abstract reconstruction", "collage technique",
                    "Abstract Expressionism", "non-representational", "emotional brushwork", "drip technique", "color field", "subconscious expression",
                    "Surrealism", "dreamlike imagery", "irrational juxtaposition", "subconscious symbols", "bizarre combinations", "fantastic scenes",
                    "Pop Art", "mass culture", "commercial imagery", "repetitive patterns", "vibrant colors", "everyday objects",
                    "Landscape", "natural scenery", "scenic views", "terrain depiction", "sky elements", "seasonal features",
                    "Portrait", "facial features", "human expression", "figure depiction", "character portrayal", "pose variation"]
    related_tokens = ["blurred outlines", "atmospheric haze", "natural light", "short strokes", "reflection capture", "outdoor scenery", "vibrant palette", "ephemeral light",
                     "lifelike figures", "social themes", "ordinary scenes", "precise perspective", "unidealized forms", "subtle tones", "textured surfaces", "domestic settings",
                     "fragmented forms", "angular shapes", "2D emphasis", "neutral palette", "synthetic forms", "distorted objects", "interlocking planes", "monochromatic tones",
                     "spontaneous marks", "thick impasto", "dynamic movement", "vibrant hues", "gestural strokes", "large scale", "emotional intensity", "unstructured composition",
                     "floating objects", "distorted perspective", "symbolic motifs",
                     "bold outlines", "screen printing", "advertising references", "celebrity portraits", "bright contrasts", "consumer goods", "media symbols", "flat colors",
                     "rolling hills", "water bodies", "vegetation details", "distant horizons", "atmospheric perspective", "weather effects", "light diffusion", "geographical forms",
                     "facial details", "emotional cues", "attire elements", "background context", "lighting on faces", "gestural hints", "eye contact", "facial proportions"]

    # 2. 初始化筛选器
    TokenSimilarityMatcher.precompute_vectors(
        token_list=anchor_tokens,
        model=model,
        save_path="anchor_vectors.json"
    )
    TokenSimilarityMatcher.precompute_vectors(
        token_list=related_tokens,
        model=model,
        save_path="related_vectors.json"
    )

    # 4. 加载向量表并计算相似度
    matcher = TokenSimilarityMatcher(
        anchor_path="anchor_vectors.json",
        related_path="related_vectors.json"
    )
    data = []
    with open("/data2/gaodz/Re-Align/COCO_clip_token.json","r",encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    # 3. 测试文本（包含风格相关词和无关词）
    new_data = []
    for item in tqdm(data):
        text_list = item["split_token"]

        res = []
        for i in range(len(text_list)):
            res.append(matcher.calculate_similarity(text_list[i])["final_weight"])
        res[0] = 0
        res[-1] = 0
        temp={
            "image": item["image"],
            "target_image": item["target_image"],
            "long_text": item["long_text"],
            "core_text": item["core_text"],
            "split_token": item["split_token"],
            "style_weight":res
        }
        new_data.append(temp)

    # 5. 输出结果（打印关键信息）
    with open("/data2/gaodz/Re-Align/COCO_clip_token_weight.json","w",encoding="utf-8") as f1:
        for line in new_data:
            f1.write(json.dumps(line, ensure_ascii=False) + "\n")
