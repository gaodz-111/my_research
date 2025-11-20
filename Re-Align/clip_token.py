import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from diffusers import StableDiffusionPipeline
import torch
import json

def load_sd_clip_components(model_name="/data2/gaodz/stable-diffusion-2-1-base"):
    """
    加载Stable Diffusion模型并提取其中的CLIP分词器和文本编码器

    参数:
        model_name: Stable Diffusion模型名称

    返回:
        tokenizer: CLIP分词器
        text_encoder: CLIP文本编码器
    """
    try:
        # 加载Stable Diffusion模型（仅需文本相关组件）
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None  # 禁用安全检查器以简化示例
        )

        # 提取CLIP相关组件
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder

        # 移动到可用设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_encoder = text_encoder.to(device)

        print(f"成功从{model_name}中提取CLIP组件")
        print(f"使用设备: {device}")

        return tokenizer, text_encoder
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None, None


def get_sd_clip_tokenization(tokenizer, text, return_tokens=False):
    """
    使用Stable Diffusion中的CLIP分词器对文本进行分词
    """
    if not tokenizer:
        print("请先加载有效的分词器")
        return None

    # 分词处理（与原生CLIP处理方式一致）
    inputs = tokenizer(
        text,
        padding=False,
        truncation=False,
        return_tensors="pt"
    )

    result = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask
    }

    # 返回分词的文本表示
    if return_tokens:
        tokens = [tokenizer.decode([id]) for id in inputs.input_ids[0]]

        result["tokens"] = tokens

    return result


if __name__ == "__main__":
    # 从Stable Diffusion中加载CLIP组件
    tokenizer, text_encoder = load_sd_clip_components()

    if tokenizer:
        data = []
        # 测试文本
        with open("/data2/gaodz/Re-Align/COCO_short_core_1.json","r",encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        new_data=[]
        # 处理并显示分词结果
        for item in data:

            token_result = get_sd_clip_tokenization(tokenizer, item["long_text"], return_tokens=True)
            temp={
                "image": item["image"],
                "target_image": item["target_image"],
                "long_text": item["long_text"],
                "core_text": item["core_text"],
                "split_token":token_result["tokens"]
            }
            new_data.append(temp)
        with open("/data2/gaodz/Re-Align/COCO_clip_token.json","w",encoding="utf-8") as f1:
            for l in new_data:
                f1.write(json.dumps(l,ensure_ascii=False)+'\n')