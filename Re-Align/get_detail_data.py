import os
os.environ["CUDA_VISIBLE_DEVICES"]='7'
import argparse
import torch
import os
import sys
import json
from tqdm import tqdm
import shortuuid

parent_dir = os.path.abspath("./llava")  # 当前 llava 所在路径
sys.path.append(parent_dir)

from constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from utils import disable_torch_init
from mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from PIL import Image
import math
from torch.utils.data import Dataset, DataLoader
import copy

device ="cuda"

model_path = "/data2/gaodz/llava-v1.6-vicuna-7b"
model_base = None
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
model.to(device)
model.eval()

datasets = [
    # {
    #     "data_json_path": "/data2/gaodz/Re-Align/hypernet_train_data_short_core.json",
    #     "image_root": "/data2/gaodz/OmniConsistency"
    # },
    {
        "data_json_path": "/data2/gaodz/sam_data/coco_with_detail_partial.json",
        "image_root": "/data2/gaodz/train2014"
    },
    {
        "data_json_path": "/data2/gaodz/sam_data/wikiart_with_detail_partial.json",
        "image_root": "/data2/gaodz/WikiArt/OpenDataLab___WikiArt/raw/train_image/wikiart"
    },
    # {
    #     "data_json_path": "/data2/gaodz/sharegpt4v/sharegpt4v_coco.json",
    #     "image_root": "/data2/gaodz/coco2017/PAI/COCO2017"
    # }
]
data=[]
# 3️⃣ 逐个读取并添加
for ds in datasets:
    json_path = ds["data_json_path"]


    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))  # 假设是标准 JSON 数组或对象


print(f"✅ Total loaded samples: {len(data)}")




class ImageDescriptionDataset(Dataset):
    def __init__(self, data, system_prompt):
        self.data = data

        self.system_prompt = system_prompt
        # 预检查系统提示是否为字符串
        assert isinstance(system_prompt, str), "System prompt must be a string"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            qs = item["long_text"]
            qs = qs.replace("<image>", "")
            # 准备prompt并确保是字符串
            prompt = self.system_prompt + f'\nUSER:{qs}\nASSISTANT:'
            if not isinstance(prompt, str):
                prompt = str(prompt)  # 强制转换为字符串
            if not prompt.strip():
                raise ValueError(f"Empty prompt generated for item {idx}")
        except Exception as e:
            print(f"Error generating prompt for item {idx}: {e}")
            # 生成默认prompt作为 fallback
            prompt = f"Describe this picture.\nUSER:<image>\nASSISTANT:"

        return {
            'item': item,
            'prompt': prompt,
        }


# 自定义collate函数处理批次数据
def collate_fn(batch):
    # 处理文本：过滤并修复无效prompt
    valid_items = []
    valid_prompts = []
    valid_images = []
    valid_image_sizes = []

    for item in batch:
        prompt = item['prompt']
        # 严格检查prompt类型
        valid_prompts.append(prompt)
        valid_items.append(item['item'])

    # 处理文本
    # input_ids_list = []  # 存储每个prompt的input_ids（单个tensor）
    # for prompt in valid_prompts:

    #     # 传入单个字符串给 tokenizer_image_token（而非列表）
    #     single_result = tokenizer_image_token(
    #         prompt=prompt,  # 重点：这里是单个字符串，不是列表
    #         tokenizer=tokenizer,
    #         image_token_index=IMAGE_TOKEN_INDEX,
    #         return_tensors='pt'  # 每个结果是 (1, seq_len) 的tensor
    #     ).squeeze(0)
    #     # 若 tokenizer_image_token 返回字典（如含 input_ids/attention_mask），需取 input_ids
    #     # （根据你的函数实现调整，若直接返回input_ids则无需此步）

    #     input_ids_list.append(single_result)
    input_ids_list = tokenizer_image_token(valid_prompts, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                                           return_tensors='pt')
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)

    return {
        'items': valid_items,
        'input_ids': input_ids
    }


def main():
    # 配置参数
    BATCH_SIZE = 64  # 根据GPU内存调整
    OUTPUT_FILE = "/data2/gaodz/data_with_attribute.json"

    # 系统提示（确保是字符串）
    SYSTEM_PROMPT = """You are an assistant that extracts semantic visual attributes from text.

Your task: Read the text carefully and extract all attributes that belong to these six categories:
["color", "shape", "material", "object", "scene", "emotion"]

Rules:
- Each category should be a list of strings.
- Include only attributes that are clearly mentioned or strongly implied.
- If a category has no attributes, use an empty list [].
- Do not include any explanations, reasoning, or text outside the JSON.

Return the final result in *pure JSON* only, following exactly this structure:

{
  "color": [],
  "shape": [],
  "material": [],
  "object": [],
  "scene": [],
  "emotion": []
}

Now extract the attributes from this text:"""

    # 加载数据
    print("Loading data...")
    main_data = data

    # 创建数据集和数据加载器
    dataset = ImageDescriptionDataset(main_data, SYSTEM_PROMPT)
    # 暂时将num_workers设为0便于调试（找到问题后可改回）
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # 先设为0，解决后再根据CPU调整
    )

    # 处理批次数据
    new_data = []
    print("Processing batches...")
    for batch in tqdm(dataloader, total=len(dataloader)):
        with torch.inference_mode():
            output_ids = model.generate(
                batch['input_ids'],
                images=None,
                do_sample=False,
                max_new_tokens=256,
                use_cache=True
            )

        # 解码输出
        results = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        results = [res.strip() for res in results]

        # 保存结果
        for item, res in zip(batch['items'], results):
            item_t = copy.deepcopy(item)
            item_t["detail"] = res
            new_data.append(item_t)

    # 保存最终结果
    print("Saving results...")
    with open(OUTPUT_FILE, "w", encoding='utf-8') as f1:
        for item in new_data:
            f1.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("Processing complete!")


if __name__ == "__main__":
    main()
