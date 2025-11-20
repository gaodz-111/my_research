import os
os.environ["CUDA_VISIBLE_DEVICES"]='6'
import argparse
import torch
import os
import sys
import json
from tqdm import tqdm


parent_dir = os.path.abspath("./llava")  # 当前 llava 所在路径
sys.path.append(parent_dir)

from constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from utils import disable_torch_init
from mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import DataLoader, Dataset

from PIL import Image
import math
IMAGE_TOKEN_INDEX = -200

model_path = "/data2/gaodz/llava_v1.6_sft_full"
# model_path = "/data2/gaodz/llava-v1.6-vicuna-7b"
model_base = None
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

from peft import PeftModel, PeftConfig


# model.load_adapter("/data2/gaodz/llava_v1.6_sft2.1_lora",adapter_name="default")
device = "cuda"
special_tokens = {
    "additional_special_tokens": [
        "<subproblem>", "</subproblem>",
        "<type>", "</type>",
        "<answer>", "</answer>",
        "<rethink>", "</rethink>",
        "<lastanswer>", "</lastanswer>",
        "[MASK]"
    ]
}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))
model.to(device)
data=[]
with open("/data2/gaodz/Re-Align/data_step6.json", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))
for item in tqdm(data):
    image_folder = "/data2/gaodz/train2014"
    image_file = "COCO_train2014_" + item["image"]
    image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
    image_sizes = image.size
    image_tensor = process_images([image], image_processor, model.config)[0]
    for j in item["sub problem"]:

        if j["question_type"] not in ["object", "attribute", "action", "description"]:
            continue  # 跳过非本函数处理类型
        question=j["question"]
        template=j["mask answer"]
        SYSTEM_PROMPT = f"""You are a helpful Visual Question Answering (VQA) assistant. 
        Give you a VQA question and answer template with [MASK], you need to replace [MASK] with the word [MASK] in the template to answer,
        question:USER:<image>\n{question}
        template:{template}\nASSISTANT:"""


        input_ids = tokenizer_image_token(SYSTEM_PROMPT, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(device)


        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=False,
                max_new_tokens=512,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(outputs)
        j["fill answer"] = outputs


with open("/data2/gaodz/Re-Align/data_step6_fill.json", "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")