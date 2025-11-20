import os
os.environ["CUDA_VISIBLE_DEVICES"]='6'
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

from PIL import Image
import math
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/data2/gaodz/llava-v1.6-vicuna-7b"
model_base = None
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
model.to(device)
model.eval()



with open("/data2/gaodz/WikiArt/OpenDataLab___WikiArt/raw/train_txt/wikiart_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
image_root = "/data2/gaodz/WikiArt/OpenDataLab___WikiArt/raw/train_image/wikiart"


new_data = []
for item in tqdm(data):

    SYSTEM_PROMPT = """ You are a creative visual description generator. 

Given an image and the following tags: 
- Style: [style label]
- Theme: [image theme]

Write a concise, vivid, and coherent textual description of the image. 
The description should highlight:
- Visual details: colors, textures, lighting, composition, objects, and scene elements
- Style characteristics: mood, artistic techniques, or stylistic features

Constraints:
- Keep it concise but informative (do not over-explain)
- Avoid repetition or redundant phrases
- Focus on conveying both the image content and the style essence

Output the description in a single paragraph in English."""
    prompt=SYSTEM_PROMPT+f'\nUSER:<image>\n Style:{item["title"][0]} , Theme:{item["title"][1]}\nASSISTANT:'
    # image_folder = "/data2/gaodz/train2014"
    qs = "Describe this picture."

    image_folder = "/data2/gaodz/WikiArt/OpenDataLab___WikiArt/raw/train_image/wikiart"

    try:
        image = Image.open(os.path.join(image_folder, item["image"])).convert('RGB')
        image_sizes = image.size
        image_tensor = image_processor(image, return_tensors='pt')['pixel_values']
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(device)
        with torch.inference_mode():
            output_ids = model.generate(
                            input_ids,
                            images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                            image_sizes=image_sizes,
                            do_sample=False,
                            max_new_tokens=256,
                            use_cache=True)
        res=tokenizer.batch_decode(output_ids,skip_special_tokens=True)[0].strip()
    except OSError:
        continue

    new_data.append({
        "image": item["image"],
        "title": item["title"],
        "text": res
    })

with open("/data2/gaodz/WikiArt/OpenDataLab___WikiArt/raw/train_txt/image_text.json","w",encoding="utf-8") as f1:
    for line in new_data:
        f1.write(json.dumps(line, ensure_ascii=False) + '\n')