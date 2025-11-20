
import random


import re
import time
from openai import OpenAI
from typing import List, Dict
import os
from tqdm import tqdm
from PIL import Image
import base64
import json
from io import BytesIO



def ask_image_question_qwen(question):
    res = []
    skipped = []  # 记录跳过的条目
    for i in tqdm(range(len(question)), desc="Processing images"):
        q = question[i]['question']
        image_path = '/data2/gaodz/train2014/' + question[i]['image']

        try:
            with open(image_path, "rb") as image_file:
                img_data = image_file.read()
            img = Image.open(BytesIO(img_data))
            img.thumbnail((512, 512))  # 建议调整尺寸以适配API限制
            buffered = BytesIO()
            img.save(buffered, format=img.format or "JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            client = OpenAI(
                api_key="sk-6b09fcd63c8a426e8412307cb8e62c65",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

            que = f"""To give you some picture QA questions, you need to extract the key entities in the text. 
If the problem description is vague, such as \"Describe this picture\", extract the entity based on the picture. 
The returned result is a list of entities.
Example: Question:Where are the man and his dog located in this image? Result:[man,dog].
Question:{q}"""

            completion = client.chat.completions.create(
                model="qwen-vl-max-2025-04-08",
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                            },
                            {"type": "text", "text": que},
                        ],
                    },
                ],
            )
            print(completion.choices[0].message.content)
            ent = {
                'idx': question[i]['idx'],
                'question': q,
                'image': question[i]['image'],
                'entity': completion.choices[0].message.content
            }
            res.append(ent)

        except Exception as e:
            print(f"Skipped due to error: {e}")
            skipped.append({
                'idx': question[i]['idx'],
                'question': q,
                'image': question[i]['image'],
                'error': str(e)
            })
            time.sleep(1)  # 避免频繁请求导致封禁

    return res, skipped
if __name__ == '__main__':
    with open('/data2/gaodz/Re-Align/filter_pref_data.json', 'r', encoding='utf-8') as f:
        question = json.load(f)
    question = random.sample(question, 1000)
    res =  ask_image_question_qwen(question)
    with open('/data2/gaodz/Re-Align/filter_pref_data_entity1k.json', 'w', encoding='utf-8') as f1:
        json.dump(res, f1, ensure_ascii=False)