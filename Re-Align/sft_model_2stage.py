import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from torch.utils.data import Dataset
from PIL import Image

from llava.mm_utils import tokenizer_image_token
from torch.nn.utils.rnn import pad_sequence
import torch

from peft import LoraConfig, get_peft_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*Trainer.tokenizer is now deprecated.*")


from transformers import Trainer, TrainingArguments
from llava.model.builder import load_pretrained_model
import json
from torch.utils.data import DataLoader
from transformers import Trainer
import traceback
from llava.constants import IMAGE_TOKEN_INDEX

import gc
import torch
import random
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
SYSTEM_PROMPT = """You are a helpful Visual Question Answering (VQA) assistant. Given an image and a question, analyze step by step and answer in a structured format. 
Your response must strictly follow this format, including the tags and newlines:
<subproblem>
(one or more subproblems, each on a separate line, with a maximum of 3 subproblems)
</subproblem>
<type>
(the type of each subproblem, e.g., description, object, attribute, action, yesno, etc., each on a separate line)
</type>
<answer>
(the answer for each subproblem, using [MASK] templates if the type is description, object, attribute, or action; each on a separate line)
</answer>
<rethink>
(a reflection on each answer: for description/object/attribute/action, re-state with filled [MASK]; for others, briefly explain; each on a separate line)
</rethink>
<lastanswer>
(the final coherent, complete answer combining all subproblem answers and reflections)
</lastanswer>
Make sure your response is precise, grounded in the image and question, and formatted exactly as shown."""


def seed_everything(seed=42):


    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()


def split_two_stage_data(data_list):

    # 按长度对 answer 文本排序，优先短文本
    sorted_data = sorted(
        data_list,
        key=lambda x: (x["task_type"] != "single", len(x["conversations"][1]["value"]))
    )
    stage1_data = sorted_data[:1000]  # 2000条短文本，专注格式学习

    # 将剩下的数据划分为multi和single
    rest_data = sorted_data[1000:]
    multi_data = [item for item in rest_data if item.get("task_type") == "multi"]
    single_data = [item for item in rest_data if item.get("task_type") == "single"]
    singword_data = [item for item in rest_data if item.get("task_type") == "single word"]
    # 随机抽取1000条 single
    random.seed(42)
    selected_single = random.sample(single_data, min(1000, len(single_data)))

    # 阶段2数据 = multi全部 + 随机single
    stage2_data = multi_data + selected_single + singword_data

    return stage1_data, stage2_data

class MultimodalSFTDataset(Dataset):
    def __init__(self, data_list, tokenizer, image_processor, image_folder):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_folder = image_folder

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]

        try:
            user_prompt = sample["conversations"][0]["value"]
            answer = sample["conversations"][1]["value"]
            prompt = f"{SYSTEM_PROMPT}\nUSER: {user_prompt}\nASSISTANT:"
            full_text = f"{prompt} {answer} {self.tokenizer.eos_token}"

            input_ids = tokenizer_image_token(full_text, self.tokenizer, return_tensors='pt').squeeze(0)
            prompt_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt').squeeze(0)

            labels = input_ids.clone()
            labels[:len(prompt_ids)] = IGNORE_INDEX
            eos_token_id = self.tokenizer.eos_token_id
            if input_ids[-1].item() != eos_token_id:
                input_ids = torch.cat([input_ids, torch.tensor([eos_token_id], dtype=input_ids.dtype)])
                # 确保 labels 也跟上 eos
                labels = torch.cat([labels, torch.tensor([eos_token_id], dtype=labels.dtype)])
            image_path = os.path.join(self.image_folder, 'COCO_train2014_' + sample["image"])
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_processor(image, return_tensors='pt')['pixel_values'][0]
            # print(f"[DEBUG] idx={idx} IMAGE_TOKEN_INDEX count={(input_ids == IMAGE_TOKEN_INDEX).sum().item()}")
            if (labels != IGNORE_INDEX).sum() == 0:
                print(f"[Invalid Labels] idx={idx} | id={sample['id']} | question={user_prompt}")
                print(f"answer={answer}")
            eos_pos = (input_ids == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) == 0:
                print(f"[WARNING] No </s> token found in input_ids at idx={idx}")
            return {
                "input_ids": input_ids,
                "labels": labels,
                "images": image_tensor,
                "idx": idx,  # 加入索引，后续调试用
                "image_path": image_path,
                "prompt_len": len(prompt_ids),
                "full_text_len": len(input_ids),
                "user_prompt": user_prompt[:100],  # 截断展示
            }

        except Exception as e:
            print(f"\n[Dataset Error] idx={idx}")
            print(f"→ conversations: {sample.get('conversations')}")
            print(f"→ image name: {sample.get('image')}")
            traceback.print_exc()
            return None


class MultimodalCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        if len(batch) == 0:
            print("[Warning] This batch has only None samples, skipping.")
            return {
                "input_ids": torch.tensor([], dtype=torch.long),
                "labels": torch.tensor([], dtype=torch.long),
                "attention_mask": torch.tensor([], dtype=torch.bool),
                "images": torch.empty(0, 3, 336, 336)
            }



        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]
        images = [b["images"] for b in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        images = torch.stack(images).to(dtype=torch.bfloat16)
        attention_mask = input_ids != self.tokenizer.pad_token_id

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "images": images,
        }



from transformers import Trainer
import torch
import torch.nn as nn

class Stage1Trainer(Trainer):
    import torch.nn as nn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]  # (B, L_label)
        input_ids = inputs["input_ids"]  # (B, L_input)
        attention_mask = inputs.get("attention_mask", None)
        images = inputs.get("images", None)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            labels=None,
        )
        logits = outputs.logits  # (B, L_logits, V)

        # 取 logits 和 labels 最小长度
        seq_len = min(logits.size(1), labels.size(1))

        logits = logits[:, :seq_len, :]  # 裁剪 logits
        labels = labels[:, :seq_len]  # 裁剪 labels

        # shift，长度减1
        shift_logits = logits[..., :-1, :].contiguous()  # (B, seq_len-1, V)
        shift_labels = labels[..., 1:].contiguous()  # (B, seq_len-1)

        # 再次确认长度一致
        assert shift_logits.size(1) == shift_labels.size(1), \
            f"shift_logits length {shift_logits.size(1)} != shift_labels length {shift_labels.size(1)}"

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())

        # 设定图像token长度
        image_token_mask = (input_ids == IMAGE_TOKEN_INDEX)
        image_token_len = image_token_mask.sum(dim=1)
        loss_mask = torch.ones_like(loss)
        for i in range(loss_mask.size(0)):
            img_len = image_token_len[i].item()
            if loss_mask.size(1) > img_len - 1:
                loss_mask[i, :img_len - 1] = 0

        # tag权重
        tag_tokens = [
            "<subproblem>", "</subproblem>",
            "<type>", "</type>",
            "<answer>", "</answer>",
            "<rethink>", "</rethink>",
            "<lastanswer>", "</lastanswer>",
        ]
        tag_token_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in tag_tokens]
        tag_mask = torch.zeros_like(shift_labels, dtype=torch.float32)
        for tag_id in tag_token_ids:
            tag_mask += (shift_labels == tag_id).float()

        weights = 1.0 + tag_mask * 1.0
        weights = weights * loss_mask

        weighted_loss = (loss * weights).sum() / weights.sum().clamp(min=1.0)

        return (weighted_loss, outputs) if return_outputs else weighted_loss


model_path = "/data2/gaodz/llava-v1.6-vicuna-7b"
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_v1.6", device="cuda")
for param in model.get_model().vision_tower.parameters():
    param.requires_grad = False



# lora_config = LoraConfig(
#         task_type="CAUSAL_LM",
#         r=64,
#         lora_alpha=128,
#         lora_dropout=0.1,
#         target_modules=['v_proj', 'q_proj', 'k_proj', 'o_proj'],
#     )

# model = get_peft_model(model, lora_config)

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

for name, module in model.named_modules():
    if "lora" not in name.lower():  # 可选：保留 LoRA 参数精度不变
        module.to(dtype=torch.bfloat16)

# 同时确保 mm_projector 也是 bfloat16
model.get_model().mm_projector.to(dtype=torch.bfloat16)
model.to(device)
# model.print_trainable_parameters()
# 加载数据
with open("/data2/gaodz/Re-Align/data_step_sft_3.json") as f:
    data_list = [json.loads(line) for line in f]


stage1_data, stage2_data = split_two_stage_data(data_list)
print(f"Stage1 samples: {len(stage1_data)}, Stage2 samples: {len(stage2_data)}")

# 构建两个数据集对象
stage1_dataset = MultimodalSFTDataset(stage1_data, tokenizer, image_processor, "/data2/gaodz/train2014")
stage2_dataset = MultimodalSFTDataset(stage2_data, tokenizer, image_processor, "/data2/gaodz/train2014")
collator = MultimodalCollator(tokenizer)

training_args_stage1 = TrainingArguments(
    output_dir="/data2/gaodz/llava_v1.6_sft_stage1",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_steps=500,
    save_steps=1000,
    learning_rate=1e-5,
    bf16=True,
    save_total_limit=1,
    report_to=[],
    remove_unused_columns=False,
    group_by_length=False,
    dataloader_drop_last=True,
)

training_args_stage2 = TrainingArguments(
    output_dir="/data2/gaodz/llava_v1.6_sft_stage2",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_steps=200,
    save_steps=1000,
    learning_rate=1e-6,
    bf16=True,
    save_total_limit=1,
    report_to=[],
    remove_unused_columns=False,  # 必须关闭，否则 images 会被移除
    group_by_length=False,  # 避免 dynamic batching
    dataloader_drop_last=True,
)

# # ---------------------- 阶段1训练 -----------------------
trainer_stage1 = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args_stage1,
    train_dataset=stage1_dataset,
    data_collator=collator,
)
print("Start Stage 1 training")
trainer_stage1.train()
trainer_stage1.save_model(training_args_stage1.output_dir)


del trainer_stage1
del model
torch.cuda.empty_cache()
gc.collect()
# # ---------------------- 阶段2训练（加载阶段1权重） -----------------------
model_path="/data2/gaodz/llava_v1.6_sft_stage1"
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_v1.6", device="cuda")
for param in model.get_model().vision_tower.parameters():
    param.requires_grad = False
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

for name, module in model.named_modules():
    if "lora" not in name.lower():  # 可选：保留 LoRA 参数精度不变
        module.to(dtype=torch.bfloat16)

# 同时确保 mm_projector 也是 bfloat16
model.get_model().mm_projector.to(dtype=torch.bfloat16)
model.to(device)




trainer_stage2 = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args_stage2,
    train_dataset=stage2_dataset,
    data_collator=collator,
)
print("Start Stage 2 training")
trainer_stage2.train()
trainer_stage2.save_model(training_args_stage2.output_dir)