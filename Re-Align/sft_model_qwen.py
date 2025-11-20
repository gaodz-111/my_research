import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import json
import traceback
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

from transformers import (
    TrainingArguments,
    Trainer
)
from modelscope import AutoModelForVision2Seq,AutoTokenizer
from peft import LoraConfig, get_peft_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IGNORE_INDEX = -100

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

class MultimodalSFTDataset(Dataset):
    def __init__(self, data_list, processor, image_folder,tokenizer):
        self.data_list = data_list
        self.processor = processor
        self.image_folder = image_folder
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]

        try:
            user_prompt = sample["conversations"][0]["value"]
            answer = sample["conversations"][1]["value"]
            user_prompt = user_prompt.replace("<image>", "")
            prompt = (
                f"<|im_start|>system\n"
                f"{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n"
                f"<|vision_start|><|image_pad|><|vision_end|> {user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            full_text = f"{prompt}{answer}<|im_end|>"

            image_path = os.path.join(self.image_folder, 'COCO_train2014_' + sample["image"])
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(image_path).convert('RGB')

            # processor 处理
            inputs = self.processor(
                text=full_text,
                images=image,
                return_tensors='pt'
            )

            # 计算 labels，mask prompt 部分
            labels = inputs["input_ids"].clone()
            prompt_ids = self.tokenizer(
                text=prompt,
                return_tensors='pt'
            )["input_ids"]

            labels[:, :prompt_ids.size(1)] = IGNORE_INDEX





            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "labels": labels.squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0)
            }

        except Exception:
            print(f"\n[Dataset Error] idx={idx}")
            traceback.print_exc()
            return None

class MultimodalCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            raise ValueError("Batch is empty!")

        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

model_path = "/data2/gaodz/Qwen2-VL-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(model_path, trust_remote_code=True).to(device)



# 添加特殊 tokens
special_tokens = {
    "additional_special_tokens": [
        "<subproblem>", "</subproblem>",
        "<type>", "</type>",
        "<answer>", "</answer>",
        "<rethink>", "</rethink>",
        "<lastanswer>", "</lastanswer>",
        "[MASK]",
    ]
}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

# LoRA 配置
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


with open("/data2/gaodz/Re-Align/data_step_sft_2.1.json") as f:
    data_list = [json.loads(line) for line in f]


from transformers import Qwen2VLImageProcessor
from transformers import ProcessorMixin

class Qwen2VLProcessor(ProcessorMixin):
    attributes = ["tokenizer", "image_processor"]
    tokenizer_class = "AutoTokenizer"
    image_processor_class = "Qwen2VLImageProcessor"

    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        image_inputs = self.image_processor(images, return_tensors=return_tensors)
        text_inputs["pixel_values"] = image_inputs["pixel_values"]
        return text_inputs
image_processor = Qwen2VLImageProcessor.from_pretrained(model_path)
processor = Qwen2VLProcessor(tokenizer, image_processor)

dataset = MultimodalSFTDataset(data_list, processor, "/data2/gaodz/train2014",tokenizer)
collator = MultimodalCollator(pad_token_id=tokenizer.pad_token_id)

training_args = TrainingArguments(
    output_dir="/data2/gaodz/qwen2_vl_sft_lora",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=100,
    learning_rate=5e-5,
    bf16=False,
    fp16=False,
    save_total_limit=2,
    report_to=[],
    remove_unused_columns=False,
    group_by_length=False,
    dataloader_drop_last=True,
)

# from transformers import TrainerCallback
#
#
# class GradientMonitorCallback(TrainerCallback):
#     def on_train_begin(self, args, state, control, **kwargs):
#         print("[Debug] Training started - Callback registered.")
#
#     def on_step_end(self, args, state, control, **kwargs):
#         print(f"[Debug] Step {state.global_step} - Callback triggered.")
#         model = kwargs["model"]
#         for name, param in model.named_parameters():
#             if param.grad is not None:
#                 grad_norm = param.grad.norm().item()
#                 print(f"Step {state.global_step} - {name}: Grad Norm = {grad_norm}")
#                 if torch.isnan(param.grad).any():
#                     print(f"[Error] NaN detected in gradients of {name}")




trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)



trainer.train()
trainer.save_model(training_args.output_dir)

