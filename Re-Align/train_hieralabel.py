import os
os.environ["CUDA_VISIBLE_DEVICES"]='6'
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    AutoProcessor,
    PreTrainedTokenizer,
    AutoModelForImageTextToText
)
from peft import PeftModel, LoraConfig, get_peft_model
from PIL import Image
from torchvision import transforms

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_anyres_image


class HierarchicalContrastiveTrainer(Trainer):
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)


    # def _get_sentence_representation(self, model, input_ids, image_path=None):
    #     """获取 [CLS] 位置的表示向量，并支持图文联合编码"""
    #     device = next(model.parameters()).device
    #
    #     if image_path is not None:
    #         # 加载图像
    #         image = Image.open(image_path).convert("RGB")
    #         image_tensor = self.transform(image).unsqueeze(0).to(device)  # 添加 batch 维度并移动到 GPU
    #     else:
    #         image_tensor = None
    #
    #     outputs = model(
    #         input_ids=input_ids.unsqueeze(0),  # 添加 batch 维度
    #         images=image_tensor,
    #         output_hidden_states=True,
    #     )
    #     hidden_states = outputs.hidden_states[-1]
    #     return hidden_states[:, 0, :].squeeze(0)  # 去掉 batch 维度

    def compute_hierarchical_loss(self, model, inputs):
        """
        inputs: dict 包含以下张量
          - pixel_values:           [B, 3, H, W]
          - chosen_input_ids:       [B, L]
          - chosen_attention_mask:  [B, L]
          - strong_neg_input_ids:   [B, M, L]
          - strong_neg_attention_mask:[B, M, L]
          - weak_neg_input_ids:     [B, N, L]
          - weak_neg_attention_mask:[B, N, L]
        返回:
          - loss_per_sample: [B] 每个样本的损失
        """
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        # —— 1. 图像编码 ——
        pixel_values = inputs["pixel_values"].to(device)  # [B,3,H,W]
        vision_tower = model.get_vision_tower()
        vision_outs = vision_tower(pixel_values.to(dtype))

        patch_feats = vision_outs.to(dtype) # [B,P,Dv]
        mm_projector = model.get_model().mm_projector
        img_embeds = mm_projector(patch_feats)  # [B,P,Dl]
        img_repr = img_embeds.mean(dim=1)  # [B,Dl]
        img_norm = F.normalize(img_repr, p=2, dim=-1)  # [B,Dl]

        # —— 2. 文本编码（正样本） ——
        chosen_ids = inputs["chosen_input_ids"].to(device)  # [B,L]
        chosen_mask = inputs["chosen_attention_mask"].to(device)  # [B,L]
        txt_outs = model(
            input_ids=chosen_ids,
            attention_mask=chosen_mask,
            return_dict=True,
            output_hidden_states=True
        )
        # hidden_states: tuple(len=layers+1) of [B,L,Dl]
        last_hidden = txt_outs.hidden_states[-1].to(dtype)  # [B,L,Dl]
        text_chosen = last_hidden.mean(dim=1)  # [B,Dl]
        txt_chosen_norm = F.normalize(text_chosen, p=2, dim=-1)  # [B,Dl]

        # —— 3. 文本编码（负样本批量） ——
        # 强负
        strong_ids = inputs["strong_neg_input_ids"].to(device)  # [B,M,L]
        strong_mask = inputs["strong_neg_attention_mask"].to(device)  # [B,M,L]
        B, M, L = strong_ids.shape
        strong_ids = strong_ids.view(B * M, L)  # [B*M, L]
        strong_mask = strong_mask.view(B * M, L)  # [B*M, L]
        so_out = model(
            input_ids=strong_ids,
            attention_mask=strong_mask,
            return_dict=True,
            output_hidden_states=True
        )
        so_hidden = so_out.hidden_states[-1].view(B, M, L, -1).to(dtype)  # [B,M,L,Dl]
        so_repr = so_hidden.mean(dim=2)  # [B,M,Dl]
        so_norm = F.normalize(so_repr, p=2, dim=-1)  # [B,M,Dl]

        # 弱负
        weak_ids = inputs["weak_neg_input_ids"].to(device)  # [B,N,L]
        weak_mask = inputs["weak_neg_attention_mask"].to(device)  # [B,N,L]
        B, N, L = weak_ids.shape
        weak_ids = weak_ids.view(B * N, L)
        weak_mask = weak_mask.view(B * N, L)
        we_out = model(
            input_ids=weak_ids,
            attention_mask=weak_mask,
            return_dict=True,
            output_hidden_states=True
        )
        we_hidden = we_out.hidden_states[-1].view(B, N, L, -1).to(dtype)  # [B,N,L,Dl]
        we_repr = we_hidden.mean(dim=2)  # [B,N,Dl]
        we_norm = F.normalize(we_repr, p=2, dim=-1)  # [B,N,Dl]


        t = 0.02
        # 正样本 logit
        pos_logit = (img_norm * txt_chosen_norm).sum(dim=-1, keepdim=True) / t  # [B,1]
        # 强负 logits
        w_strong = 1.0
        strong_logits = torch.einsum("bd,bmd->bm", img_norm, so_norm) / t  # [B,M]
        strong_logits = strong_logits + torch.log(torch.full_like(strong_logits, w_strong)).to(device)
        # 弱负 logits
        w_weak = 1.0
        weak_logits = torch.einsum("bd,bnd->bn", img_norm, we_norm) / t  # [B,N]
        weak_logits = weak_logits + torch.log(torch.full_like(weak_logits, w_weak)).to(device)



        # 拼接
        logits = torch.cat([pos_logit, strong_logits, weak_logits], dim=1)  # [B,1+M+N]
        log_probs = F.log_softmax(logits, dim=1)  # [B,1+M+N]
        loss_per_sample = -log_probs[:, 0]  # [B]

        return loss_per_sample

    def training_step(self, model, inputs,num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_hierarchical_loss(model, inputs)
            # print('batch_loss_list:',loss)
            # loss = loss_per_sample.sum()
            loss = loss.mean()
        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)
        return loss.detach()



# def preprocess_multimodal(sources, data_args):
#     for source in sources:
#         for sentence in source:
#             if DEFAULT_IMAGE_TOKEN in sentence['content']:
#                 sentence['content'] = sentence['content'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
#                 sentence['content'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['content']
#                 sentence['content'] = sentence['content'].strip()
#                 if "mmtag" in "v1":
#                     sentence['content'] = sentence['content'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
#             replace_token = DEFAULT_IMAGE_TOKEN
#             if data_args.mm_use_im_start_end:
#                 replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
#             sentence["content"] = sentence["content"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
#     return sources
#
#
# def preprocess(sources, tokenizer, has_image=False):
#     conversations = []
#     for source in sources:
#         conv = "BEGINNING\n\n"
#         for sentence in source:
#             role = "user" if sentence["role"].lower() == "user" else "assistant"
#             conv += f"### {role}: {sentence['content']}\n"
#         conversations.append(conv)
#
#     if has_image:
#         input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
#     else:
#         input_ids = tokenizer(conversations, return_tensors="pt", padding="longest", truncation=True).input_ids
#
#     return torch.stack(input_ids)



class HierarchicalContrastiveDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,     # 例如 LLaVA 对应的 tokenizer
        image_folder: str,
        processor: AutoProcessor,           # AutoProcessor 也可以
        has_images: bool = True,
        max_length: int = 128,
    ):
        self.data = json.load(open(data_path, "r"))
        self.tokenizer = tokenizer
        self.image_folder = image_folder
        self.processor = processor
        self.has_images = has_images
        self.max_length = max_length

        # 如果想做额外的图像变换，也可以在 processor 之外加一层 transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        out: Dict[str, torch.Tensor] = {}

        # —— 1. 图像处理 ——
        if self.has_images and 'image' in item:
            img_path = os.path.join(self.image_folder, item['image'])

            # print("Loading image from:", img_path)

            image = Image.open(img_path).convert("RGB")
            if image.mode == "P":
                image = image.convert("RGBA")

                # 再转换为 RGB
            image = image.convert("RGB")
            # 如果 processor 支持 images 直接送进去：
            image_processor = self.processor
            tokenizer = self.tokenizer
            img_proc = image_processor(
                images=image,
                return_tensors="pt"
            )
            # 取出 pixel_values： [1, 3, H, W]，去掉 batch 维
            out['pixel_values'] = img_proc.pixel_values.squeeze(0)
        else:
            # 如果没有图片，用全 0 tensor 做占位
            out['pixel_values'] = torch.zeros(3, 224, 224)

        # —— 2. 文本处理：正样本 ——
        chosen = 'Please describe the entity in the image:'+item["chosen"]

        chosen_proc = tokenizer(
            text=chosen,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # squeeze 掉 batch 维 => [seq_len]
        out['chosen_input_ids'] = chosen_proc.input_ids.squeeze(0)
        out['chosen_attention_mask'] = chosen_proc.attention_mask.squeeze(0)

        # —— 3. 强负样本批量处理 ——
        strong_list = [item["rejected_strong"]] if isinstance(item["rejected_strong"], str) else item["rejected_strong"]
        strong_list = ['Please describe the entity in the image:' + s for s in strong_list]
        strong_proc = tokenizer(
            text=strong_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # [num_strong, seq_len]
        out['strong_neg_input_ids'] = strong_proc.input_ids
        out['strong_neg_attention_mask'] = strong_proc.attention_mask

        # —— 4. 弱负样本批量处理 ——
        weak_list = [item["rejected_weak"]] if isinstance(item["rejected_weak"], str) else item["rejected_weak"]
        weak_list = ['Please describe the entity in the image:' + w for w in weak_list]
        weak_proc = tokenizer(
            text=weak_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        out['weak_neg_input_ids'] = weak_proc.input_ids
        out['weak_neg_attention_mask'] = weak_proc.attention_mask

        return out
def custom_collate(batch):
    """
    将多个样本（dict）合并成一个 batch（dict of tensors）
    """
    from torch.utils.data._utils.collate import default_collate

    # 使用 defaultdict 来收集每个字段的数据
    collated_batch = {}

    # 获取第一个样本的所有键
    keys = batch[0].keys()

    for k in keys:
        collated_batch[k] = default_collate([sample[k] for sample in batch])

    return collated_batch
def add_gradient_hooks(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            def hook(grad, n=name):
                print(f"Grad of {n}: {grad.norm().item()}")
            param.register_hook(hook)

def train():
    @dataclass
    class Args:
        model_name_or_path: str = "/data2/gaodz/llava-v1.6-vicuna-7b"   # 基础模型 A
        lora_checkpoint: str = "/data2/gaodz/llava-vicuna-7b-rdpo-lora-1e-6-beta-0.1"             # LoRA 检查点 B
        data_path: str = "/data2/gaodz/Re-Align/re_align_data/Hierarchical_traindata.json"
        image_folder: str = "/data2/gaodz/Re-Align/re_align_data"
        output_dir: str = "/data2/gaodz/Re-Align/output/hierar_train_tem0.02_v1.6"
        num_train_epochs: int = 10
        per_device_train_batch_size: int = 2
        learning_rate: float = 2e-5
        save_steps: int = 100
        logging_steps: int = 10
        max_length: int = 256
        lora_r: int = 64
        lora_alpha: int = 128
        use_lora: bool = True

    args = Args()

    disable_torch_init()  # 禁用不必要的初始化

    # 加载预训练模型及其组件
    model_path = args.model_name_or_path
    model_name = get_model_name_from_path(model_path)

    # 使用 LLaVA 自定义的加载方式
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)

    # # 3. 如果需要，应用初始 LoRA 并训练（可选）
    # if args.use_lora:
    #     peft_config = LoraConfig(
    #         r=args.lora_r,
    #         lora_alpha=args.lora_alpha,
    #         target_modules=["q_proj","k_proj","v_proj","o_proj"],
    #         lora_dropout=0.1,
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #     )
    #     # 得到可训练的 LoRA 模型
    #     model = get_peft_model(model, peft_config)



    peft_model = PeftModel.from_pretrained(model, args.lora_checkpoint)
    # 合并 LoRA 权重到基础模型，并释放适配器
    peft_model.train()
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:  # 只冻结非 LoRA 参数
            param.requires_grad = False
        else:
            param.requires_grad = True  # 确保 LoRA 参数可训练

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


    dataset = HierarchicalContrastiveDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        image_folder=args.image_folder,
        processor=processor,
        has_images=True
    )

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=custom_collate
    )



    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=True,
        push_to_hub=False,
    )



    trainer = HierarchicalContrastiveTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=custom_collate
    )

    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params / 1e6:.2f}M || total params: {all_param / 1e6:.2f}M || trainable%: {100 * trainable_params / all_param:.2f}%")



    trainer.train()


    peft_model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    train()


