import os
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import torch
import transformers
from transformers import Trainer, GenerationConfig
from torchvision import transforms
from torch.utils.data import Dataset
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer,Qwen2VLForConditionalGeneration,AutoProcessor
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch.distributed as dist
from llava import conversation as conversation_lib
from llava.train.qwen_trainer import QWENrDPOTrainer
# Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "&lt;image&gt;"
DEFAULT_IM_START_TOKEN = "&lt;im_start&gt;"
DEFAULT_IM_END_TOKEN = "&lt;im_end&gt;"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-VL-7B")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-2)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_use_im_start_end: bool = field(default=True)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

@dataclass
class DataArguments:
    data_path: str = field(default="./preference_data/pref_data.json",
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default="")
    image_aspect_ratio: str = 'pad'
    image_grid_pinpoints: Optional[str] = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = 2e-5
    group_by_modality_length: bool = field(default=False) 
    output_dir: str = "./output/qwen2-vl-rdpo"
    project_name: str = "VLM-alignment"
    wandb_run_name: str = "qwen2-vl-rdpo"
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter for DPO loss"}
    )
    max_steps: int = field(
        default=10000,
        metadata={"help": "Maximum number of training steps"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Logging steps"}
    ) 
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint steps"}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Eval steps"}
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Gradient accumulation steps"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU for training"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Learning rate"}
    )
def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['content']:
                sentence['content'] = sentence['content'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['content'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['content']
                sentence['content'] = sentence['content'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['content'] = sentence['content'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["content"] = sentence["content"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources



def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['content']:
                sentence['content'] = sentence['content'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['content'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['content']
                sentence['content'] = sentence['content'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['content'] = sentence['content'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["content"] = sentence["content"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"user": conv.roles[0], "assistant": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["role"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["role"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["content"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                # 新增调试信息
                print("====== Tokenization Mismatch Debug ======")
                print(f"conversation: {repr(conversation)}")
                print(f"total_len (tokenizer分词后非pad token数): {total_len}")
                print(f"cur_len (手动累加): {cur_len}")
                print("每一轮内容及分词长度：")
                rounds = conversation.split(conv.sep2)
                for i, rou in enumerate(rounds):
                    print(f"  round {i}: {repr(rou)}")
                    if has_image:
                        print(f"    round_len: {len(tokenizer_image_token(rou, tokenizer))}")
                    else:
                        print(f"    round_len: {len(tokenizer(rou).input_ids)}")
                print("=========================================")
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    chosen: int = 1
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"user": conv.roles[0], "assistant": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    prompts = []
    for i, source in enumerate(sources):
        if roles[source[0]["role"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["role"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["content"])
            if j==0:
                prompts.append(conv.get_prompt() + conv.roles[1] + ": ")

        conversations.append(conv.get_prompt())

    # Tokenize conversations

    def pad_or_truncate(tensor: torch.Tensor, max_len: int, pad_id: int) -> torch.Tensor:
        """
        将输入 tensor 的最后一维截断或填充到长度 max_len，保持前面所有维度不变。
        """
        # 打印调试（训练时先开，确认 shape）


        seq_len = tensor.size(-1)
        if seq_len > max_len:
            out = tensor[..., :max_len]

            return out

        if seq_len < max_len:
            pad_len = max_len - seq_len
            # 构造 pad_shape，跟 tensor 除最后一维外的所有维度一致
            pad_shape = list(tensor.shape)
            pad_shape[-1] = pad_len
            pad = torch.full(pad_shape, pad_id, dtype=tensor.dtype, device=tensor.device)
            out = torch.cat([tensor, pad], dim=-1)

            return out

        # 恰好相等
        return tensor
    if has_image:
        raw_input_ids = [tokenizer_image_token(p, tokenizer, return_tensors="pt") for p in conversations]
        raw_prompt_ids = [tokenizer_image_token(p, tokenizer, return_tensors="pt") for p in prompts]
        input_ids = torch.stack([pad_or_truncate(x, tokenizer.model_max_length, tokenizer.pad_token_id)
                                 for x in raw_input_ids], dim=0)
        prompt_input_ids = torch.stack([pad_or_truncate(x, tokenizer.model_max_length, tokenizer.pad_token_id)
                                        for x in raw_prompt_ids], dim=0)

    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        prompt_input_ids = tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                # 新增调试信息
                print("====== Tokenization Mismatch Debug ======")
                print(f"conversation: {repr(conversation)}")
                print(f"total_len (tokenizer分词后非pad token数): {total_len}")
                print(f"cur_len (手动累加): {cur_len}")
                print("每一轮内容及分词长度：")
                rounds = conversation.split(conv.sep2)
                for i, rou in enumerate(rounds):
                    print(f"  round {i}: {repr(rou)}")
                    if has_image:
                        print(f"    round_len: {len(tokenizer_image_token(rou, tokenizer))}")
                    else:
                        print(f"    round_len: {len(tokenizer(rou).input_ids)}")
                print("=========================================")
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )


    if chosen == 1:
        return dict(chosen_input_ids=input_ids, chosen_labels=targets, prompt = prompts, prompt_input_ids = prompt_input_ids)
    else:
        return dict(rejected_input_ids=input_ids, rejected_labels=targets)


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['content']
        source[0]['content'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['content'] + source[1]['content'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['content'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)
def rank0_print(*args):
    if dist.get_rank() == 0:
        print(*args)

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['content'].split()) for conv in sample['chosen']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['content'].split()) for conv in sample['chosen'])
            cur_len = cur_len if 'images' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            retrieved_image_file = self.list_data_dict[i]["retrieved_image"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            retrieved_image = Image.open(os.path.join(image_folder, retrieved_image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.feature_extractor.image_mean))
                image = processor.feature_extractor(image, return_tensors='pt')['pixel_values'][0]

                retrieved_image = expand2square(retrieved_image, tuple(int(x*255) for x in processor.feature_extractor.image_mean))
                retrieved_image = processor.feature_extractor(retrieved_image, return_tensors='pt')['pixel_values'][0]

            else:
                image = processor.feature_extractor(image, return_tensors='pt')['pixel_values'][0]
                retrieved_image = processor.feature_extractor(retrieved_image, return_tensors='pt')['pixel_values'][0]

            sources = preprocess_multimodal(
                copy.deepcopy([e["chosen"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["chosen"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]),
            chosen=1
        )
        if isinstance(i, int):
            data_dict = dict(chosen_input_ids=data_dict["chosen_input_ids"][0],
                             chosen_labels=data_dict["chosen_labels"][0],
                             prompt_input_ids = data_dict["prompt_input_ids"][0],
                             prompt = data_dict["prompt"][0])


        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            rejected_sources = preprocess_multimodal(
                copy.deepcopy([e["rejected"] for e in sources]),
                self.data_args)
        else:
            rejected_sources = copy.deepcopy([e["rejected"] for e in sources])
        rejected_data_dict = preprocess(
            rejected_sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]),
            chosen=0)
        if isinstance(i, int):
            rejected_data_dict = dict(rejected_input_ids=rejected_data_dict["rejected_input_ids"][0],
                             rejected_labels=rejected_data_dict["rejected_labels"][0],
                             )
        data_dict.update(rejected_data_dict)
        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['images'] = image
            data_dict['retrieved_images'] = retrieved_image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['images'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['retrieved_images'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer  # 必须传入 tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 提取 chosen 分支的 input_ids 和 labels
        chosen_input_ids, chosen_labels = tuple([instance[key] for instance in instances]
                                              for key in ("chosen_input_ids", "chosen_labels"))
        prompt_input_ids = [instance["prompt_input_ids"] for instance in instances]

        # 对 prompt_input_ids 做 padding
        max_prompt_length = max([item.shape[0] for item in prompt_input_ids])
        prompt_input_ids = torch.nn.utils.rnn.pad_sequence(
            prompt_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        prompt_input_ids = prompt_input_ids[:, :max_prompt_length]

        # 对 chosen_input_ids 和 chosen_labels 做 padding
        chosen_input_ids = torch.nn.utils.rnn.pad_sequence(
            chosen_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        chosen_labels = torch.nn.utils.rnn.pad_sequence(
            chosen_labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )

        # 截断到模型最大长度
        max_len = self.tokenizer.model_max_length
        chosen_input_ids = chosen_input_ids[:, :max_len]
        chosen_labels = chosen_labels[:, :max_len]

        # 构建 chosen 分支的 batch 字典
        batch = dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            chosen_attention_mask=chosen_input_ids.ne(self.tokenizer.pad_token_id),
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_input_ids.ne(self.tokenizer.pad_token_id),
            prompt=[instance["prompt"] for instance in instances],
        )

        # 处理 rejected 分支
        rejected_input_ids, rejected_labels = tuple([instance[key] for instance in instances]
                                                  for key in ("rejected_input_ids", "rejected_labels"))
        rejected_input_ids = torch.nn.utils.rnn.pad_sequence(
            rejected_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        rejected_labels = torch.nn.utils.rnn.pad_sequence(
            rejected_labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )
        rejected_input_ids = rejected_input_ids[:, :max_len]
        rejected_labels = rejected_labels[:, :max_len]

        # 构建 rejected 分支的 batch 字典
        rejected_batch = dict(
            rejected_input_ids=rejected_input_ids,
            rejected_labels=rejected_labels,
            rejected_attention_mask=rejected_input_ids.ne(self.tokenizer.pad_token_id),
        )

        # 合并两个分支
        batch.update(rejected_batch)

        # 如果存在图像数据，则加入 batch
        if 'images' in instances[0]:
            images = [instance['images'] for instance in instances]
            retrieved_images = [instance['retrieved_images'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
                batch['retrieved_images'] = torch.stack(retrieved_images)
            else:
                batch['images'] = images
                batch['retrieved_images'] = retrieved_images

        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize wandb
    wandb.init(project=training_args.project_name, name=training_args.wandb_run_name)

    # Set up quantization config if needed
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["visual"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type
            )
        ))

    # Load model and tokenizer
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        **bnb_model_from_pretrained_args
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            # 默认使用qwen2_vl模板
            conversation_lib.default_conversation = conversation_lib.conv_templates["qwen2_vl"]

    # Load reference model
    model_ref = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        **bnb_model_from_pretrained_args
    )
    model_ref.config.use_cache = False
    
    # Freeze reference model
    for param in model_ref.parameters():
        param.requires_grad = False
    model_ref.eval()

    # Setup LoRA if enabled
    if training_args.lora_enable:
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        
        if training_args.bits in [4, 8]:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # Prepare dataset
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # Initialize trainer
    trainer = QWENrDPOTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        tokenizer=tokenizer,
        **data_module,
        beta=training_args.beta
    )

    # Start training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # Save final model
    trainer.save_state()
    if training_args.lora_enable:
        model.save_pretrained(training_args.output_dir)
    else:
        trainer.save_model()

if __name__ == "__main__":
    train() 