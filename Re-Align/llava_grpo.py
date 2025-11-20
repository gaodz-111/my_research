import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
import copy
from trl import GRPOTrainer
import torch
import transformers
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import re

from trl import GRPOConfig
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from PIL import Image
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_anyres_image
from llava.model import *
from transformers import CLIPImageProcessor
from torch.nn.utils.rnn import pad_sequence
from trl import GRPOTrainer
from trl.trainer.grpo_trainer import shuffle_tensor_dict
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path
from transformers import default_data_collator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







def build_model_and_tokenizer(model_path, model_base=None, device="cuda", use_flash_attn=False):
    """
    使用 LLaVA 标准方式加载模型和 tokenizer。
    """
    # 禁用 torch 初始化以加快加载速度
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)

    #  使用标准方法加载模型和 tokenizer
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        device_map=device,
        device=device,
        use_flash_attn=use_flash_attn
    )

    return tokenizer, model, image_processor



class CustomGRPOTrainer(GRPOTrainer):
    def __init__(
            self,
            model,
            tokenizer,  # 确保从外部传入
            image_processor,  # 确保从外部传入
            *args,
            **kwargs
    ):
        # 保存 tokenizer 和 image_processor 为实例变量
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        # 调用父类构造函数时不要传入 tokenizer/image_processor
        super().__init__(model=model, *args, **kwargs)

    def _prepare_inputs(self, inputs):
        if isinstance(inputs, list):
            batch_dict = self.data_collator(inputs)
        elif isinstance(inputs, dict):
            batch_dict = inputs
        else:
            raise ValueError("Unsupported input format for _prepare_inputs")

        # 与原始逻辑一致
        generation_batch = self._generate_and_score_completions(batch_dict)

        shufflable_fields = ["prompt_ids", "prompt_mask", "completion_ids", "completion_mask", "images", "rewards"]
        shufflable_batch = {
            key: value.to(device=self.args.device)
            for key, value in generation_batch.items()
            if key in shufflable_fields and isinstance(value, torch.Tensor) and value.layout == torch.strided
        }

        # 修改点：验证张量形状一致性
        batch_sizes = []
        for key, value in shufflable_batch.items():
            if value is not None and isinstance(value, torch.Tensor):
                batch_sizes.append(value.shape[0])

        if len(set(batch_sizes)) > 1:
            error_msg = "All tensors must have the same batch size for shuffling. "
            error_msg += "Found sizes: " + ", ".join(
                f"{k}={v.shape[0]}" for k, v in shufflable_batch.items() if v is not None)
            raise ValueError(error_msg)

        shuffled_batch = shuffle_tensor_dict(shufflable_batch)

        raw_data_list = generation_batch.get("raw_data", [])
        if raw_data_list:
            shuffled_batch["raw_data"] = raw_data_list

        return shuffled_batch

    def _generate_and_score_completions(self, generation_batch, return_logits=False):
        if isinstance(generation_batch, dict):
            batch_dict = generation_batch
        elif isinstance(generation_batch, list):
            batch_dict = generation_batch[0]
        else:
            raise TypeError(f"Unsupported generation_batch type: {type(generation_batch)}")

        input_ids = batch_dict.get("prompt_ids")
        images = batch_dict.get("images")

        # 确保字段存在
        if input_ids is None:
            raise ValueError("`prompt_input_ids` missing from batch.")
        if images is None:
            raise ValueError("`images` missing from batch.")

        # 确保是 Tensor 类型
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, device=self.accelerator.device)

        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images, device=self.accelerator.device)

        # 确保张量维度正确
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # 添加 batch 维度 → [1, L]
        if images.dim() == 3:
            images = images.unsqueeze(0)  # 添加 batch 维度 → [1, C, H, W]
        if images.dtype != torch.bfloat16:
            images = images.to(dtype=torch.bfloat16)
        # 保持原始设备
        input_ids = input_ids.to(device=self.accelerator.device)
        images = images.to(device=self.accelerator.device)

        # 修改点1：获取batch size和生成数量
        batch_size = input_ids.shape[0]
        num_generations = self.args.num_generations

        # 修改点2：扩展输入张量和图像张量
        expanded_input_ids = input_ids.repeat(num_generations, 1)  # [B*N, L]
        expanded_images = images.repeat(num_generations, 1, 1, 1)  # [B*N, C, H, W]

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.raw_data = batch_dict.get("raw_data", None)

        # 修改点3：使用扩展后的输入进行生成
        prompt_completion_ids = unwrapped_model.base_model.generate(
            inputs=expanded_input_ids,  # 使用扩展后的输入
            images=expanded_images,  # 使用扩展后的图像
            max_new_tokens=self.args.max_completion_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1  # 每个样本生成1个结果（总共有batch_size*num_generations个结果）
        )

        # generated = prompt_completion_ids[:, expanded_input_ids.shape[1]:]

        # 修改点4：正确处理生成结果
        completions = []
        for i in range(prompt_completion_ids.shape[0]):
            completions.append({
                "content": self.tokenizer.decode(
                    prompt_completion_ids[i],
                    skip_special_tokens=False
                ).strip()
            })

        # 修改点5：优化调试信息输出
        raw_data_list = batch_dict.get("raw_data", [])
        if not isinstance(raw_data_list, list):
            raw_data_list = [raw_data_list] * batch_size

        # 为每个原始样本生成多个副本以匹配生成数量
        expanded_raw_data = []
        for raw in raw_data_list:
            expanded_raw_data.extend([raw] * num_generations)

        if raw_data_list:
            first_raw = raw_data_list[0]
            if isinstance(first_raw, dict):
                question = first_raw["conversations"][0]["value"]
                if isinstance(question, (list, tuple)):
                    question = question[0]
                print(f"Raw Question:\n{question}")
                print("-" * 40)

        # 使用扩展后的原始数据
        rewards = combined_reward(completions=completions, raw_data=expanded_raw_data)

        rewards = [r if r is not None else 0.0 for r in rewards]
        print("rewards:",rewards)
        return {
            "prompt_ids": expanded_input_ids,  # 返回扩展后的输入
            "prompt_mask": (expanded_input_ids != self.tokenizer.pad_token_id),
            "completion_ids": prompt_completion_ids,
            "completion_mask": (prompt_completion_ids != self.tokenizer.pad_token_id),
            "images": expanded_images,  # 返回扩展后的图像
            "raw_data": expanded_raw_data,  # 返回扩展后的原始数据
            "rewards": rewards
        }

    def _get_per_token_logps(
            self,
            model,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            images: Optional[torch.FloatTensor] = None,
    ):
        """
        获取每个 token 的 logp 值，支持传入 images 参数。
        """
        if input_ids is None:
            raise ValueError("`input_ids` is None in _get_grpo_completions")

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            output_hidden_states=True,
            return_dict=True,
        )

        logits = outputs.logits
        per_token_logps = torch.log_softmax(logits, dim=-1)

        labels = input_ids[:, 1:].clone()
        per_token_logps = per_token_logps[:, :-1]
        token_logps = torch.gather(per_token_logps, -1, labels.unsqueeze(-1)).squeeze(-1)

        return token_logps

    # def _get_grpo_completions(self, model, input_ids, attention_mask, images=None):
    #     """
    #     使用 tokenizer 解码生成内容，并包装成 Dict[str, str] 格式供奖励函数使用。
    #     """
    #     # print("\nDebugging _get_grpo_completions inputs:")
    #     # print(f"input_ids type: {type(input_ids)}")
    #     # if isinstance(input_ids, torch.Tensor):
    #     #     print(f"input_ids shape: {input_ids.shape}")
    #     #     print(f"input_ids device: {input_ids.device}")
    #     #     print(f"input_ids dtype: {input_ids.dtype}")
    #     # else:
    #     #     print("input_ids is not a tensor")
    #     #
    #     # print(f"\nimages type: {type(images)}")
    #     # if isinstance(images, torch.Tensor):
    #     #     print(f"images shape: {images.shape}")
    #     #     print(f"images device: {images.device}")
    #     #     print(f"images dtype: {images.dtype}")
    #     # else:
    #     #     print("images is None or not a tensor")
    #
    #
    #     #  使用 unwrap 后的模型中的 tokenizer
    #     unwrapped_model = self.accelerator.unwrap_model(model)
    #     base_model = unwrapped_model.base_model
    #
    #     completions_text = tokenizer.batch_decode(input_ids)
    #     print('answer:',completions_text)
    #     # 包装成 reward_func 可接受的格式
    #     completions = [{"content": text} for text in completions_text]
    #     # print("\nDebugging completions structure:")
    #     # for idx, comp in enumerate(completions):
    #     #     print(f"Completion {idx}: {comp}")
    #     return completions

    def _compute_loss(self, model, inputs):
        prompt_ids = inputs.get("prompt_ids")
        completion_ids = inputs.get("completion_ids")
        completion_mask = inputs.get("completion_mask")
        images = inputs.get("images")
        # 修改点1：强化rewards获取逻辑
        rewards = inputs.get("rewards")

        if completion_ids is None or completion_mask is None:
            raise ValueError("`completion_ids` or `completion_mask` is None.")

        # 确保 completion_mask 长度一致
        if completion_mask.shape[1] != completion_ids.shape[1]:
            completion_mask = completion_mask[:, :completion_ids.shape[1]]

        # 修改点2：强化rewards有效性检查
        batch_size = prompt_ids.shape[0] if prompt_ids is not None else 1
        total_samples = completion_ids.shape[
            0] if completion_ids is not None else batch_size * self.args.num_generations

        # 修改点3：确保rewards始终是有效列表
        if rewards is None:
            rewards = [0.0] * total_samples
        elif isinstance(rewards, torch.Tensor):
            rewards = rewards.tolist()

        # 修改点4：确保每个元素都是有效数字
        rewards = [r if r is not None and isinstance(r, (int, float)) else 0.0 for r in rewards]

        # 修改点5：显式计算num_generations
        if prompt_ids is None:
            num_generations = self.args.num_generations
        else:
            num_generations = completion_ids.shape[0] // batch_size if completion_ids.shape[
                                                                           0] > 0 else self.args.num_generations

        # 修改点6：添加安全形状验证
        try:
            completion_ids = completion_ids.view(batch_size, num_generations, -1)
            completion_mask = completion_mask.view(batch_size, num_generations, -1)
        except Exception as e:
            print(f"Shape error with: {e}")
            print(f"batch_size={batch_size}, num_generations={num_generations}")
            print(f"completion_ids shape={completion_ids.shape if completion_ids is not None else 'None'}")
            print(f"completion_mask shape={completion_mask.shape if completion_mask is not None else 'None'}")
            raise

        # 修改点7：确保rewards_tensor形状匹配
        try:
            rewards_tensor = torch.tensor(
                rewards[:batch_size * num_generations],  # 限制最大长度
                device=completion_ids.device,
                dtype=torch.float32
            ).view(batch_size, num_generations)
        except Exception as e:
            print(f"Reward tensor creation error: {e}")
            rewards_tensor = torch.zeros((batch_size, num_generations), device=completion_ids.device,
                                         dtype=torch.float32)

        # 修改点8：安全处理logps计算
        per_token_logps = []
        for i in range(num_generations):
            try:
                current_ids = completion_ids[:, i]
                current_mask = completion_mask[:, i]

                # 添加空序列检查
                if current_ids.numel() == 0 or current_mask.numel() == 0:
                    print(f"Skipping empty input in generation {i}")
                    continue

                logps = self._get_per_token_logps(model, current_ids, current_mask, images=images)
                per_token_logps.append(logps.unsqueeze(1))
            except Exception as e:
                print(f"Error in generation {i}: {str(e)}")
                continue

        if not per_token_logps:
            print("Warning: All completions were empty, using zero loss")
            return torch.tensor(0.0, device=prompt_ids.device if prompt_ids is not None else device, requires_grad=True)

        # 修改点9：安全拼接张量
        try:
            per_token_logps = torch.cat(per_token_logps, dim=1)
        except Exception as e:
            print(f"Tensor concatenation error: {e}")
            return torch.tensor(0.0, device=prompt_ids.device if prompt_ids is not None else device, requires_grad=True)

        # 修改点10：安全的mask应用
        try:
            chosen_logps = (per_token_logps * completion_mask[:, :, :per_token_logps.shape[2]]).sum(-1)
        except Exception as e:
            print(f"Mask application error: {e}")
            chosen_logps = per_token_logps.sum(-1)

        # 修改点11：安全的权重计算
        try:
            weights = torch.softmax(rewards_tensor, dim=1)
            weighted_logps = (weights * chosen_logps).sum(dim=1)
        except Exception as e:
            print(f"Weight calculation error: {e}")
            weighted_logps = chosen_logps.mean(dim=1)

        loss = -weighted_logps.mean()
        return loss


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion contains specific XML-like tags."""
    tags = ["subproblem", "type", "answer", "rethink", "lastanswer"]
    rewards_list = []

    for completion in completions:
        # 检查是否为空或结构错误
        if not isinstance(completion, dict) or "content" not in completion:
            print("Invalid completion structure:", completion)
            rewards_list.append(0.0)
            continue

        content = completion["content"]
        score = 0.0

        for tag in tags:
            pattern = fr"<{tag}>\s*.*?\s*</{tag}>"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                score += 0.2

        rewards_list.append(score)

    return rewards_list




def compute_mask_match(completions, **kwargs):
    """
    只处理 question_type in ["object", "attribute", "action", "description"] 的子问题。
    使用已有的 answer 和 mask answer 字段进行逐词对比。
    返回：该类型所有子问题的匹配率平均值。
    """
    rewards = []

    for completion in completions:
        content = completion["content"]

        # 提取 <type> 中的问题类型
        type_match = re.search(r"<type>\s*$$(.*?)$$\s*</type>", content, re.DOTALL)
        if not type_match:
            rewards.append(0.0)
            continue
        types_needed = [t.strip() for t in type_match.group(1).split(",")]

        # 提取 <rethink> 中模型生成的填词列表
        rethink_match = re.search(r"<rethink>\s*$$(.*?)$$\s*</rethink>", content, re.DOTALL)
        if not rethink_match:
            rewards.append(0.0)
            continue
        list_b = [w.strip() for w in rethink_match.group(1).split(",")]

        # 获取原始数据
        raw_data = kwargs.get("raw_data", None)
        if not raw_data:
            raise ValueError("Missing raw_data in kwargs")

        sub_problems = raw_data.get("sub problem", [])

        # 只保留需要类型的参考答案
        valid_count = 0
        correct_sum = 0.0

        for q_type in types_needed:
            if q_type not in ["object", "attribute", "action", "description"]:
                continue  # 跳过非本函数处理类型

            for sp in sub_problems:
                if sp.get("question_type") == q_type:
                    answer = sp.get("answer", "").strip()
                    mask_answer = sp.get("mask answer", "").strip()

                    # 直接从 answer 中提取真实词（已预处理好）
                    answer_words = answer.split()
                    mask_words = mask_answer.split()

                    # 找出 mask_answer 中的 [MASK] 位置
                    mask_positions = [i for i, word in enumerate(mask_words) if word == "[MASK]"]

                    # 提取对应的真实词和模型生成词
                    refs = [answer_words[i] for i in mask_positions]
                    preds = [list_b.pop(0) for _ in mask_positions]  # 假设 list_b 按顺序排列

                    # 计算正确数
                    for pred, ref in zip(preds, refs):
                        if pred.lower() == ref.lower():
                            correct_sum += 1.0
                        valid_count += 1

        score = correct_sum / (valid_count + 1e-5) if valid_count else 0.0
        rewards.append(score)

    return rewards




emb_model = SentenceTransformer('/data2/gaodz/bge_m3')

def compute_consistency(completions, **kwargs):
    """
    只处理 question_type in ["yesno", "count", "location"] 的子问题。
    返回每个子问题的语义相似度，不匹配的问题返回 0.0。
    """

    rewards = []

    for completion in completions:
        content = completion["content"]

        # 提取 type 中的问题类型
        type_match = re.search(r"<type>\s*$$(.*?)$$\s*</type>", content, re.DOTALL)
        if not type_match:
            rewards.append(0.0)
            continue
        types_needed = [t.strip() for t in type_match.group(1).split(",")]

        # 提取 rethink 中的回答
        rethink_match = re.search(r"<rethink>\s*$$(.*?)$$\s*</rethink>", content, re.DOTALL)
        if not rethink_match:
            rewards.append(0.0)
            continue
        list_b = [w.strip() for w in rethink_match.group(1).split(",")]

        # 获取原始数据
        raw_data = kwargs.get("raw_data", None)
        if not raw_data:
            raise ValueError("Missing raw_data in kwargs")

        sub_problems = raw_data.get("sub problem", [])

        # 只保留需要类型的参考答案
        filtered_answers = []
        valid_count = 0
        total_score = 0.0

        for q_type in types_needed:
            if q_type not in ["yesno", "count", "location"]:
                continue  # 跳过非本函数处理类型

            for sp in sub_problems:
                if sp.get("question_type") == q_type:
                    answer = sp.get("rethink", "")
                    original_answer = answer.split("reflection:")[-1].strip()
                    filtered_answers.append(original_answer)

        # 计算语义相似度
        for pred, ref in zip(list_b, filtered_answers):
            embedding1 = emb_model.encode(pred, convert_to_tensor=True)
            embedding2 = emb_model.encode(ref, convert_to_tensor=True)
            cosine_score = util.cos_sim(embedding1, embedding2).item()
            total_score += cosine_score
            valid_count += 1

        score = total_score / (valid_count + 1e-5) if valid_count else 0.0
        rewards.append(score)

    return rewards



def accuracy_reward(completions, **kwargs):
    """
    提取 <lastanswer> 中的回答并与原始数据中的真实答案比较，计算一致性评分。
    返回：匹配为 1.0，不匹配为 0.0
    """
    rewards = []

    # 获取原始数据
    raw_data_list = kwargs.get("raw_data", None)
    if not isinstance(raw_data_list, list):
        raw_data_list = [raw_data_list]  # 统一为列表格式

    for idx, completion in enumerate(completions):
        if isinstance(completion, dict):
            content = completion.get("content", "")
        elif isinstance(completion, (list, tuple)) and len(completion) > 0:
            content = completion[0].get("content", "")
        else:
            content = ""

        if not content:
            rewards.append(0.0)
            continue

        # 提取模型输出的 lastanswer
        last_answer_match = re.search(r"<lastanswer>\s*(.*?)\s*</lastanswer>", content, re.DOTALL)
        if not last_answer_match:
            rewards.append(0.0)
            continue

        pred_answer = last_answer_match.group(1).strip()

        # 获取对应的真实答案
        try:
            raw_data = raw_data_list[idx]
            sub_problems = raw_data.get("sub problem", [])
            if not sub_problems:
                rewards.append(0.0)
                continue

            # 假设只比较第一个问题的答案
            true_answer = sub_problems[0].get("answer", "").strip()
            # 提取 "Original Answer:" 后面的部分
            true_answer = true_answer.split("Original Answer:")[-1].strip()

        except (IndexError, TypeError, KeyError):
            rewards.append(0.0)
            continue

        pred_embedding = emb_model.encode(pred_answer, convert_to_tensor=True)
        true_embedding = emb_model.encode(true_answer, convert_to_tensor=True)

        # 计算余弦相似度
        cosine_score = util.cos_sim(pred_embedding, true_embedding).item()

        rewards.append(cosine_score)
    return rewards

def combined_reward(completions, **kwargs):
    # 调用各个奖励函数
    r1 = format_reward(completions, **kwargs)
    r2 = accuracy_reward(completions, **kwargs)
    r3 = compute_mask_match(completions, **kwargs)
    r4 = compute_consistency(completions, **kwargs)
    # 设置权重
    w1 = 0.4
    w2 = 0.3
    w3 = 1-w1-w2

    # 返回加权和
    return [w1 * r1[i] + w2 * r2[i] + w3 * (r3[i]+r4[i]) for i in range(len(completions))]


DEFAULT_IMAGE_TOKEN="<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IGNORE_INDEX = -100


# SYSTEM_PROMPT = (
#     "A conversation between User and Assistant. The user asks a VQA question, and the Assistant solves it step by step."
#     "First, decompose the question into sub-problems: <subproblem>[sub-questions, comma-separated]</subproblem>. "
#     "Then determine their types: <type>[yesno, object, attribute, count, location, action, description]</type>. "
#     "Answer each sub-question: <answer>[answers, comma-separated]</answer>. "
#     "Reflect based on type: <rethink>[reflections, comma-separated]</rethink>. "
#     "Finally, summarize the result: <lastanswer>[final answer]</lastanswer>."
#     "For 'yesno' type: if yes, provide object position (four corner coordinates); if no, explain why. "
#     "For 'count' and 'location' types: return object positions and reflect on correctness. "
#     "For 'object', 'attribute', 'action', 'description': answer based on masked content."
# )

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



# def process_data(item):
#
#     s = (SYSTEM_PROMPT+"\n"
#          +"USER:"+
#          item["conversations"][0]['value']
#          +"\nASSISTANT:"
#          +"<subproblem>"+str([item["sub problem"][j]["question"] for j in range(len(item["sub problem"]))])+"</subproblem>,"
#          +"<type>"+str([item["sub problem"][j]["question_type"] for j in range(len(item["sub problem"]))])+"</type>,"
#          +"<answer>"+str([item["sub problem"][j]["answer"] for j in range(len(item["sub problem"]))])+"</answer>,"
#          + "<rethink>" + str([item["sub problem"][j].get("rethink", item["sub problem"][j]["mask answer"]) for j in range(len(item["sub problem"]))]) + "</rethink>,"
#          +"<lastanswer>"+str(item["conversations"][1]['value'])+"</lastanswer>"
#          )
#
#     return s


def process_data(item):

    s = (SYSTEM_PROMPT+"\n"
         +"USER:"+
         item["conversations"][0]['value']
         +"\nASSISTANT:"
         +item["conversations"][1]['value']
         )

    return s


def preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        chosen: int = 1
) -> Dict:
    conversations = []
    prompts = []
    for source in sources:
        # 构造 prompt

        user_value = source["conversations"][0]['value']
        if isinstance(user_value, (list, tuple)):
            user_value = user_value[0]  # 提取第一个元素作为字符串
        user_value = str(user_value).strip()

        s1 = SYSTEM_PROMPT + "\nUSER:" + user_value + "\nASSISTANT:"
        prompts.append(s1)
        s2 = process_data(source)
        conversations.append(s2)

    # Tokenize conversations

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt').squeeze(0) for prompt in
                     conversations]
        prompt_input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt').squeeze(0) for prompt in
                            prompts]

        eos_token_id = tokenizer.eos_token_id

        for i in range(len(input_ids)):
            if input_ids[i][-1].item() != eos_token_id:
                input_ids[i] = torch.cat([input_ids[i], torch.tensor([eos_token_id], dtype=input_ids[i].dtype)])

        # 再进行 pad_sequence
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        prompt_input_ids = pad_sequence(prompt_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

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

    # Mask targets
    sep = "\n" + "ASSISTANT" + ":"  # 自定义分隔符，如 "ASSISTANT: "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split("</s>")
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX  # 初始偏移位置设为 IGNORE_INDEX

        for i, rou in enumerate(rounds):
            if not rou.strip():
                continue

            parts = rou.split(sep)
            if len(parts) < 2:
                break

            user_part = parts[0].strip()
            assistant_part = parts[1].strip()

            if has_image:
                user_tokens = len(tokenizer_image_token(user_part, tokenizer))
                assistant_tokens = len(tokenizer_image_token(assistant_part, tokenizer))


            else:
                user_tokens = len(tokenizer(user_part).input_ids)
                assistant_tokens = len(tokenizer(assistant_part).input_ids)

            # Mask 用户输入部分
            target[cur_len: cur_len + user_tokens] = IGNORE_INDEX
            cur_len += user_tokens

            # 不 mask 助手回答部分
            cur_len += assistant_tokens

        target[cur_len:] = IGNORE_INDEX  # 剩余部分也设为 IGNORE_INDEX（超出 max_length）

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                # print(
                #     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                #     f" (ignored)"
                # )
    for i, ids in enumerate(input_ids[:3]):
        if tokenizer.eos_token_id in ids:
            print(f"[DEBUG] Sample {i} contains </s> at position {ids.tolist().index(tokenizer.eos_token_id)}")
        else:
            print(f"[WARNING] Sample {i} missing </s> !!!")

    if chosen == 1:
        return dict(chosen_input_ids=input_ids, chosen_labels=targets, prompt=prompts,
                    prompt_input_ids=prompt_input_ids)
    else:
        return dict(rejected_input_ids=input_ids, rejected_labels=targets)






local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# def preprocess_multimodal(
#     sources: Sequence[str],
#     data_args: DataArguments
# ) -> Dict:
#     is_multimodal = data_args.is_multimodal
#     if not is_multimodal:
#         return sources
#
#     for source in sources:
#         for sentence in source:
#             if DEFAULT_IMAGE_TOKEN in sentence['content']:
#                 sentence['content'] = sentence['content'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
#                 sentence['content'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['content']
#                 sentence['content'] = sentence['content'].strip()
#             if data_args.mm_use_im_start_end:
#                 replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
#             sentence["content"] = sentence["content"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
#
#     return sources

@dataclass
class DataArguments:
    data_path: str = field(default="/data2/gaodz/Re-Align/data_step6.json",
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default="/data2/gaodz/train2014")
    image_aspect_ratio: str = 'pad'


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        # 不再调用 super().__init__()，因为 torch Dataset 没有 init 参数
        self.tokenizer = tokenizer
        with open(data_path, "r") as f:
            self.list_data_dict = [json.loads(line) for line in f if line.strip()]
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
        source = self.list_data_dict[i]
        sources = [source]  # preprocess expects list

        # 图像处理
        if 'image' in source:
            image_file = 'COCO_train2014_' + source['image']
            image_path = os.path.join(self.data_args.image_folder, image_file)
            image = Image.open(image_path).convert('RGB')
            processor = self.data_args.image_processor

            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    w, h = pil_img.size
                    size = max(w, h)
                    result = Image.new(pil_img.mode, (size, size), background_color)
                    result.paste(pil_img, ((size - w) // 2, (size - h) // 2))
                    return result

                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))

            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]  # shape: [3, H, W]
        else:
            image = None

        # 调用 preprocess 构造 token
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in source),
            chosen=1
        )

        sample = dict(
            prompt_ids=data_dict["prompt_input_ids"][0],
            completion_ids=data_dict["chosen_input_ids"][0],
            raw_data=source
        )
        if image is not None:
            sample["images"] = image

        return sample


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        chosen_input_ids = [instance["chosen_input_ids"] for instance in instances]
        prompt_input_ids = [instance["prompt_input_ids"] for instance in instances]
        images = [instance["images"] for instance in instances if "images" in instance]

        chosen_input_ids = pad_sequence(chosen_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        prompt_input_ids = pad_sequence(prompt_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        if images:
            images = torch.stack(images).to(dtype=torch.bfloat16)
        else:
            crop_size = self.data_args.image_processor.crop_size
            images = torch.zeros(1, 3, crop_size['height'], crop_size['width']).to(dtype=torch.bfloat16)

        return dict(
            prompt_ids=prompt_input_ids,
            prompt_mask=(prompt_input_ids != self.tokenizer.pad_token_id),
            completion_ids=chosen_input_ids,
            completion_mask=(chosen_input_ids != self.tokenizer.pad_token_id),
            images=images,
            raw_data=[instance.get("raw_data") for instance in instances],
        )


def make_supervised_data_module(tokenizer, data_args):
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )















@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="liuhaotian/llava-v1.6-mistral-7b")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default="openai/clip-vit-large-patch14-336")
    mm_vision_select_layer: Optional[int] = field(default=-2)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_patch_merge_type: Optional[str] = field(default='flat')




@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=1048,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
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
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = 2e-5
    group_by_modality_length: bool = field(default=True)
    output_dir: str = "./output/llava-1.6"
    project_name: str = "VLM-alignment"
    wandb_run_name: str = "17K-FULL"
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "todo"}
    )

    sft_weight: float = field(
        default=0.0,
        metadata={"help": "todo"}
    )
if __name__ == "__main__":
    model_path = "/data2/gaodz/llava_v1.6_sft_full"

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.image_processor = CLIPImageProcessor.from_pretrained(model_path)


    model_base = None  # 如果是完整模型，设为 None；如果是 LoRA，设为基础模型路径
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_v1.6", device="cuda")

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        target_modules=['v_proj', 'q_proj', 'k_proj', 'o_proj'],
    )

    model = get_peft_model(model, lora_config)

    for name, module in model.named_modules():
        if "lora" not in name.lower():  # 可选：保留 LoRA 参数精度不变
            module.to(dtype=torch.bfloat16)

    # 同时确保 mm_projector 也是 bfloat16
    model.get_model().mm_projector.to(dtype=torch.bfloat16)
    model.to(device)
    model.print_trainable_parameters()

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

    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,data_path=data_args.data_path,data_args=data_args)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # Configure training arguments using GRPOConfig
    training_args = GRPOConfig(
        output_dir="/data2/gaodz/llava_v1.6_grpo_lora_1",

        learning_rate=1e-5,
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        bf16=True,
        max_completion_length=256,
        num_generations=4,
        max_prompt_length=512,
        report_to=[],
        push_to_hub=False,
        hub_token=None,
        logging_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=1
    )

    trainer = CustomGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        # reward_funcs=[lambda x: combined_reward(x, raw_data=trainer.accelerator.unwrap_model(trainer.model).raw_data)],
        reward_funcs=[combined_reward]
    )


    trainer.train()

    trainer.save_model(training_args.output_dir)

