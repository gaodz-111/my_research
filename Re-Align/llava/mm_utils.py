from PIL import Image
from io import BytesIO
import base64
import torch
import math
import ast

from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size['height'])

    image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


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


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    elif image_aspect_ratio == "anyres":
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


# def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
#     import re
#     # Step 1: 匹配所有 special token（包括你新添加的）
#     pattern = r"(<\/?(?:subproblem|type|answer|rethink|lastanswer)>)"
#     parts = re.split(pattern, prompt)
#
#     input_ids = []
#     offset = 0
#
#     for part in parts:
#         if not part:
#             continue
#
#         # Step 2: 判断是否为 special token
#         if re.fullmatch(r"<\/?(?:subproblem|type|answer|rethink|lastanswer)>", part):
#             token_id = tokenizer.convert_tokens_to_ids(part)
#             if token_id is None or token_id < 0:
#                 token_id = tokenizer.unk_token_id
#             input_ids.append(token_id)
#
#         elif '<image>' in part:
#             # 新增逻辑：只要包含 <image> 就拆分处理
#             segments = part.split('<image>')
#             for i, seg in enumerate(segments):
#                 if seg:
#                     chunk_ids = tokenizer(seg, add_special_tokens=False).input_ids
#                     if chunk_ids:
#                         input_ids.extend(chunk_ids)
#                 if i < len(segments) - 1:
#                     input_ids.append(image_token_index)
#
#         else:
#             # Step 3: 对其他部分进行 tokenize，并去掉 BOS/EOS
#             chunk_ids = tokenizer(part, add_special_tokens=False).input_ids
#             if len(chunk_ids) == 0:
#                 continue
#             if chunk_ids[0] == tokenizer.bos_token_id:
#                 chunk_ids = chunk_ids[1:]
#             input_ids.extend(chunk_ids)
#
#     # Step 4: 处理图像 token 的 offset
#     def insert_separator(X, sep):
#         return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]
#
#     # Step 5: 分割 input_ids 并插入图像 token
#     chunks = []
#     current = []
#
#     for token in input_ids:
#         if token == image_token_index:
#             if current:
#                 chunks.append(current)
#                 current = []
#             chunks.append([image_token_index])
#         else:
#             current.append(token)
#     if current:
#         chunks.append(current)
#
#     # Step 6: 去掉开头的 bos_token（如果有的话）
#     final_ids = []
#     offset = 0
#     if chunks and chunks[0] and chunks[0][0] == tokenizer.bos_token_id:
#         offset = 1
#
#     for i, chunk in enumerate(chunks):
#         if i > 0 and image_token_index in chunk:
#             final_ids.append(image_token_index)
#         else:
#             final_ids.extend(chunk[offset:])
#             offset = 0
#
#     if return_tensors == 'pt':
#         return torch.tensor(final_ids, dtype=torch.long).unsqueeze(0)  # (1, N)
#     return final_ids


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    import re
    import torch

    # 内部函数：处理单个prompt，复用原逻辑
    def process_single_prompt(single_prompt):
        # Step 1: 匹配所有 special token（包括新添加的）
        pattern = r"(<\/?(?:subproblem|type|answer|rethink|lastanswer)>)"
        parts = re.split(pattern, single_prompt)

        input_ids = []

        for part in parts:
            if not part:
                continue

            # Step 2: 判断是否为 special token
            if re.fullmatch(r"<\/?(?:subproblem|type|answer|rethink|lastanswer)>", part):
                token_id = tokenizer.convert_tokens_to_ids(part)
                if token_id is None or token_id < 0:
                    token_id = tokenizer.unk_token_id
                input_ids.append(token_id)

            elif '<image>' in part:
                # 处理包含 <image> 的片段
                segments = part.split('<image>')
                for i, seg in enumerate(segments):
                    if seg:
                        chunk_ids = tokenizer(seg, add_special_tokens=False).input_ids
                        if chunk_ids:
                            input_ids.extend(chunk_ids)
                    if i < len(segments) - 1:  # 不是最后一个片段，添加image token
                        input_ids.append(image_token_index)

            else:
                # 处理其他文本片段，去掉 BOS/EOS
                chunk_ids = tokenizer(part, add_special_tokens=False).input_ids
                if len(chunk_ids) == 0:
                    continue
                if chunk_ids[0] == tokenizer.bos_token_id:  # 移除开头的BOS
                    chunk_ids = chunk_ids[1:]
                input_ids.extend(chunk_ids)

        # Step 3: 分割input_ids并处理图像token的offset
        chunks = []
        current = []
        for token in input_ids:
            if token == image_token_index:
                if current:
                    chunks.append(current)
                    current = []
                chunks.append([image_token_index])  # 图像token单独作为一个chunk
            else:
                current.append(token)
        if current:  # 处理剩余部分
            chunks.append(current)

        # Step 4: 去掉开头的BOS（如果存在）
        final_ids = []
        offset = 0
        if chunks and chunks[0] and chunks[0][0] == tokenizer.bos_token_id:
            offset = 1  # 跳过第一个BOS

        for i, chunk in enumerate(chunks):
            if i > 0 and image_token_index in chunk:
                final_ids.append(image_token_index)
            else:
                final_ids.extend(chunk[offset:])  # 应用offset（仅第一个chunk可能需要）
                offset = 0  # 后续chunk无需offset

        return final_ids

    # 处理批量输入：支持单个字符串或字符串列表
    if isinstance(prompt, str):
        prompts = [prompt]  # 单个prompt转为列表
    else:
        prompts = prompt  # 假设是列表（批量输入）

    # 对每个prompt执行处理逻辑
    all_final_ids = [process_single_prompt(p) for p in prompts]

    # 根据return_tensors返回对应格式
    if return_tensors == 'pt':
        # 转为tensor列表，每个形状为(seq_len,)，后续可通过pad_sequence拼接
        return [torch.tensor(ids, dtype=torch.long) for ids in all_final_ids]
    else:
        # 返回列表的列表（批量结果）
        return all_final_ids


# def tokenizer_image_token(prompts, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
#     """
#     处理包含<image>标记的prompt列表，转换为模型输入ID
#
#     参数:
#         prompts: 单个字符串或字符串列表，每个字符串可能包含<image>标记
#         tokenizer: 文本分词器
#         image_token_index: 图像令牌的索引值
#         return_tensors: 返回张量类型，支持'pt'（PyTorch张量）或None（返回列表）
#
#     返回:
#         若return_tensors='pt'，返回形状为[batch_size, seq_len]的张量；否则返回列表的列表
#     """
#     # 确保输入是列表（统一处理单个prompt和批量prompt）
#     if isinstance(prompts, str):
#         prompts = [prompts]
#
#     batch_input_ids = []
#     for prompt in prompts:
#         # 分割prompt中<image>标记前后的文本
#         prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
#
#         def insert_separator(X, sep):
#             """在 chunks 之间插入图像令牌"""
#             return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]
#
#         input_ids = []
#         offset = 0
#         # 处理bos_token（如果存在）
#         if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
#             offset = 1
#             input_ids.append(prompt_chunks[0][0])
#
#         # 插入图像令牌并拼接所有chunk
#         for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
#             input_ids.extend(x[offset:])
#
#         batch_input_ids.append(input_ids)
#
#     # 处理返回格式
#     if return_tensors is not None:
#         if return_tensors == 'pt':
#             # 计算批次中最长序列的长度，用于补齐
#             max_len = max(len(ids) for ids in batch_input_ids)
#             # 补齐序列（用pad_token_id填充）
#             padded_ids = []
#             for ids in batch_input_ids:
#                 pad_len = max_len - len(ids)
#                 padded_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
#             return torch.tensor(padded_ids, dtype=torch.long)
#         raise ValueError(f'Unsupported tensor type: {return_tensors}')
#
#     return batch_input_ids

# def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
#     prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
#
#     def insert_separator(X, sep):
#         return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]
#
#     input_ids = []
#     offset = 0
#     if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
#         offset = 1
#         input_ids.append(prompt_chunks[0][0])
#
#     for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
#         input_ids.extend(x[offset:])
#
#     if return_tensors is not None:
#         if return_tensors == 'pt':
#             return torch.tensor(input_ids, dtype=torch.long)
#         raise ValueError(f'Unsupported tensor type: {return_tensors}')
#     return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
