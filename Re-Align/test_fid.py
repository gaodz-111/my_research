import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
from modelscope import StableDiffusionPipeline
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
import sys
from tqdm import tqdm
import json
# 内存优化依赖（可选）
try:
    import xformers
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

# ===== LLaVA 依赖 =====
parent_dir = os.path.abspath("./llava")  # 确保指向正确的llava目录
sys.path.append(parent_dir)
from conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from get_codebook import VQDiffusionVAE  # 你的VQDiffusionVAE定义
from train_Hypernetwork import HypernetworkForCodebook, LLaVAStyleEncoder, vq_quantize  # 确保vq_quantize已定义


# 1. 全局配置（根据你的路径修改）
MODEL_ID = "/data2/gaodz/stable-diffusion-2-1-base"  # SD模型路径
TRAINED_PTH = "/data2/gaodz/VQDiffusionVAE/epoch_10.pth"  # 训练好的VAE+Codebook路径
HY_PATH = "/data2/gaodz/HypernetworkVQ/hypernet_epoch_10.pth"  # Hypernetwork路径
LLAVA_MODEL_PATH = "/data2/gaodz/llava-v1.6-vicuna-7b"  # LLaVA模型路径
JSON_PATH = "/data2/gaodz/Re-Align/hypernet_train_data.json"  # 测试数据JSON
IMAGE_ROOT = "/data2/gaodz/OmniConsistency"  # 参考图片根目录（与JSON中image路径拼接）
OUTPUT_DIR_SD = "/data2/gaodz/VQ_test/SD"  # 原始模型输出目录
OUTPUT_DIR_HY = "/data2/gaodz/VQ_test/HY"  # 调制模型输出目录
os.makedirs(OUTPUT_DIR_SD, exist_ok=True)
os.makedirs(OUTPUT_DIR_HY, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32  # 统一数据类型


# 2. 加载原始SD模型（无调制，用于对比）
def load_original_model(model_path):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True  # 内存优化
    ).to(device)
    # 高效注意力（可选，需安装xformers）
    if XFORMERS_AVAILABLE and device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
    return pipe


# 3. 核心：加载带Hypernetwork的模型（修复LLaVA多模态输入）
def load_trained_hypernetwork(model_path, trained_vae_pth, trained_hypernet_pth, llava_model_path):
    # 3.1 加载SD管道
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)
    if XFORMERS_AVAILABLE and device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()

    # 3.2 加载VAE和基础Codebook
    your_model = VQDiffusionVAE(
        model_path,
        codebook_K=8192,
        beta=0.25
    ).to(device, dtype=dtype)
    vae_checkpoint = torch.load(trained_vae_pth, map_location=device)
    # 加载VAE和Codebook权重（strict=False兼容结构差异）
    your_model.load_state_dict({
        **vae_checkpoint["vae_state_dict"],
        **vae_checkpoint["codebook_state_dict"]
    }, strict=False)
    your_model.eval()
    pipe.vae = your_model.diffusion_vae.to(dtype)  # 替换SD的VAE

    # 3.3 加载Hypernetwork（调制层）
    hypernet_checkpoint = torch.load(trained_hypernet_pth, map_location=device)
    hypernet = HypernetworkForCodebook(
        codebook_K=hypernet_checkpoint["codebook_K"],
        codebook_D=hypernet_checkpoint["codebook_D"],
        style_dim=4096,  # LLaVA-7B输出维度
        modulation_type="affine"
    ).to(device, dtype=dtype)
    hypernet.load_state_dict(hypernet_checkpoint["hypernet_state_dict"], strict=True)
    hypernet.eval()
    print("✅ 成功加载Hypernetwork")

    # 3.4 加载LLaVA（用于提取“图片+文本”多模态向量）
    tokenizer, llava_model, image_processor, _ = load_pretrained_model(
        llava_model_path,
        None,
        "llava_v1.6",
        device=device,
        torch_dtype=dtype  # 与SD统一精度
    )
    llava_encoder = LLaVAStyleEncoder(model=llava_model, image_processor=image_processor)
    print("✅ 成功加载LLaVA风格编码器")

    # 3.5 核心修复：修补VAE的encode逻辑（接收参考图片+文本）
    def patched_encode(self, x, reference_image=None, reference_text=None):
        """
        修复后：LLaVA用“参考图片+参考文本”提向量，调制Codebook
        reference_image: 外部传入的参考图片张量（[1,3,224,224]）
        reference_text: 参考图片对应的文本描述（含风格信息）
        """
        with torch.no_grad():
            # 步骤1：生成VAE潜变量z_e（SD生成流程不变）
            z_e = your_model.diffusion_vae.encode(x).latent_dist.sample()
            z_e = z_e * your_model.scale_factor  # 应用SD VAE的缩放因子

            # 步骤2：LLaVA提取“参考图片+参考文本”的多模态向量（核心修复）
            if reference_image is None or reference_text is None:
                # 无参考时用默认风格（兼容逻辑，实际测试不会走这里）
                style_prompt = "default style"
                llava_prompt = f"<image>{style_prompt}"
                # 用x占位（无参考时的 fallback）
                processed_img = image_processor(x, return_tensors="pt")["pixel_values"].to(device, dtype=dtype)
            else:
                # 有参考时：用“参考文本+参考图片”（你的核心需求）
                # LLaVA要求prompt必须包含<image>标记（匹配训练格式）
                llava_prompt = f"<image>{reference_text}"
                # 预处理参考图片（匹配LLaVA输入格式：224x224，归一化）
                processed_img = image_processor(
                    reference_image,
                    return_tensors="pt"
                )["pixel_values"].to(device, dtype=dtype)  # [1,3,224,224]

            # 构造LLaVA的多模态输入batch
            llava_batch = {
                "input_ids": tokenizer(llava_prompt, return_tensors="pt").input_ids.to(device),
                "attention_mask": torch.ones_like(tokenizer(llava_prompt, return_tensors="pt").input_ids).to(device),
                "image": processed_img  # 传入参考图片（不是x！）
            }
            # 提取多模态风格向量（图片+文本融合）
            style_emb = llava_encoder.get_style_embedding(llava_batch).to(dtype)  # [1,4096]

            # 步骤3：Hypernetwork调制Codebook
            modulated_codebook = hypernet(style_emb, your_model.codebook.weight)  # [1,8192,4]

            # 步骤4：用调制后的Codebook量化z_e
            z_q_st_list = []
            for b in range(x.shape[0]):
                z_e_b = z_e[b:b + 1]  # 单个样本的潜变量 [1,4,32,32]
                # 为每个样本创建调制后的Codebook Embedding
                codebook_b = nn.Embedding.from_pretrained(modulated_codebook[b]).to(device, dtype=dtype)
                z_q_st_b, _, _ = vq_quantize(z_e_b, codebook_b)  # 复用训练时的量化函数
                z_q_st_list.append(z_q_st_b)
            z_q = torch.cat(z_q_st_list, dim=0)  # [B,4,32,32]

            return z_q.to(dtype)

    # 替换VAE的encode方法（让SD生成时自动调用调制逻辑）
    pipe.vae.encode = patched_encode.__get__(pipe.vae)
    print("✅ 已修复VAE编码逻辑，支持‘图片+文本’调制")

    # 返回管道和依赖组件（llava_encoder用于预处理参考图）
    return pipe, hypernet, llava_encoder, image_processor, tokenizer


# 4. 核心：生成对比函数（支持传入参考图片+文本）
def generate_comparison(
    original_pipe, trained_pipe,
    gen_prompt, prompt_name,
    reference_image, reference_text,
    image_processor  # 用于预处理参考图片
):
    """
    生成对比图：原始SD vs 调制SD
    gen_prompt: 要生成的内容文本（如“一只猫”）
    prompt_name: 输出文件名前缀
    reference_image: PIL格式的参考图片（用于调制）
    reference_text: 参考图片对应的文本（含风格信息）
    image_processor: LLaVA的图像处理器（预处理参考图）
    """
    gen_kwargs = {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "width": 512,
        "height": 512,
        "return_dict": True
    }

    # --------------------------
    # 1. 原始模型生成（无调制）
    # --------------------------
    with torch.inference_mode():
        original_out = original_pipe(gen_prompt, **gen_kwargs)
        original_image = original_out.images[0]
    # 保存原始模型结果
    original_save_path = os.path.join(OUTPUT_DIR_SD, f"{prompt_name}.png")
    original_image.save(original_save_path)
    print(f"📌 原始模型结果已保存：{original_save_path}")

    # --------------------------
    # 2. 调制模型生成（用参考图片+文本）
    # --------------------------
    with torch.inference_mode():
        # 预处理参考图片（转为张量，匹配LLaVA输入）
        processed_ref_img = image_processor(
            reference_image,
            return_tensors="pt"
        )["pixel_values"].to(device, dtype=dtype)  # [1,3,224,224]

        # 关键：让trained_pipe的VAE能拿到参考图片和文本
        trained_pipe.vae.reference_image = processed_ref_img
        trained_pipe.vae.reference_text = reference_text

        # 生成（SD生成流程不变，但VAE encode时会自动调制）
        trained_out = trained_pipe(gen_prompt, **gen_kwargs)
        trained_image = trained_out.images[0]

        # 清理临时变量（避免内存堆积）
        del trained_pipe.vae.reference_image
        del trained_pipe.vae.reference_text

    # 保存调制模型结果
    trained_save_path = os.path.join(OUTPUT_DIR_HY, f"{prompt_name}.png")
    trained_image.save(trained_save_path)
    print(f"📌 调制模型结果已保存：{trained_save_path}")

    # 清理内存
    del original_image, trained_image, original_out, trained_out
    torch.cuda.empty_cache()

    return original_save_path, trained_save_path


# 5. 主测试逻辑（加载数据→生成对比）
if __name__ == "__main__":
    # 5.1 加载模型
    print("🔄 加载原始SD模型...")
    original_pipe = load_original_model(MODEL_ID)

    print("\n🔄 加载带Hypernetwork的调制模型...")
    trained_pipe, hypernet, llava_encoder, image_processor, tokenizer = load_trained_hypernetwork(
        MODEL_ID, TRAINED_PTH, HY_PATH, LLAVA_MODEL_PATH
    )

    # 5.2 加载测试数据（逐行读取，避免内存堆积）
    print(f"\n📄 加载测试数据：{JSON_PATH}")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        # 用tqdm显示进度
        index = 0
        for line_num, line in enumerate(tqdm(f, desc="处理测试数据")):
            try:
                item = json.loads(line)

                # 5.2.1 提取关键信息（从JSON item中）
                # 参考图片路径（拼接根目录）
                ref_img_filename = item["image"]  # JSON中存储的图片文件名（如"xxx.png"）
                ref_img_path = os.path.join(IMAGE_ROOT, ref_img_filename)
                # 参考文本（用户问题+模型回答，含风格信息）
                qs = item["conversations"][0]["value"].replace("<image>\n", "")  # 去除原始<image>标记
                answer = item["conversations"][1]["value"]
                system_prompt = "You are a VQA assistant and you need to describe the content of the image in different styles of text."
                reference_text = system_prompt + "\n" + f"USER: {qs}\nASSISTANT: {answer}"
                # 生成目标文本（要生成的内容，可根据需求调整，这里用用户问题的核心内容）
                gen_prompt = qs  # 例如：若qs是“描述这张图的风格”，可改为具体生成内容如“一只狗”
                # 输出文件名（用图片名+行号，避免重复）
                prompt_name = index

                # 5.2.2 加载并预处理参考图片（PIL→保持原始尺寸，后续由LLaVA处理器缩放）
                if not os.path.exists(ref_img_path):
                    print(f"⚠️ 参考图片不存在：{ref_img_path}，跳过该样本")
                    continue
                reference_image = Image.open(ref_img_path).convert("RGB")  # 加载为RGB

                # 5.2.3 生成对比图（核心调用）
                print(f"\n🎯 处理样本 {line_num+1}：{ref_img_filename}")
                generate_comparison(
                    original_pipe=original_pipe,
                    trained_pipe=trained_pipe,
                    gen_prompt=gen_prompt,
                    prompt_name=prompt_name,
                    reference_image=reference_image,
                    reference_text=reference_text,
                    image_processor=image_processor
                )
                index += 1
            except Exception as e:
                print(f"❌ 处理第{line_num+1}行样本失败：{str(e)}，跳过")
                continue

    # 5.3 测试完成，释放资源
    print("\n✅ 所有样本处理完成！")
    del original_pipe, trained_pipe, hypernet, llava_encoder, image_processor, tokenizer
    torch.cuda.empty_cache()
