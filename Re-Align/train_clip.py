import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    get_scheduler
)
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL  # 集成VAE模型（SD v1.5默认VAE）
)
from diffusers.training_utils import EMAModel
from diffusers.schedulers import PNDMScheduler
import lpips
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# --------------------------
# 1. 配置与全局参数
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_TYPE = torch.float16  # 用FP16节省显存
BATCH_SIZE = 1  # 单卡24GB显存建议BATCH_SIZE=1
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
IMAGE_SIZE = 512  # SD v1.5默认输入尺寸
VAE_LATENT_SCALE = 0.18215  # SD VAE的隐空间缩放系数（固定值）


# --------------------------
# 2. 核心创新机制模块
# --------------------------
class HyperCodebook(nn.Module):
    """动态可塑代码簇：通过Hyper-network调节VAE的codebook中心（基于SD VAE的量化层）"""

    def __init__(self, vae_quantizer, hyper_hidden_dim=256):
        super().__init__()
        self.vae_quantizer = vae_quantizer  # SD VAE的量化层（含初始codebook）
        self.base_codebook = vae_quantizer.embedding.weight  # 初始codebook (num_tokens=8192, token_dim=4)
        self.num_tokens, self.token_dim = self.base_codebook.shape

        # Hyper-network：输入LLM的程序嵌入，输出codebook偏移量
        self.hyper_net = nn.Sequential(
            nn.Linear(768, hyper_hidden_dim),  # 768=LLaVA-v1.6的hidden_dim
            nn.ReLU(),
            nn.LayerNorm(hyper_hidden_dim),
            nn.Linear(hyper_hidden_dim, self.num_tokens * self.token_dim)
        ).to(DEVICE, D_TYPE)

    def forward(self, program_embedding):
        """
        Args:
            program_embedding: LLaVA生成的scene-program特征 (batch_size, 768)
        Returns:
            dynamic_codebook: 动态调节后的codebook (num_tokens, token_dim)
        """
        # 生成codebook偏移量（按程序语义适配）
        offsets = self.hyper_net(program_embedding)  # (batch_size, num_tokens*token_dim)
        offsets = offsets.view(-1, self.num_tokens, self.token_dim).mean(dim=0)  # 批量平均，匹配codebook维度

        # 双重更新：EMA（80%基础codebook稳定性） + Hyper-network（20%动态适配）
        dynamic_codebook = 0.8 * self.base_codebook + 0.2 * offsets
        # 更新VAE量化层的codebook（实时生效）
        self.vae_quantizer.embedding.weight.data = dynamic_codebook
        return dynamic_codebook


class AttentionReplay(nn.Module):
    """注意力复放：将LLM的文本→程序交叉注意力，投影到SD U-Net的条件注意力层"""

    def __init__(self, llm_attn_dim=4096, unet_attn_dim=320):
        """
        Args:
            llm_attn_dim: LLaVA交叉注意力的维度（vicuna-7b的cross_attn维度为4096）
            unet_attn_dim: SD v1.5 U-Net的注意力维度（默认320）
        """
        super().__init__()
        # 两级投影：解决LLM与U-Net的维度鸿沟
        self.proj = nn.Sequential(
            nn.Conv1d(llm_attn_dim, 2048, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(2048, unet_attn_dim, kernel_size=1)
        ).to(DEVICE, D_TYPE)
        # 门控机制：动态控制注意力融合比例
        self.gate = nn.Sigmoid().to(DEVICE, D_TYPE)

    def forward(self, llm_cross_attn, unet_self_attn):
        """
        Args:
            llm_cross_attn: LLaVA的文本→程序交叉注意力 (batch_size, num_heads, seq_len, seq_len)
            unet_self_attn: SD U-Net的自注意力 (batch_size, num_heads, feat_len, feat_len)
        Returns:
            fused_attn: 融合后的U-Net注意力 (batch_size, num_heads, feat_len, feat_len)
        """
        batch_size, num_heads, llm_seq_len, _ = llm_cross_attn.shape
        _, _, unet_feat_len, _ = unet_self_attn.shape

        # 1. LLM注意力降维与维度对齐
        llm_attn_flat = llm_cross_attn.mean(dim=1)  # 平均多头注意力 (batch_size, seq_len, seq_len)
        llm_attn_flat = llm_attn_flat.view(batch_size, 1, llm_seq_len * llm_seq_len)  # (batch_size, 1, seq_flat)
        # 投影到U-Net注意力维度
        projected_attn = self.proj(llm_attn_flat.permute(0, 2, 1))  # (batch_size, unet_attn_dim, 1)
        # 扩展为U-Net注意力的空间维度
        projected_attn = projected_attn.unsqueeze(1).repeat(1, num_heads, 1,
                                                            unet_feat_len)  # (batch_size, heads, unet_dim, feat_len)

        # 2. 门控融合：保留U-Net基础注意力，叠加LLM语义聚焦
        gate_weight = self.gate(projected_attn.mean(dim=2, keepdim=True))  # (batch_size, heads, 1, feat_len)
        fused_attn = gate_weight * projected_attn + (1 - gate_weight) * unet_self_attn
        return fused_attn


class ConvRenderer(nn.Module):
    """可微渲染器：将scene-program渲染为多尺度隐空间特征（匹配SD U-Net的分辨率）"""

    def __init__(self, program_dim=768, vae_latent_dim=4):
        """
        Args:
            program_dim: LLaVA程序嵌入维度（768）
            vae_latent_dim: SD VAE隐空间通道数（默认4）
        """
        super().__init__()
        # 多尺度渲染（匹配U-Net的1/16、1/8、1/4分辨率：32×32, 64×64, 128×128）
        self.render_layers = nn.ModuleDict({
            "32x32": nn.Sequential(
                nn.Linear(program_dim, vae_latent_dim * 32 * 32),
                nn.Unflatten(1, (vae_latent_dim, 32, 32))
            ),
            "64x64": nn.Sequential(
                nn.Linear(program_dim, vae_latent_dim * 64 * 64),
                nn.Unflatten(1, (vae_latent_dim, 64, 64))
            ),
            "128x128": nn.Sequential(
                nn.Linear(program_dim, vae_latent_dim * 128 * 128),
                nn.Unflatten(1, (vae_latent_dim, 128, 128))
            )
        }).to(DEVICE, D_TYPE)

    def forward(self, program_embedding):
        """
        Args:
            program_embedding: LLaVA程序嵌入 (batch_size, 768)
        Returns:
            multi_scale_latents: 多尺度隐空间特征（dict: 分辨率→特征）
        """
        return {
            res: layer(program_embedding) for res, layer in self.render_layers.items()
        }


# --------------------------
# 3. 完整模型整合（LLaVA + SD + VAE + 创新机制）
# --------------------------
class MultiScaleVisualProgramModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 加载LLaVA-v1.6（生成scene-program）
        self.llava_processor = LlavaProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b")
        self.llava = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-vicuna-7b",
            torch_dtype=D_TYPE,
            low_cpu_mem_usage=True,
            device_map="auto"  # 自动分配设备（节省显存）
        ).eval()  # 初始冻结LLM，后续可解冻微调

        # 2. 加载Stable Diffusion v1.5核心组件（U-Net + VAE）
        self.sd_unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet",
            torch_dtype=D_TYPE,
            device_map="auto"
        )
        self.sd_vae = AutoencoderKL.from_pretrained(  # 集成VAE（用于隐空间编码/解码）
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae",
            torch_dtype=D_TYPE,
            device_map="auto"
        )
        self.sd_scheduler = PNDMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="scheduler"
        )

        # 3. 初始化创新机制
        self.hyper_codebook = HyperCodebook(vae_quantizer=self.sd_vae.quantize)
        self.attention_replay = AttentionReplay()
        self.conv_renderer = ConvRenderer()

        # 4. 损失函数（联合训练用）
        self.ce_loss = nn.CrossEntropyLoss().to(DEVICE, D_TYPE)  # LLM自回归损失
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE, D_TYPE)  # 结构损失
        self.lpips_loss = lpips.LPIPS(net="vgg").to(DEVICE, D_TYPE)  # 感知损失

    def generate_scene_program(self, text_prompt, reference_image=None):
        """
        用LLaVA生成结构化scene-program（对象/属性/关系/布局/风格）
        Args:
            text_prompt: 文本提示（如"A red cube on blue sphere"）
            reference_image: 参考图像（可选，用于更精准的程序生成）
        Returns:
            program_text: 结构化程序文本（字符串）
            program_embedding: 程序特征嵌入 (batch_size, 768)
            llm_cross_attn: LLaVA的文本→程序交叉注意力 (batch_size, heads, seq_len, seq_len)
        """
        # LLaVA输入处理
        if reference_image is not None:
            inputs = self.llava_processor(
                text=text_prompt,
                images=reference_image,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(DEVICE, D_TYPE)
        else:
            inputs = self.llava_processor(
                text=text_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(DEVICE, D_TYPE)

        # 生成scene-program（带注意力输出）
        with torch.no_grad():  # 初始冻结LLM，后续训练可取消
            outputs = self.llava.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=3,
                output_attentions=True,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

        # 解析输出
        program_text = self.llava_processor.decode(outputs.sequences[0], skip_special_tokens=True)
        # 提取程序嵌入（LLM最后一层hidden state的平均）
        program_embedding = outputs.hidden_states[-1].mean(dim=1)  # (1, 768)
        # 提取文本→程序交叉注意力（最后一层交叉注意力）
        llm_cross_attn = outputs.cross_attentions[-1]  # (1, num_heads, seq_len_text, seq_len_program)

        return program_text, program_embedding, llm_cross_attn

    def vae_encode(self, image):
        """VAE编码：图像→隐空间特征（用于计算扩散重建损失）"""
        image = (image - 0.5) * 2.0  # 归一化到[-1, 1]（SD VAE要求）
        with torch.no_grad():
            latent = self.sd_vae.encode(image).latent_dist.mode()  # (batch_size, 4, 64, 64)
        return latent * VAE_LATENT_SCALE  # 应用SD的隐空间缩放

    def vae_decode(self, latent):
        """VAE解码：隐空间特征→图像（用于可视化生成结果）"""
        latent = latent / VAE_LATENT_SCALE  # 逆缩放
        with torch.no_grad():
            image = self.sd_vae.decode(latent).sample  # (batch_size, 3, 512, 512)
        # 归一化到[0, 1]并转PIL
        image = (image / 2.0 + 0.5).clamp(0, 1)
        image = (image.permute(0, 2, 3, 1) * 255).cpu().numpy().astype(np.uint8)
        return [Image.fromarray(img) for img in image]

    def diffusion_generate(self, program_embedding, llm_cross_attn):
        """
        SD扩散生成：基于程序特征+注意力复放，生成图像
        Args:
            program_embedding: 程序嵌入 (batch_size, 768)
            llm_cross_attn: LLaVA交叉注意力
        Returns:
            generated_latent: 生成的隐空间特征 (batch_size, 4, 64, 64)
            generated_image: 生成的图像（PIL格式）
        """
        batch_size = program_embedding.shape[0]
        # 1. 可微渲染：生成多尺度初始隐空间特征
        multi_scale_latents = self.conv_renderer(program_embedding)
        # 取64×64作为扩散初始隐空间（匹配SD VAE的隐空间分辨率：512/8=64）
        init_latent = multi_scale_latents["64x64"]  # (batch_size, 4, 64, 64)

        # 2. Hyper-codebook动态调节
        self.hyper_codebook(program_embedding)

        # 3. 扩散去噪过程（带注意力复放）
        self.sd_scheduler.set_timesteps(50)  # 简化采样步长
        latents = init_latent * self.sd_scheduler.init_noise_sigma  # 加初始噪声

        for t in tqdm(self.sd_scheduler.timesteps, desc="Diffusion Denoising"):
            # 复制latents用于残差计算
            latent_model_input = self.sd_scheduler.scale_model_input(latents, t)

            # 提取U-Net当前层注意力，叠加注意力复放
            unet_output = self.sd_unet(
                latent_model_input,
                t,
                encoder_hidden_states=program_embedding,
                output_attentions=True,
                return_dict=True
            )
            # 注意力复放：融合LLM注意力到U-Net
            unet_attn = unet_output.attentions[0]  # U-Net第一层自注意力
            fused_attn = self.attention_replay(llm_cross_attn, unet_attn)
            # 用融合注意力重新计算U-Net输出（简化：直接替换注意力权重）
            unet_output.sample = self.sd_unet(
                latent_model_input,
                t,
                encoder_hidden_states=program_embedding,
                attention_mask=fused_attn
            ).sample

            # 去噪更新
            latents = self.sd_scheduler.step(unet_output.sample, t, latents).prev_sample

        # 4. VAE解码生成图像
        generated_image = self.vae_decode(latents)
        return latents, generated_image[0]

    def compute_joint_loss(self, program_outputs, generated_latent, real_image):
        """
        计算联合损失：自回归损失（LLM） + 结构一致性损失（扩散）
        Args:
            program_outputs: LLaVA生成程序的logits (batch_size, seq_len, vocab_size)
            generated_latent: 扩散生成的隐空间特征 (batch_size, 4, 64, 64)
            real_image: 真实图像 (batch_size, 3, 512, 512)
        Returns:
            total_loss: 总损失
            loss_dict: 各损失分项（便于监控）
        """
        # 1. LLM自回归损失（程序生成逻辑正确性）
        logits = program_outputs.logits[:, :-1, :].contiguous()  # 错开一位（预测下一个token）
        labels = program_outputs.sequences[:, 1:].contiguous()
        ce_loss = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))

        # 2. 扩散结构一致性损失（图像生成质量）
        real_latent = self.vae_encode(real_image)  # 真实图像的隐空间特征
        # MSE损失（隐空间重建误差）
        mse_loss = F.mse_loss(generated_latent, real_latent)
        # SSIM损失（结构相似性）
        generated_image = self.vae_decode(generated_latent)[0]
        generated_image_tensor = torch.tensor(np.array(generated_image)).permute(2, 0, 1).unsqueeze(0).to(DEVICE,
                                                                                                          D_TYPE) / 255.0
        real_image_norm = real_image / 255.0  # 归一化到[0,1]
        ssim = self.ssim_loss(generated_image_tensor, real_image_norm)
        ssim_loss = 1 - ssim  # SSIM越大越好，转为损失
        # LPIPS损失（感知相似性）
        lpips_loss = self.lpips_loss(generated_image_tensor, real_image_norm).mean()

        # 总损失（权重可调整）
        total_loss = 0.4 * ce_loss + 0.2 * mse_loss + 0.2 * ssim_loss + 0.2 * lpips_loss
        loss_dict = {
            "total_loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "mse_loss": mse_loss.item(),
            "ssim_loss": ssim_loss.item(),
            "lpips_loss": lpips_loss.item()
        }
        return total_loss, loss_dict


# --------------------------
# 4. 测试数据集（简化版，无需外部标注文件）
# --------------------------
class DemoSceneDataset(Dataset):
    """演示用数据集：包含少量图像和对应的文本提示"""

    def __init__(self, data_dir="demo_data"):
        self.data_dir = data_dir
        # 确保演示数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        # 生成2张演示图像（可替换为自己的图像）
        self._prepare_demo_data()
        # 数据列表：(图像路径, 文本提示)
        self.data = [
            (os.path.join(self.data_dir, "demo1.jpg"), "A red cube on a blue sphere, photorealistic, 8k"),
            (os.path.join(self.data_dir, "demo2.jpg"), "A small yellow cat sitting on a wooden table, cartoon style")
        ]

    def _prepare_demo_data(self):
        """生成演示用图像（随机噪声图，实际用真实图像替换）"""
        for i in range(2):
            demo_img = Image.fromarray(np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
            demo_img.save(os.path.join(self.data_dir, f"demo{i + 1}.jpg"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, prompt = self.data[idx]
        image = Image.open(img_path).resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
        # 转为tensor（供模型使用）
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(DEVICE, D_TYPE)
        return {
            "image": image,
            "image_tensor": image_tensor,
            "prompt": prompt
        }


# --------------------------
# 5. 训练与推理示例
# --------------------------
def train_model(model, dataloader, optimizer, scheduler):
    """训练流程：分阶段优化（先训渲染器/Hyper-codebook，再联合训LLM）"""
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch in progress_bar:
            prompt = batch["prompt"][0]  # 单batch，取第一个样本
            real_image = batch["image_tensor"]  # (1, 3, 512, 512)

            # 1. 生成scene-program（带LLM输出用于损失计算）
            program_text, program_emb, llm_cross_attn = model.generate_scene_program(prompt, batch["image"][0])

            # 2. 扩散生成（带梯度回传）
            generated_latent, generated_img = model.diffusion_generate(program_emb, llm_cross_attn)

            # 3. 计算LLM程序生成的logits（用于自回归损失）
            program_inputs = model.llava_processor(
                text=program_text,
                return_tensors="pt"
            ).to(DEVICE, D_TYPE)
            program_outputs = model.llava(**program_inputs)  # 非generate，获取logits

            # 4. 计算联合损失
            loss, loss_dict = model.compute_joint_loss(program_outputs, generated_latent, real_image)

            # 5. 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 监控
            total_loss += loss.item()
            progress_bar.set_postfix(loss_dict)

        #  epoch结束：打印平均损失 + 保存模型
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
        torch.save({
            "hyper_codebook": model.hyper_codebook.state_dict(),
            "attention_replay": model.attention_replay.state_dict(),
            "conv_renderer": model.conv_renderer.state_dict(),
            "optimizer": optimizer.state_dict()
        }, f"checkpoint_epoch_{epoch + 1}.pt")

        # 可视化生成结果
        generated_img.save(f"generated_epoch_{epoch + 1}.jpg")
        print(f"Generated Image Saved: generated_epoch_{epoch + 1}.jpg")


def infer_model(model, text_prompt):
    """推理流程：文本提示→scene-program→生成图像"""
    model.eval()
    with torch.no_grad():
        # 1. 生成scene-program
        program_text, program_emb, llm_cross_attn = model.generate_scene_program(text_prompt)
        print(f"Generated Scene Program:\n{program_text}\n")

        # 2. 扩散生成图像
        _, generated_img = model.diffusion_generate(program_emb, llm_cross_attn)

        # 3. 显示与保存
        generated_img.show()
        generated_img.save("infer_result.jpg")
        print("Inference Result Saved: infer_result.jpg")
    return generated_img


if __name__ == "__main__":
    # 1. 初始化模型
    model = MultiScaleVisualProgramModel()
    print("Model Initialized Successfully!\n")

    # 2. 初始化优化器与调度器（仅优化创新机制模块，初始冻结LLM/SD主体）
    trainable_params = list(model.hyper_codebook.parameters()) + \
                       list(model.attention_replay.parameters()) + \
                       list(model.conv_renderer.parameters())
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=NUM_EPOCHS * len(DemoSceneDataset())
    )

    # 3. 加载演示数据集
    dataset = DemoSceneDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. 训练模型（注释掉可直接跑推理）
    print("Start Training...")
    train_model(model, dataloader, optimizer, scheduler)

    # 5. 推理示例（训练后运行，或直接注释训练部分跑推理）
    # infer_text = "A green tree with pink flowers in a grass field, watercolor style"
    # infer_model(model, infer_text)