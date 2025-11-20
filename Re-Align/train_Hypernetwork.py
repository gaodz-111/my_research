import os
import sys  # 新增：导入sys模块（原代码缺少，导致sys.path.append报错）

os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # 与你的VQDiffusionVAE一致
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset  # 新增：导入Dataset（原代码MultimodalDataset继承自Dataset但未导入）
from torchvision import transforms
from PIL import Image
import json
from tqdm import tqdm
from diffusers import AutoencoderKL
from itertools import chain
from transformers import AutoProcessor, AutoModelForCausalLM  # LLaVA相关

# ===== LLaVA 依赖 =====
parent_dir = os.path.abspath("./llava")  # 指向本机 llava 目录
sys.path.append(parent_dir)
from conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX


# --------------------------
# 1. 复用你的基础组件（数据加载、VQDiffusionVAE核心逻辑）- 修正参数兼容问题
# --------------------------

class MultimodalDataset(Dataset):
    def __init__(self, json_file, image_folder, tokenizer, image_processor, max_length=512):
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

        # 修正：支持json_file为"文件路径"或"已加载的data列表"（兼容你后续传data_samples的逻辑）
        if isinstance(json_file, str):  # 若传文件路径，读文件
            self.data = []
            with open(json_file, "r", encoding="utf-8") as f:
                for line in f:
                    self.data.append(json.loads(line))
        else:  # 若传已加载的列表（如你后续的data_samples），直接使用
            self.data = json_file

        self.conv_mode = "vicuna_v1"
        self.conv_template = conv_templates[self.conv_mode]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        qs = item["conversations"][0]["value"]  # 用户问题
        answer = item["conversations"][1]["value"]  # 模型回答

        system_prompt = "You are a VQA assistant and you need to describe the content of the image in different styles of text."
        prompt = system_prompt + "\n" + 'USER:' + qs + "\nASSISTANT:" + answer

        # 修正：tokenizer_image_token返回[1, seq_len]，此处保持原逻辑，后续collate_fn再squeeze
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        image_path = os.path.join(self.image_folder, item["image"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_processor(image, return_tensors="pt")["pixel_values"][0]  # [3, H, W]

        return {
            "input_ids": input_ids,  # 形状：[1, seq_len]
            "image": image_tensor  # 形状：[3, H, W]
        }


# =========================
# collate_fn - 修正：补充attention_mask生成（原代码缺失导致返回报错）
# =========================
def collate_fn(batch, pad_id=0):
    # 处理input_ids：挤压batch维度，再pad
    input_ids_list = [item["input_ids"].squeeze(0) for item in batch]  # 每个元素形状：[seq_len]
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=pad_id
    )  # 形状：[B, max_seq_len]

    # 新增：生成attention_mask（pad位置为0，其他为1）
    attention_mask = (input_ids_padded != pad_id).float()  # 形状：[B, max_seq_len]

    # 处理image：堆叠成batch
    images = torch.stack([item["image"] for item in batch], dim=0)  # 形状：[B, 3, H, W]

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "image": images  # 新增：将image加入返回，供后续VAE使用
    }


# --------------------------
# 2. 复用你的VAE加载逻辑（与VQDiffusionVAE一致）- 无修改
# --------------------------
def load_pretrained_vae_and_codebook(vae_dir, codebook_pth_path):
    """
    加载你训练好的VAE（微调后的）和Codebook
    vae_dir: Stable Diffusion模型目录（与你的VQDiffusionVAE一致）
    codebook_pth_path: 你训练的VQDiffusionVAE的epoch_x.pth路径
    """
    vae_dir = os.path.join(vae_dir, "vae") if os.path.exists(os.path.join(vae_dir, "vae")) else vae_dir
    vae = AutoencoderKL.from_pretrained(
        vae_dir,
        torch_dtype=torch.float32  # 与你的VQDiffusionVAE一致
    ).cuda()

    # 加载你训练好的Codebook权重
    checkpoint = torch.load(codebook_pth_path, map_location="cuda")
    codebook_K = checkpoint["codebook_state_dict"]["weight"].shape[0]
    codebook_D = checkpoint["codebook_state_dict"]["weight"].shape[1]  # 固定为4（与你的VAE潜变量一致）
    codebook = nn.Embedding(codebook_K, codebook_D).to(torch.float32).cuda()
    codebook.load_state_dict(checkpoint["codebook_state_dict"])

    # 冻结VAE和Codebook（Hypernetwork训练只调调制参数）
    for param in vae.parameters():
        param.requires_grad_(False)
    codebook.requires_grad_(False)

    # 复用你的scale_factor
    scale_factor = vae.config.scaling_factor if hasattr(vae.config, 'scaling_factor') else 0.18215
    return vae, codebook, codebook_K, codebook_D, scale_factor


# --------------------------
# 3. 对齐你的VQ量化逻辑（复用直通梯度和维度处理）- 无修改
# --------------------------
def vq_quantize(z_e, codebook):
    """复用你的量化逻辑：z_e→z_q_st（直通梯度）、indices"""
    codebook_D = codebook.weight.shape[1]
    # 展平潜变量：[B,4,32,32] → [B*32*32,4]
    z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, codebook_D)
    # 计算L2距离
    dist = torch.cdist(z_e_flat, codebook.weight, p=2)
    indices = dist.argmin(dim=1)
    indices = torch.clamp(indices, 0, codebook.weight.shape[0] - 1)
    # 量化向量
    z_q_flat = codebook(indices)
    # 重塑回空间维度
    z_q = z_q_flat.view(z_e.shape[0], z_e.shape[2], z_e.shape[3], codebook_D).permute(0, 3, 1, 2)
    # 直通梯度
    z_q_st = z_e + (z_q - z_e).detach()
    return z_q_st, indices, z_q


# --------------------------
# 4. LLaVA风格Embedding提取（对齐你的图文数据）- 修正参数和调用逻辑
# --------------------------
class LLaVAStyleEncoder:
    def __init__(self, model, image_processor):
        self.model = model
        self.image_processor = image_processor
        self.device = next(model.parameters()).device

        # 冻结LLaVA模型参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 确保模型处于eval模式
        self.model.eval()


    def get_style_embedding(self, batch) -> torch.Tensor:
        # 输入文本张量转换为模型一致的类型（通常是float16）
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # 图像张量强制为float16（与LLaVA模型匹配）
        images = batch["image"].to(self.device, dtype=torch.float16)

        # 检查并修复<image>标记数量
        image_token_id = IMAGE_TOKEN_INDEX
        batch_size = input_ids.shape[0]
        valid_input_ids = []

        for i in range(batch_size):
            image_token_count = (input_ids[i] == image_token_id).sum().item()

            if image_token_count > 1:
                image_token_indices = (input_ids[i] == image_token_id).nonzero().squeeze()
                if image_token_indices.numel() > 1:
                    for idx in image_token_indices[1:]:
                        input_ids[i, idx] = 0  # 替换为pad token

            valid_input_ids.append(input_ids[i])

        input_ids = torch.stack(valid_input_ids)

        # 使用no_grad避免梯度计算，但保持数据类型正确
        with torch.no_grad():
            # 模型前向传播，确保所有中间张量为float16
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,  # 显式转为float16
                return_dict=True,
                output_hidden_states=True,
            )

        # 提取隐藏状态并保持float16类型
        text_embedding = outputs.hidden_states[-1].mean(dim=1).clone()
        text_embedding = text_embedding.half()  # 确保输出为float16

        # 克隆并启用梯度（转为float32供hypernet使用，因为hypernet可能用float32）
        text_embedding = text_embedding.float().requires_grad_(True)
        return text_embedding

# --------------------------
# 5. 修改HypernetworkForCodebook（对齐你的Codebook维度D=4）- 无修改
# --------------------------
class HypernetworkForCodebook(nn.Module):
    def __init__(self, codebook_K, codebook_D=4, style_dim=4096, modulation_type="affine"):
        super().__init__()
        self.modulation_type = modulation_type
        self.codebook_K = codebook_K
        self.codebook_D = codebook_D

        # 投影层初始化时使用float32，与转换后的style_emb匹配
        self.style_proj = nn.Linear(style_dim, codebook_D).float()

        if modulation_type == "affine":
            self.modulator = nn.Sequential(
                nn.Linear(codebook_D, 256).float(),
                nn.ReLU(),
                nn.Linear(256, codebook_K * codebook_D * 2).float()
            )
            nn.init.zeros_(self.modulator[-1].weight)
            nn.init.constant_(self.modulator[-1].bias, 0.0)
            bias = self.modulator[-1].bias.data
            bias[codebook_K * codebook_D:] = 0.0
            bias[:codebook_K * codebook_D] = 1.0

    def forward(self, style_emb, codebook_weight):
        # 确保codebook权重与输入类型匹配
        codebook_weight = codebook_weight.float()

        B = style_emb.shape[0]
        # 投影风格embedding（此时style_emb已转为float32）
        proj_emb = self.style_proj(style_emb)  # [B,4]

        mod_params = self.modulator(proj_emb)  # [B, 2*K*D]
        gamma = mod_params[:, :self.codebook_K * self.codebook_D].view(B, self.codebook_K, self.codebook_D)
        beta = mod_params[:, self.codebook_K * self.codebook_D:].view(B, self.codebook_K, self.codebook_D)

        codebook_weight = codebook_weight.unsqueeze(0).expand(B, -1, -1)  # [B,K,4]
        modulated_codebook = codebook_weight * gamma + beta
        return modulated_codebook


# --------------------------
# 6. 训练主逻辑（完全对齐你的VQDiffusionVAE训练流程）- 修正数据流转和语法错误
# --------------------------
if __name__ == "__main__":
    # --------------------------
    # 6.1 配置参数（与你的VQDiffusionVAE一致）- 无修改
    # --------------------------
    VAE_DIR = "/data2/gaodz/stable-diffusion-2-1-base"  # 你的扩散模型目录
    CODEBOOK_PTH = "/data2/gaodz/VQDiffusionVAE/epoch_10.pth"  # 你训练的VQ模型
    DATA_JSON = "/data2/gaodz/Re-Align/hypernet_train_data.json"  # 你的数据JSON
    IMAGE_ROOT = "/data2/gaodz/OmniConsistency"  # 你的图像根目录
    BATCH_SIZE = 4  # 与你的VQ训练一致
    EPOCHS = 10
    LEARNING_RATE = 1e-5  # 与你的优化器一致
    BETA = 0.25  # 承诺损失权重，与你的VQDiffusionVAE一致
    PAD_ID = 0  # 补充：tokenizer的pad_id（默认0，与collate_fn一致）

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/data2/gaodz/llava-v1.6-vicuna-7b"
    # 复用你的LLaVA加载逻辑（无修改）
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_v1.6", device=device)

    # --------------------------

    # --------------------------
    # 加载数据样本（保持你的原逻辑：先读JSON到列表）
    data_samples = []
    with open(DATA_JSON, "r",encoding="utf-8") as f:
        for line in f:
            data_samples.append(json.loads(line))
    # 初始化数据集（修正：传入data_samples列表，而非文件路径）
    dataset = MultimodalDataset(
        json_file=data_samples,  # 此处传已加载的列表（兼容你原逻辑）
        image_folder=IMAGE_ROOT,
        tokenizer=tokenizer,
        image_processor=image_processor
    )
    # 初始化dataloader（修正：collate_fn传入pad_id）
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        drop_last=True,
        collate_fn=lambda x: collate_fn(x, pad_id=PAD_ID)  # 传入pad_id，避免硬编码
    )

    # --------------------------
    # 6.3 加载预训练组件（VAE、Codebook、LLaVA）- 修正LLaVA编码器初始化
    # --------------------------
    # 加载你的VAE和Codebook（无修改）
    vae, codebook, codebook_K, codebook_D, scale_factor = load_pretrained_vae_and_codebook(VAE_DIR, CODEBOOK_PTH)
    print(f"✅ 加载完成：VAE(scale={scale_factor}) | Codebook(K={codebook_K}, D={codebook_D})")

    # 修正：初始化LLaVA风格编码器（传入model和image_processor，原代码缺失）
    llava_encoder = LLaVAStyleEncoder(model=model, image_processor=image_processor)
    print(f"✅ LLaVA风格编码器初始化完成")

    # --------------------------
    # 6.4 初始化Hypernetwork和优化器 - 无修改
    # --------------------------
    hypernet = HypernetworkForCodebook(
        codebook_K=codebook_K,
        codebook_D=codebook_D,  # D=4，与你的Codebook一致
        style_dim=4096,  # LLaVA输出维度
        modulation_type="affine"
    ).cuda()

    # 优化器：只优化Hypernetwork（复用你的AdamW）
    optimizer = torch.optim.AdamW(
        hypernet.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-6  # 可选：防止过拟合
    )

    # --------------------------
    # 6.5 训练循环（对齐你的VQDiffusionVAE损失逻辑）- 修正数据读取和语法错误
    # --------------------------
    for epoch in range(EPOCHS):
        hypernet.train()
        total_loss_epoch = 0.0
        pbar = tqdm(dataloader, desc=f"Hypernet Epoch {epoch + 1}/{EPOCHS}")

        # 修正：dataloader返回的是batch字典，而非(images, texts)
        for batch in pbar:
            # 从batch中提取数据并移到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device, dtype=torch.float16)  # 修正：与VAE的float32对齐
            B = images.shape[0]
            optimizer.zero_grad()

            # --------------------------
            # 步骤1：VAE编码生成z_e（复用你的VAE前向逻辑）- 无修改
            # --------------------------
            images_t = images.to(torch.float32)
                # 复用你的VAE编码：图像→后验分布→采样z_e→应用scale_factor
            posterior = vae.encode(images_t).latent_dist
            z_e = posterior.sample()  # [B,4,32,32]
            z_e = z_e * scale_factor  # 与你的VQDiffusionVAE一致

            # --------------------------
            # 步骤2：LLaVA提取图文风格embedding - 修正：传入batch字典
            # --------------------------

            style_emb = llava_encoder.get_style_embedding(batch=batch)  # [B,4096]

            # --------------------------
            # 步骤3：Hypernetwork调制Codebook - 无修改
            # --------------------------
            modulated_codebook = hypernet(style_emb, codebook.weight)  # [B,K,4]

            # --------------------------
            # 步骤4：VQ量化（复用你的量化逻辑）- 无修改
            # --------------------------
            # 对每个样本，用自己的调制Codebook量化
            z_q_st_list = []
            z_q_list = []
            indices_list = []
            for b in range(B):
                # 第b个样本的z_e和调制Codebook
                z_e_b = z_e[b:b + 1]  # [1,4,32,32]
                codebook_b = nn.Embedding.from_pretrained(modulated_codebook[b]).cuda()  # [K,4]
                # 量化
                z_q_st_b, indices_b, z_q_b = vq_quantize(z_e_b, codebook_b)
                z_q_st_list.append(z_q_st_b)
                z_q_list.append(z_q_b)
                indices_list.append(indices_b)
            # 合并batch
            z_q_st = torch.cat(z_q_st_list, dim=0)  # [B,4,32,32]
            z_q = torch.cat(z_q_list, dim=0)  # [B,4,32,32]
            indices = torch.cat(indices_list, dim=0)  # [B*32*32]

            # --------------------------
            # 步骤5：VAE解码生成重建图（复用你的解码逻辑）- 无修改
            # --------------------------

                # 复用你的解码前处理：z_q_st / scale_factor
            decoder_input = z_q_st / scale_factor
            decoder_input = torch.clamp(decoder_input, -5.0, 5.0)  # 与你的VQ逻辑一致
            x_recon = vae.decode(decoder_input, return_dict=False)[0]  # [B,3,256,256]

            # --------------------------
            # 步骤6：计算损失（完全复用你的VQ损失逻辑）- 无修改
            # --------------------------
            # 1. 重建损失（MSE，与你的recon_loss一致）
            recon_loss = F.mse_loss(x_recon, images)

            # 2. Le损失（z_q_detach vs z_e，与你的Le一致）
            z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, codebook_D)
            z_q_flat = z_q.permute(0, 2, 3, 1).reshape(-1, codebook_D)
            le = F.mse_loss(z_q_flat.detach(), z_e_flat)

            # 3. Lcommit损失（z_e_detach vs z_q，与你的Lcommit一致）
            lcommit = F.mse_loss(z_e_flat.detach(), z_q_flat)

            # 4. 新增：风格对齐损失（调制后的Codebook特征 vs LLaVA embedding）
            # 取每个样本激活的Codebook向量均值，与style_emb对齐
            activated_vecs = []
            for b in range(B):
                # 第b个样本的激活索引
                indices_b = indices[b * 32 * 32: (b + 1) * 32 * 32]
                # 第b个样本的调制Codebook
                codebook_b = modulated_codebook[b]
                # 激活向量的均值
                activated_vec = codebook_b[indices_b].mean(dim=0)  # [4]
                activated_vecs.append(activated_vec)
            activated_vecs = torch.stack(activated_vecs)  # [B,4]
            # 投影style_emb到4维，计算余弦相似度损失
            style_emb_proj = hypernet.style_proj(style_emb)  # [B,4]
            style_cos_sim = F.cosine_similarity(activated_vecs, style_emb_proj, dim=1).mean()
            style_loss = 1 - style_cos_sim  # 相似度越高，损失越小

            # 总损失（复用你的加权逻辑，新增style_loss）
            total_loss = recon_loss + le + BETA * lcommit + 0.1 * style_loss  # 0.1是style_loss权重，可调整

            # --------------------------
            # 步骤7：反向传播与优化 - 无修改
            # --------------------------
            total_loss.backward()
            optimizer.step()

            # --------------------------
            # 步骤8：监控指标（复用你的perplexity计算）- 修正语法错误
            # --------------------------
            total_loss_epoch += total_loss.item() * B
            # 计算码本利用率（perplexity，与你的逻辑一致）
            avg_probs = torch.bincount(indices, minlength=codebook_K).float() / (B * 32 * 32)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            # 修正：进度条显示语法错误（原代码少f字符串）
            pbar.set_postfix({
                "Total Loss": f"{total_loss.item():.4f}",
                "Recon Loss": f"{recon_loss.item():.4f}",
                "Le": f"{le.item():.6f}",
                "Lcommit": f"{lcommit.item():.6f}",
                "Style Loss": f"{style_loss.item():.4f}",
                "Perplexity": f"{perplexity.item():.2f}"
            })

        # --------------------------
        # 6.6 保存模型（复用你的保存逻辑）- 无修改
        # --------------------------
        avg_total_loss = total_loss_epoch / len(dataset)
        print(f"Epoch {epoch + 1} | Avg Total Loss: {avg_total_loss:.4f}")

        # 每5个epoch保存（与你的保存频率一致）
        if (epoch + 1) % 5 == 0:
            save_dir = "/data2/gaodz/HypernetworkVQ"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"hypernet_epoch_{epoch + 1}.pth")
            torch.save({
                "hypernet_state_dict": hypernet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "codebook_K": codebook_K,
                "codebook_D": codebook_D,
                "epoch": epoch + 1
            }, save_path)
            print(f"✅ Hypernetwork saved to {save_path}")