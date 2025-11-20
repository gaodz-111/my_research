# train_vqvae.py（严格对照网页梯度公式）
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import json
from torchvision import transforms
from tqdm import tqdm
from diffusers import AutoencoderKL
# class COCOLikeDataset(Dataset):
#     def __init__(self, data_samples, root_dir, transform=None):
#         """
#         Args:
#             data_samples: 列表，每个元素是字典 {"file_name": str, "text": str}
#             root_dir: 图片根目录（例如 "data/coco2014"，与 file_name 拼接得到完整路径）
#             transform: 图片预处理变换
#         """
#         self.data_samples = data_samples  # 您的数据集列表
#         self.root_dir = root_dir          # 图片根目录
#         self.transform = transform        # 图片变换
#
#     def __len__(self):
#         return len(self.data_samples)
#
#     def __getitem__(self, idx):
#         # 获取单一样本的信息
#         sample = self.data_samples[idx]
#         img_path = os.path.join(self.root_dir, sample["file_name"])  # 拼接完整路径
#         text = sample["text"]  # 获取文本描述
#
#         # 加载图片
#         try:
#             image = Image.open(img_path).convert("RGB")  # 确保是RGB格式
#         except Exception as e:
#             print(f"加载图片失败: {img_path}, 错误: {e}")
#             # 可选：返回前一个样本或占位符
#             return self.__getitem__((idx + 1) % len(self))
#
#         # 应用图片变换
#         if self.transform:
#             image = self.transform(image)
#
#         # 返回 (图片张量, 文本描述)
#         return image, text
#
# class VQVAE(nn.Module):
#     def __init__(self, codebook_K=8192, codebook_D=256, beta=0.25, ema=False):
#         super().__init__()
#         # 编码器：3×256→D×64×64（添加padding=1，确保尺寸整除）
#         self.encoder = nn.Sequential(
#             # 输入256×256 → 输出(256-4+2×1)/2 +1 = 254//2 +1 = 127+1=128
#             nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1),  # 修复1：添加padding=1
#             nn.ReLU(),
#             # 输入128×128 → 输出(128-4+2×1)/2 +1 = 126//2 +1 = 63+1=64
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 修复2：添加padding=1
#             nn.ReLU(),
#             nn.Conv2d(256, codebook_D, kernel_size=1)  # 最终输出：[B, D, 64, 64]
#         )
#         # 解码器：D×64×64→3×256（添加padding=1，确保尺寸回传）
#         self.decoder = nn.Sequential(
#             # 输入64×64 → 输出(64-1)×2 +4 - 2×1 = 126 +4 -2=128
#             nn.ConvTranspose2d(codebook_D, 256, kernel_size=4, stride=2, padding=1),  # 修复3：添加padding=1
#             nn.ReLU(),
#             # 输入128×128 → 输出(128-1)×2 +4 -2×1=254+4-2=256
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 修复4：添加padding=1
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 3, kernel_size=1)  # 最终输出：[B, 3, 256, 256]（与输入一致）
#         )
#         # Codebook部分不变
#         self.codebook = nn.Embedding(codebook_K, codebook_D)
#         nn.init.uniform_(self.codebook.weight, -1 / codebook_K, 1 / codebook_K)
#         self.beta = beta
#         self.ema = ema
#
#     def forward(self, x):
#         z_e = self.encoder(x)  # 现在输出：[B, D, 64, 64]（原62→64）
#         z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, z_e.shape[1])  # [B*64*64, D]（原B*62*62）
#
#         # 向量距离计算（逻辑不变）
#         dist = torch.cdist(z_e_flat, self.codebook.weight, p=2)  # [B*64*64, K]
#         indices = dist.argmin(dim=1)  # [B*64*64]
#         z_q_flat = self.codebook(indices)  # [B*64*64, D]
#         # 重塑回空间维度（现在是64×64）
#         z_q = z_q_flat.view(z_e.shape[0], z_e.shape[2], z_e.shape[3], z_e.shape[1]).permute(0, 3, 1, 2)  # [B, D, 64, 64]
#
#         # 直通梯度技巧（逻辑不变）
#         z_q_st = z_e + (z_q - z_e).detach()
#         x_recon = self.decoder(z_q_st)  # 现在输出：[B, 3, 256, 256]（与输入x尺寸一致）
#
#         # 损失计算（现在x_recon和x尺寸完全匹配，可正常计算MSE）
#         recon_loss = F.mse_loss(x_recon, x)  # 修复后无维度冲突
#         le = F.mse_loss(z_q.detach(), z_e)
#         lcommit = F.mse_loss(z_q, z_e.detach())
#         total_loss = recon_loss + le + self.beta * lcommit
#
#         # EMA更新（逻辑不变）
#         if self.training and self.ema:
#             with torch.no_grad():
#                 avg_z_e = z_e_flat.mean(dim=0)
#                 self.codebook.weight.data = 0.99 * self.codebook.weight + 0.01 * avg_z_e.repeat(self.codebook_K, 1)
#
#         # 返回的indices尺寸变为[B, 64, 64]（原[B, 62, 62]）
#         return total_loss, x_recon, indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])
#
#
# # 训练循环（对照网页训练逻辑）
# if __name__ == "__main__":
#     coco_data = []
#     with open("/data2/gaodz/Flickr8k/metadata.jsonl", "r") as f:
#         for line in f:
#             item = json.loads(line)
#             if '.jpg.1' in item['file_name']:
#                 item['file_name'] = item['file_name'].replace('.jpg.1', '.jpg')
#             coco_data.append(json.loads(line))
#     transform = transforms.Compose([
#         transforms.Resize(256),  # 缩放短边到256
#         transforms.RandomCrop(256),  # 随机裁剪256×256
#         transforms.ToTensor(),  # 转为张量
#         transforms.Normalize([0.5] * 3, [0.5] * 3)  # 归一化到[-1, 1]
#     ])
#
#     # 2. 初始化数据集（假设 coco_data 是您的数据集列表）
#     dataset = COCOLikeDataset(
#         data_samples=coco_data,  # 例如 [{"file_name": "train/xxx.jpg", "text": "..."}, ...]
#         root_dir="/data2/gaodz/Flickr8k",  # 图片根目录，与 file_name 拼接
#         transform=transform
#     )
#
#     # 3. 数据加载器
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=8,
#         shuffle=True,
#         num_workers=4  # 多进程加载
#     )
#
#     vqvae = VQVAE(codebook_K=8192, codebook_D=256, beta=0.25).cuda()
#     optimizer = torch.optim.Adam([
#         {'params': vqvae.encoder.parameters()},  # 仅编码器+解码器参与承诺损失
#         {'params': vqvae.decoder.parameters()},
#         {'params': vqvae.codebook.parameters(), 'lr': 1e-5}  # Codebook单独低学习率
#     ], lr=1e-4)
#
#     for epoch in range(10):
#         pbar = tqdm(dataloader)
#         # 关键修复：分离图像和文本，只处理图像
#         for images, texts in pbar:  # 这里x其实是(images, texts)，需显式解包
#             # 只对图像张量进行GPU迁移（文本在VQ-VAE训练中暂时不用）
#             images = images.cuda()  # 替换x.cuda()为images.cuda()
#
#             loss, recon, indices = vqvae(images)  # 传入图像张量
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # 计算困惑度（保持不变）
#             avg_probs = torch.bincount(indices.flatten(), minlength=8192).float() / (8 * 64 * 64)  # 修复分母：32→64
#             perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
#             pbar.set_description(f"Loss: {loss.item():.4f} | PPL: {perplexity.item():.2f}")
#
#     save_dir = "/data2/gaodz/VQ_VAE"  # 目标文件夹
#     save_path = os.path.join(save_dir, "codebook.pth")  # 完整保存路径
#
#     # 2. 检查文件夹是否存在，不存在则创建（包括多级目录）
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir, exist_ok=True)  # exist_ok=True：避免目录已存在时报错
#         print(f"已自动创建目标文件夹：{save_dir}")
#
#     # 3. 保存Codebook
#     torch.save(vqvae.codebook.state_dict(), save_path)
#     print(f"Codebook已保存到：{save_path}")


class VQDiffusionVAE(nn.Module):
    def __init__(self, model_dir, codebook_K=8192, beta=0.25):
        super().__init__()
        # 1. 优先初始化设备（避免属性未定义错误）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae_latent_channels = 4  # 固定为VAE config的latent_channels=4

        # 2. 加载VAE（支持ModelScope或官方模型目录）
        vae_dir = os.path.join(model_dir, "vae") if os.path.exists(os.path.join(model_dir, "vae")) else model_dir
        self.diffusion_vae = AutoencoderKL.from_pretrained(
            vae_dir,
            torch_dtype=torch.float32  # 匹配Stable Diffusion精度
        ).to(self.device)
        print(f"✅ 成功加载VAE，设备: {self.diffusion_vae.device}")

        # 3. 筛选并解冻VAE的部分层（协同优化）
        def get_unfrozen_weight_layers(vae_part, num_layers, is_encoder=True):
            """只解冻带权重的层（Conv2d等），跳过激活函数"""
            all_weight_layers = [l for l in vae_part.modules() if hasattr(l, 'weight')]
            # 选择目标层（Encoder取最后2层，Decoder取前2层）
            if is_encoder:
                target_layers = all_weight_layers[-num_layers:] if len(
                    all_weight_layers) >= num_layers else all_weight_layers
            else:
                target_layers = all_weight_layers[:num_layers] if len(
                    all_weight_layers) >= num_layers else all_weight_layers
            # 解冻并调整精度
            for layer in target_layers:
                layer.requires_grad_(True)
                if layer.weight.data.dtype != torch.float32:
                    layer.weight.data = layer.weight.data.to(torch.float32)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.data = layer.bias.data.to(torch.float32)
            return target_layers

        # 解冻Encoder和解码器的关键层
        self.unfrozen_encoder_layers = get_unfrozen_weight_layers(
            self.diffusion_vae.encoder, num_layers=2, is_encoder=True
        )
        self.unfrozen_decoder_layers = get_unfrozen_weight_layers(
            self.diffusion_vae.decoder, num_layers=2, is_encoder=False
        )
        print(f"✅ 解冻Encoder层数量: {len(self.unfrozen_encoder_layers)}")
        print(f"✅ 解冻Decoder层数量: {len(self.unfrozen_decoder_layers)}")

        # 4. 初始化Codebook（嵌入空间，维度=4，匹配VAE潜变量）
        self.codebook_K = codebook_K  # 码本大小
        self.codebook_D = self.vae_latent_channels  # 固定为4通道
        self.codebook = nn.Embedding(self.codebook_K, self.codebook_D).to(torch.float32).to(self.device)
        nn.init.normal_(self.codebook.weight, mean=0, std=0.1)
        print(f"✅ 初始化Codebook: K={self.codebook_K}, D={self.codebook_D}")

        # 5. 其他固定参数
        self.beta = beta  # 承诺损失权重（VQ-VAE标准值）
        self.scale_factor = self.diffusion_vae.config.scaling_factor if hasattr(self.diffusion_vae.config,
                                                                                'scaling_factor') else 0.18215  # 官方缩放因子
        self.to(self.device)  # 模型整体移到目标设备

    def quantize(self, z_e):
        """
        VQ量化核心逻辑（含直通梯度）
        z_e: 4通道潜变量 → [B, 4, H, W]
        返回: 量化后潜变量(z_q_st)、离散索引(indices)、原始量化向量(z_q)
        """

        # 1. 展平潜变量：[B,4,H,W] → [B*H*W, 4]
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, self.codebook_D)
        assert not torch.isnan(z_e_flat).any(), "输入quantize的z_e_flat含nan！"

        # 2. 计算与Codebook的L2距离：[B*H*W, K]
        dist = torch.cdist(z_e_flat, self.codebook.weight, p=2)
        # 3. 找最近邻索引：[B*H*W]
        indices = dist.argmin(dim=1)
        indices = torch.clamp(indices, 0, self.codebook_K - 1)

        # 4. 生成量化向量：[B*H*W, 4]
        z_q_flat = self.codebook(indices)
        if torch.isnan(z_q_flat).any() or torch.isinf(z_q_flat).any():
            # 用z_e_flat替换无效的量化向量（避免污染后续计算）
            mask = torch.isnan(z_q_flat) | torch.isinf(z_q_flat)
            z_q_flat = torch.where(mask, z_e_flat, z_q_flat)
            print("⚠️ 检测到无效的z_q_flat，已替换为z_e_flat")
        # 5. 重塑回空间维度：[B,4,H,W]
        z_q = z_q_flat.view(z_e.shape[0], z_e.shape[2], z_e.shape[3], self.codebook_D).permute(0, 3, 1, 2)

        # 6. 直通梯度技巧（前向用z_q，反向梯度传给z_e）
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, indices, z_q

    def forward(self, x):
        """完整前向流程：图像→Encoder→采样→量化→Decoder→重建"""
        # 输入预处理（匹配VAE输入要求）
        x = x.to(torch.float32).to(self.device)
        B, C, H, W = x.shape
        assert C == 3 and H == 256 and W == 256, "输入需为[B,3,256,256]的图像"

        # --------------------------
        # 1. Encoder：图像→8通道分布参数（mean+logvar）
        # --------------------------

        posterior = self.diffusion_vae.encode(x).latent_dist  # 关键：返回DiagonalGaussianDistribution

        # 2. 重参数化采样（直接调用分布对象的sample()，避免手动计算std/eps）
        z_e = posterior.sample()  # 自动计算：mean + exp(0.5*logvar) * eps，形状[B,4,32,32]

        # 3. 应用scale_factor（与原逻辑一致，还原潜变量范围）
        z_e = z_e * self.scale_factor


        # --------------------------
        # 2. 量化：4通道潜变量→离散编码
        # --------------------------
        z_q_st, indices, z_q = self.quantize(z_e)



        assert not torch.isnan(z_q_st).any() and not torch.isinf(z_q_st).any(), "z_q_st含无效值"

        # Decoder前向（保留no_grad()）
        with torch.no_grad():
            decoder_input = z_e + (z_q - z_e).detach()
            x_recon = self.diffusion_vae.decode(decoder_input, return_dict=False)[0]

        # --------------------------
        # 4. 计算损失（VQ-VAE标准三部分损失）
        # --------------------------
        # 重建损失：重建图像 vs 原始图像

        assert not torch.isnan(x).any() and not torch.isinf(x).any(), "原始图像x含无效值"
        assert not torch.isnan(x_recon).any() and not torch.isinf(x_recon).any(), "重建图像x_recon含无效值"
        recon_loss = F.mse_loss(x_recon, x)


        # 2. Le损失：检查z_e_flat和z_q_flat
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, self.codebook_D)
        z_q_flat = z_q.permute(0, 2, 3, 1).reshape(-1, self.codebook_D)

        assert not torch.isnan(z_e_flat).any() and not torch.isinf(z_e_flat).any(), "z_e_flat含无效值"
        assert not torch.isnan(z_q_flat).any() and not torch.isinf(z_q_flat).any(), "z_q_flat含无效值"
        le = F.mse_loss(z_q_flat, z_e_flat.detach())  # 正确的detach位置


        # 3. Lcommit损失：检查z_e_flat和z_q_flat（复用上面的张量）
        lcommit = F.mse_loss(z_e_flat, z_q_flat.detach())  # 正确的detach位置


        total_loss = recon_loss + le + self.beta * lcommit


        # 额外信息（训练监控用）
        extra_info = {
            "recon_loss": recon_loss.item(),
            "le": le.item(),
            "lcommit": lcommit.item(),
            "indices": indices.view(B, z_e.shape[2], z_e.shape[3])  # 离散编码：[B,32,32]
        }

        return total_loss, x_recon, extra_info


class COCOLikeDataset(nn.Module):
    def __init__(self, data_samples, root_dir, transform=None):
        self.data_samples = data_samples
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        img_path = os.path.join(self.root_dir, sample["file_name"])
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            image = self.transform(image)
        return image, sample["text"]  # 文本暂用不到，后续可用于文本引导

if __name__ == "__main__":
    # 2. 数据预处理（匹配Stable Diffusion输入要求：256×256，归一化到[-1,1]）
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)  # 与Stable Diffusion预训练一致
    ])

    # 3. 加载数据集（以Flickr8k为例）
    data_samples = []
    with open("/data2/gaodz/Re-Align/llava_11k_mult_answer_clean.json", "r") as f:
        for line in f:
            data_samples.append(json.loads(line))
    dataset = COCOLikeDataset(
        data_samples=data_samples,
        root_dir="/data2/gaodz/train2014",
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # 4. 初始化模型与优化器（关键：同时优化VAE部分层、Codebook）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQDiffusionVAE(
        model_dir="/data2/gaodz/stable-diffusion-2-1-base",
        codebook_K=8192,
        beta=0.25
    )
    # 在主训练流程的model初始化后添加
    torch.autograd.set_detect_anomaly(True)
    print("⚠️ 已开启NaN溯源，首次出现无效操作会自动报错")
    from itertools import chain
    # 优化器：同时更新VAE解冻层、Codebook
    LEARNING_RATE = 1e-5
    optimizer_params = [
            # Encoder解冻层参数
            {"params": chain.from_iterable(layer.parameters() for layer in model.unfrozen_encoder_layers), "lr": LEARNING_RATE},
            # Decoder解冻层参数
            {"params": chain.from_iterable(layer.parameters() for layer in model.unfrozen_decoder_layers), "lr": LEARNING_RATE},
            # Codebook参数（单独低学习率，避免更新过快）
            {"params": model.codebook.parameters(), "lr": LEARNING_RATE}
        ]
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=LEARNING_RATE,
    )



    # 5. 训练循环（协同优化三者）
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        total_loss_epoch = 0.0

        for images, _ in pbar:  # 文本暂不用
            images = images.to(device)
            optimizer.zero_grad()

            # 前向传播（计算损失）
            total_loss, x_recon, extra_info = model(images)

            # 反向传播（梯度同时更新VAE、Codebook）
            total_loss.backward()  # 临时添加，验证梯度


            optimizer.step()

            # 监控指标
            total_loss_epoch += total_loss.item() * images.size(0)
            avg_probs = torch.bincount(extra_info["indices"].flatten(), minlength=model.codebook_K).float() / (
                        images.size(0) * 32 * 32)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))  # 码本利用率（越接近K越好）
            pbar.set_postfix({
                "Total Loss": total_loss.item(),
                "Recon Loss": extra_info["recon_loss"],
                "Le": extra_info["le"],
                "Lcommit": extra_info["lcommit"],
                "Perplexity": perplexity.item()
            })


            # 保存模型（同时保存VAE微调参数、Codebook）
            if (epoch + 1) % 5 == 0:
                save_path = f"/data2/gaodz/VQDiffusionVAE/epoch_{epoch + 1}.pth"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    "vae_state_dict": model.diffusion_vae.state_dict(),
                    "codebook_state_dict": model.codebook.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1
                }, save_path)
                print(f"Model saved to {save_path}")