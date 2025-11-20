# model/finelip_text_encoder.py（Finelip 内部文件）
import torch
import torch.nn as nn
import clip
from .alibi_attention import FinelipResidualAttentionBlock

class FinelipTextEncoder(nn.Module):
    """Finelip 自定义的文本编码器：复用 CLIP 预训练权重，带 ALiBi 偏置"""
    def __init__(self, clip_model_name: str = "ViT-L/14", alibi_start_alpha: float = 1.0):
        super().__init__()
        # 1. 加载 CLIP 预训练模型（仅用于获取权重，不修改其源码）
        self.clip_model, _ = clip.load(clip_model_name, device="cpu")
        self.d_model = self.clip_model.transformer.width  # CLIP 文本特征维度（如 768）
        self.n_heads = self.clip_model.transformer.heads  # CLIP 注意力头数（如 12）
        self.n_layers = len(self.clip_model.transformer.resblocks)  # CLIP Transformer 层数（如 12）

        # 2. 复用 CLIP 的核心组件（直接复制，不修改）
        self.token_embedding = self.clip_model.token_embedding  # 词嵌入层
        self.positional_embedding = self.clip_model.positional_embedding  # 原始绝对位置嵌入
        self.ln_final = self.clip_model.ln_final  # 输出层归一化
        self.text_projection = self.clip_model.text_projection  # 文本投影层

        # 3. 构建 Finelip 带 ALiBi 的 Transformer 层（替换 CLIP 原始层）
        self.transformer = nn.ModuleList([
            FinelipResidualAttentionBlock(
                d_model=self.d_model,
                n_head=self.n_heads,
                alibi_start_alpha=alibi_start_alpha
            ) for _ in range(self.n_layers)
        ])

        # 4. 从 CLIP 加载预训练权重（忽略 ALiBi 新增的参数）
        self._load_clip_weights()

    def _load_clip_weights(self):
        """从 CLIP 模型加载权重到 Finelip 文本编码器（仅复用相同结构的参数）"""
        # 加载 Transformer 每一层的权重
        for finelip_block, clip_block in zip(self.transformer, self.clip_model.transformer.resblocks):
            # 加载层归一化权重（ln_1、ln_2）
            finelip_block.ln_1.load_state_dict(clip_block.ln_1.state_dict())
            finelip_block.ln_2.load_state_dict(clip_block.ln_2.state_dict())
            # 加载 MLP 权重
            finelip_block.mlp.load_state_dict(clip_block.mlp.state_dict())
            # 加载注意力层的 QKV 和 proj 权重（ALiBi 的 alibi_alpha 不加载，后续训练）
            finelip_block.attn.qkv.load_state_dict(clip_block.attn.qkv.state_dict())
            finelip_block.attn.proj.load_state_dict(clip_block.attn.proj.state_dict())
            # （可选）加载交叉注意力权重（若有）
            if finelip_block.cross_attn and clip_block.cross_attn:
                finelip_block.cross_attn.qkv.load_state_dict(clip_block.cross_attn.qkv.state_dict())
                finelip_block.cross_attn.proj.load_state_dict(clip_block.cross_attn.proj.state_dict())

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """与 CLIP 文本编码器输出格式完全一致，确保 Finelip 其他模块兼容"""
        B, T = text.shape
        # 1. 词嵌入 + 原始绝对位置嵌入（与 CLIP 一致）
        x = self.token_embedding(text)  # [B, T, d_model]
        x = x + self.positional_embedding[:T, :]  # 取前 T 个位置嵌入（支持任意长度）

        # 2. 带 ALiBi 的 Transformer 编码（Finelip 自定义）
        attn_mask = self.clip_model.transformer.attn_mask  # 复用 CLIP 掩码
        if attn_mask is not None and attn_mask.shape[1] != T:
            # 适配任意长度文本的掩码（若 CLIP 掩码长度固定，扩展为当前文本长度）
            attn_mask = torch.zeros((1, T, T), device=x.device, dtype=torch.float32)
            attn_mask[:, :, :] = -torch.finfo(x.dtype).max
            attn_mask[:, :, :T] = 0.0  # 仅有效文本区域无掩码

        for block in self.transformer:
            x = block(x, attn_mask)

        # 3. 输出层（与 CLIP 一致）
        x = self.ln_final(x)  # [B, T, d_model]
        # 取 CLS  token（或按 Finelip 需求返回全序列）
        x = x @ self.text_projection  # [B, T, embed_dim]
        return x

    def encode_text_full(self, text: torch.Tensor) -> torch.Tensor:
        """Finelip 常用接口：返回完整序列的文本嵌入（含所有 token，用于跨模态融合）"""
        return self.forward(text)  # [B, T, d_model]

    def encode_text_cls(self, text: torch.Tensor) -> torch.Tensor:
        """Finelip 常用接口：仅返回 CLS token 嵌入（用于全局对齐）"""
        full_emb = self.forward(text)  # [B, T, d_model]
        return full_emb[:, 0, :]  # [B, d_model]