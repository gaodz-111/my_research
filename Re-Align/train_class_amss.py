
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalClassifier(nn.Module):
    def __init__(self, input_dim, proj_dim, num_classes):
        super().__init__()
        self.proj_q = nn.Linear(input_dim, proj_dim)
        self.proj_v = nn.Linear(input_dim, proj_dim)
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, q_tokens, v_tokens, attn_qv, attn_vq):
        """
        q_tokens: [B, T_q, D]  文本 token 表示
        v_tokens: [B, T_v, D]  图像 token 表示
        attn_qv: [B, T_q, T_v] 文到图注意力
        attn_vq: [B, T_v, T_q] 图到文注意力
        """
        # 加权求和得到跨模态表示
        h_q = torch.einsum("bqt,bqd->btd", attn_qv, v_tokens).mean(1)  # [B, D]
        h_v = torch.einsum("bvt,bvd->btd", attn_vq, q_tokens).mean(1)  # [B, D]

        # 投影到统一空间
        h_q = self.proj_q(h_q)
        h_v = self.proj_v(h_v)

        # 融合表示
        h = (h_q + h_v) / 2

        # 分类预测
        logits = self.classifier(h)
        return logits

def find_spans(input_ids_row: torch.Tensor, image_token_id: int, assistant_token_ids: list, img_len_fallback: int = 577):
    image_pos_tensor = (input_ids_row == image_token_id).nonzero(as_tuple=False)
    if image_pos_tensor.numel() == 0:
        raise ValueError("未找到 <image> token")
    image_pos = int(image_pos_tensor[0,0].item())
    L = input_ids_row.size(0)
    seq_len = len(assistant_token_ids)
    start_assist = None
    for i in range(image_pos, L - seq_len + 1):
        if input_ids_row[i:i+seq_len].tolist() == assistant_token_ids:
            # 保持原有逻辑（脚本中使用了 +576 的偏移），这里维持返回的 q_end 语义与原代码一致
            start_assist = i + 576
            break
    if start_assist is None:
        start_assist = L
    img_len = img_len_fallback
    q_start = image_pos + img_len
    q_end = start_assist
    if q_end <= q_start:
        q_end = q_start + 1
    return image_pos, img_len, q_start, q_end


class AMSSClassifier(nn.Module):
    def __init__(self, text_dim, img_dim, hidden_dim=1024, proj_dim=512, num_classes=len(LABELS),
                 dropout=0.1, num_heads=4, num_layers=4):
                     super().__init__()
                     self.img_proj = nn.Linear(img_dim, proj_dim)
    def forward_backward(self, batch)
        device = next(self.model.parameters()).device
        # ensure cls_head on same device
        self.cls_head.to(device)

        input_ids = batch["input_ids"].to(device)
        images = batch["image"].to(device, dtype=torch.float16)
        labels = input_ids.clone()
        labels[input_ids == IMAGE_TOKEN_INDEX] = -100
        pad_id = self.tokenizer.pad_token_id or 0
        labels[input_ids == pad_id] = -100

        # forward: request hidden_states & attentions
        outputs = self.model(input_ids=input_ids, images=images, labels=labels,
                             output_attentions=True, output_hidden_states=True, return_dict=True)
        lm_loss = outputs.loss

        # --- compute attention mean ---
        attn_stack = torch.stack(outputs.attentions, dim=0)  # [layers, B, heads, L, L]
        attn_head_mean = attn_stack.mean(dim=2)  # [layers, B, L, L]
        if self.tga_layers.startswith("last_"):
            k = int(self.tga_layers.split("_")[1])
            attn_mean = attn_head_mean[-k:].mean(dim=0)  # [B, L, L]
        else:
            attn_mean = attn_head_mean.mean(dim=0)

        B = input_ids.size(0)

        # collect indices for embedding extraction
        image_pos_list = []
        qend_list = []
        for b in range(B):
            ids_row = input_ids[b]
            assistant_token_ids = [319, 1799, 9047, 13566, 29901]
            try:
                image_pos, img_len, q_start, q_end = find_spans(ids_row, image_token_id, assistant_token_ids)
            except Exception:
                q_start, q_end = 0, ids_row.size(0)
                image_pos = 0

            image_pos_list.append(image_pos)
            qend_list.append(q_end)

            qa_to_image = attn_mean[b, q_end - len_q:q_end, image_pos:image_pos + img_len]  # [q_len, img_and_prompt_len]
            image_to_qa = attn_mean[b, image_pos:image_pos + img_len,q_end - len_q:q_end]

