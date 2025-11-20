import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from pkg_resources import packaging
from torch import nn
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
import numpy as np

from .model_finelip import build_model,build_model_2
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["load", "tokenize"]
_tokenizer = _Tokenizer()


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])



# def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None, run_finelip=False):
#     """Load a long CLIP model
#
#     Parameters
#     ----------
#     name : str
#         A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
#
#     device : Union[str, torch.device]
#         The device to put the loaded model
#
#     Returns
#     -------
#     model : torch.nn.Module
#         The CLIP model
#
#     preprocess : Callable[[PIL.Image], torch.Tensor]
#         A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
#     """
#
#     model_path = name
#
#     state_dict = torch.load(model_path, map_location="cpu")
#
#     model = build_model(state_dict or model.state_dict(), load_from_clip = False, run_finelip=run_finelip).to(device)
#
#     if str(device) == "cpu":
#         model.float()
#
#     return model, _transform(model.visual.input_resolution)
#
#
#
#     def _node_get(node: torch._C.Node, key: str):
#         """Gets attributes of a node which is polymorphic over return type.
#
#         From https://github.com/pytorch/pytorch/pull/82628
#         """
#         sel = node.kindOf(key)
#         return getattr(node, sel)(key)
#
#     def patch_device(module):
#         try:
#             graphs = [module.graph] if hasattr(module, "graph") else []
#         except RuntimeError:
#             graphs = []
#
#         if hasattr(module, "forward1"):
#             graphs.append(module.forward1.graph)
#
#         for graph in graphs:
#             for node in graph.findAllNodes("prim::Constant"):
#                 if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
#                     node.copyAttributes(device_node)
#
#     model.apply(patch_device)
#     patch_device(model.encode_image)
#     patch_device(model.encode_text)
#
#     # patch dtype to float32 on CPU
#     if str(device) == "cpu":
#         float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
#         float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
#         float_node = float_input.node()
#
#         def patch_float(module):
#             try:
#                 graphs = [module.graph] if hasattr(module, "graph") else []
#             except RuntimeError:
#                 graphs = []
#
#             if hasattr(module, "forward1"):
#                 graphs.append(module.forward1.graph)
#
#             for graph in graphs:
#                 for node in graph.findAllNodes("aten::to"):
#                     inputs = list(node.inputs())
#                     for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
#                         if _node_get(inputs[i].node(), "value") == 5:
#                             inputs[i].node().copyAttributes(float_node)
#
#         model.apply(patch_float)
#         patch_float(model.encode_image)
#         patch_float(model.encode_text)
#
#         model.float()
#
#     return model, _transform(model.input_resolution.item())
#
#
# def load_from_clip(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None, run_finelip=False):
#     """Load from CLIP model for fine-tuning
#
#     Parameters
#     ----------
#     name : str
#         A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
#
#     device : Union[str, torch.device]
#         The device to put the loaded model
#
#     jit : bool
#         Whether to load the optimized JIT model or more hackable non-JIT model (default).
#
#     download_root: str
#         path to download the model files; by default, it uses "~/.cache/clip"
#
#     Returns
#     -------
#     model : torch.nn.Module
#         The CLIP model
#
#     preprocess : Callable[[PIL.Image], torch.Tensor]
#         A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
#     """
#
#     _MODELS = {
#     "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
#     "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
#     "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
#     "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
#     "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
#     "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
#     "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",##
#     "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
#     "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
#
#     }
#
#     def available_models() -> List[str]:
#         """Returns the names of available CLIP models"""
#         return list(_MODELS.keys())
#
#     def _download(url: str, root: str):
#         os.makedirs(root, exist_ok=True)
#         filename = os.path.basename(url)
#
#         expected_sha256 = url.split("/")[-2]
#         download_target = os.path.join(root, filename)
#
#         if os.path.exists(download_target) and not os.path.isfile(download_target):
#             raise RuntimeError(f"{download_target} exists and is not a regular file")
#
#         if os.path.isfile(download_target):
#             if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
#                 return download_target
#             else:
#                 warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
#
#         with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
#             with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
#                 while True:
#                     buffer = source.read(8192)
#                     if not buffer:
#                         break
#
#                     output.write(buffer)
#                     loop.update(len(buffer))
#
#         if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
#             raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")
#
#         return download_target
#
#     if name in _MODELS:
#         model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
#     elif os.path.isfile(name):
#         model_path = name
#     else:
#         raise RuntimeError(f"Model {name} not found; available models = {available_models()}")
#
#     with open(model_path, 'rb') as opened_file:
#         try:
#             # loading JIT archive
#             model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
#             state_dict = None
#         except RuntimeError:
#             # loading saved state dict
#             if jit:
#                 warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
#                 jit = False
#             state_dict = torch.load(opened_file, map_location="cpu")
#
#     model = build_model(state_dict or model.state_dict(), load_from_clip = True, run_finelip=run_finelip).to(device)
#
#     positional_embedding_pre = model.positional_embedding.type(model.dtype)
#
#     length, dim = positional_embedding_pre.shape
#     keep_len = 20
#     posisitonal_embedding_new = torch.zeros([4*length-3*keep_len, dim], dtype=model.dtype)
#     for i in range(keep_len):
#         posisitonal_embedding_new[i] = positional_embedding_pre[i]
#     for i in range(length-1-keep_len):
#         posisitonal_embedding_new[4*i + keep_len] = positional_embedding_pre[i + keep_len]
#         posisitonal_embedding_new[4*i + 1 + keep_len] = 3*positional_embedding_pre[i + keep_len]/4 + 1*positional_embedding_pre[i+1+keep_len]/4
#         posisitonal_embedding_new[4*i + 2+keep_len] = 2*positional_embedding_pre[i+keep_len]/4 + 2*positional_embedding_pre[i+1+keep_len]/4
#         posisitonal_embedding_new[4*i + 3+keep_len] = 1*positional_embedding_pre[i+keep_len]/4 + 3*positional_embedding_pre[i+1+keep_len]/4
#
#     posisitonal_embedding_new[4*length -3*keep_len - 4] = positional_embedding_pre[length-1] + 0*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
#     posisitonal_embedding_new[4*length -3*keep_len - 3] = positional_embedding_pre[length-1] + 1*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
#     posisitonal_embedding_new[4*length -3*keep_len - 2] = positional_embedding_pre[length-1] + 2*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
#     posisitonal_embedding_new[4*length -3*keep_len - 1] = positional_embedding_pre[length-1] + 3*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
#
#     positional_embedding_res = posisitonal_embedding_new.clone()
#
#     model.positional_embedding = nn.Parameter(posisitonal_embedding_new, requires_grad=False)
#     model.positional_embedding_res = nn.Parameter(positional_embedding_res, requires_grad=True)
#
#     if str(device) == "cpu":
#         model.float()
#     return model, _transform(model.visual.input_resolution)
#
#     def _node_get(node: torch._C.Node, key: str):
#         """Gets attributes of a node which is polymorphic over return type.
#
#         From https://github.com/pytorch/pytorch/pull/82628
#         """
#         sel = node.kindOf(key)
#         return getattr(node, sel)(key)
#
#     def patch_device(module):
#         try:
#             graphs = [module.graph] if hasattr(module, "graph") else []
#         except RuntimeError:
#             graphs = []
#
#         if hasattr(module, "forward1"):
#             graphs.append(module.forward1.graph)
#
#         for graph in graphs:
#             for node in graph.findAllNodes("prim::Constant"):
#                 if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
#                     node.copyAttributes(device_node)
#
#     model.apply(patch_device)
#     patch_device(model.encode_image)
#     patch_device(model.encode_text)
#
#     # patch dtype to float32 on CPU
#     if str(device) == "cpu":
#         float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
#         float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
#         float_node = float_input.node()
#
#         def patch_float(module):
#             try:
#                 graphs = [module.graph] if hasattr(module, "graph") else []
#             except RuntimeError:
#                 graphs = []
#
#             if hasattr(module, "forward1"):
#                 graphs.append(module.forward1.graph)
#
#             for graph in graphs:
#                 for node in graph.findAllNodes("aten::to"):
#                     inputs = list(node.inputs())
#                     for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
#                         if _node_get(inputs[i].node(), "value") == 5:
#                             inputs[i].node().copyAttributes(float_node)
#
#         model.apply(patch_float)
#         patch_float(model.encode_image)
#         patch_float(model.encode_text)
#
#         model.float()
#
#     return model, _transform(model.input_resolution.item())


def load(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    download_root: str = None,
    run_finelip=False,
    use_ot: bool = False,  # 是否使用带OT和融合的ALiBi模型
    fusion_num_heads: int = 8,  # 融合模块参数（use_ot=True时有效）
    ot_epsilon: float = 1e-2,   # OT损失参数（use_ot=True时有效）
    pos_embed_mode: str = "linear"  # 位置嵌入模式（use_ot=True时有效）
):
    """Load a long CLIP model, with option to use OT and cross-modal fusion"""
    # 加载本地模型权重（state_dict）
    model_path = name
    state_dict = torch.load(model_path, map_location="cpu")  # 从本地文件加载state_dict

    # 根据use_ot分支构建模型
    if use_ot:
        # 分支1：使用CLIPWithFusionAndOT模型（ALiBi+OT）
        model = build_model_2(
            state_dict=state_dict,  # 仅使用加载的state_dict（删除未定义的model.state_dict()）
            load_from_clip=False,
            device=device,  # 传递设备参数（需确保build_model_2已支持）

            fusion_num_heads=fusion_num_heads,
            ot_epsilon=ot_epsilon
        )

        # 获取视觉输入分辨率（适配CLIPWithFusionAndOT的属性）
        input_resolution = model.input_resolution.item() if hasattr(model, "input_resolution") else 224
    else:
        # 分支2：使用原始CLIP模型
        model = build_model(
            state_dict=state_dict,  # 仅使用加载的state_dict
            load_from_clip=False,
            run_finelip=run_finelip
        ).to(device)
        # 获取视觉输入分辨率（适配原始模型的属性）
        input_resolution = model.input_resolution.item() if hasattr(model, "input_resolution") else 224

    # ---------------------- 设备和数据类型适配逻辑（移到return之前，确保执行） ----------------------
    # 定义JIT节点属性获取函数（内部使用）
    def _node_get(node: torch._C.Node, key: str):
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    # 设备适配补丁（针对JIT模型）
    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []
        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)
        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    # 创建设备节点并复制属性
                    device_node = torch.jit.trace(lambda: torch.ones([]).to(device), example_inputs=[]).graph.findNode("prim::Constant")
                    node.copyAttributes(device_node)

    # 应用设备补丁
    model.apply(patch_device)
    if hasattr(model, "encode_image"):
        patch_device(model.encode_image)
    if hasattr(model, "encode_text"):
        patch_device(model.encode_text)

    # CPU数据类型适配（浮点类型转换）
    if str(device) == "cpu":
        # 创建浮点类型节点
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        # 浮点类型补丁
        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)
            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:
                        if _node_get(inputs[i].node(), "value") == 5:  # 5对应float32
                            inputs[i].node().copyAttributes(float_node)

        # 应用浮点补丁
        model.apply(patch_float)
        if hasattr(model, "encode_image"):
            patch_float(model.encode_image)
        if hasattr(model, "encode_text"):
            patch_float(model.encode_text)
        model.float()  # 转为float32

    # 返回模型和预处理函数（使用正确的输入分辨率）
    return model, _transform(input_resolution)

# def verify_spectral_matching(P_original, P_new, Σ_original, r):
#     """验证新嵌入与原始嵌入的谱匹配度"""
#     # 计算新嵌入的奇异值谱
#     _, Σ_new, _ = torch.linalg.svd(P_new, full_matrices=False)
#     # 谱余弦相似度（越高越好）
#     pu_sim = torch.nn.functional.cosine_similarity(Σ_original[:r], Σ_new[:r], dim=0).item()
#     print(f"谱匹配度：前{r}个奇异值余弦相似度 = {pu_sim:.4f}（目标>0.95）")
#
#
# def verify_qk_statistics(model, pos_orig, pos_new, device):
#     """最终修正：适配PyTorch原生MultiheadAttention的权重结构"""
#     # 获取注意力层（以第一个resblock的注意力为例）
#     attn = model.transformer.resblocks[0].attn  # 原生MultiheadAttention对象
#
#     # 原生MultiheadAttention中，Q/K/V的合并投影权重存储在in_proj_weight
#     # 形状：[3*embed_dim, embed_dim]，前1/3是Q，中间1/3是K，最后1/3是V
#     embed_dim = pos_orig.shape[1]  # 嵌入维度（如768）
#     in_proj_weight = attn.in_proj_weight  # [3*768, 768]
#
#     # 拆分Q和K的投影权重
#     q_weight = in_proj_weight[:embed_dim, :]  # 前768行：Q的权重 [768, 768]
#     k_weight = in_proj_weight[embed_dim:2 * embed_dim, :]  # 中间768行：K的权重 [768, 768]
#
#     # 转置为[embed_dim, embed_dim]（与Q=E·W_Q的矩阵乘法逻辑一致）
#     W_Q = q_weight.T.to(device=device, dtype=pos_orig.dtype)
#     W_K = k_weight.T.to(device=device, dtype=pos_orig.dtype)
#
#     # 计算QK矩阵（后续逻辑不变）
#     Q_orig = torch.matmul(pos_orig, W_Q)
#     K_orig = torch.matmul(pos_orig, W_K)
#     QK_orig = torch.matmul(Q_orig, K_orig.T) / (embed_dim ** 0.5)  # 除以sqrt(dim)
#
#     # 新嵌入的QK矩阵（取前77个位置对比）
#     pos_new_77 = pos_new[:77] if pos_new.shape[0] >= 77 else pos_new
#     Q_new = torch.matmul(pos_new_77, W_Q)
#     K_new = torch.matmul(pos_new_77, W_K)
#     QK_new = torch.matmul(Q_new, K_new.T) / (embed_dim ** 0.5)
#
#     # 输出统计差异
#     mean_diff = torch.abs(QK_orig.mean() - QK_new.mean()).item()
#     var_diff = torch.abs(QK_orig.var() - QK_new.var()).item()
#     fro_diff = torch.norm(QK_orig - QK_new, p='fro') / torch.norm(QK_orig, p='fro')
#     print(f"QK统计差异：均值差={mean_diff:.6f}, 方差差={var_diff:.6f}, 相对F范数差={fro_diff:.4f}")
#
#
#
# def check_redundant_dims(positional_embedding_pre, top_k=77):
#     """
#     验证前77个位置嵌入是否存在冗余维度（修复float16不支持问题）
#     """
#     # 取前77个嵌入，先转为float32（NumPy支持），再转NumPy数组
#     emb = positional_embedding_pre[:top_k].cpu().detach().to(dtype=torch.float32).numpy()  # 关键修改：转为float32
#
#     # 中心化（去除均值，避免影响SVD）
#     emb_centered = emb - np.mean(emb, axis=0, keepdims=True)
#
#     # SVD分解：此时emb_centered是float32，NumPy支持
#     _, singular_values, _ = np.linalg.svd(emb_centered)
#
#     # 计算秩：奇异值大于阈值（如1e-5）的数量
#     threshold = 1e-5
#     rank = np.sum(singular_values > threshold)
#
#     print(f"前77个位置嵌入的秩：{rank}（维度dim={emb.shape[1]}）")
#     print(f"前10个奇异值：{singular_values[:10].round(3)}")
#     print(f"最后10个奇异值：{singular_values[-10:].round(6)}")
#
#     return rank, singular_values
# def verify_ortho_property(orig_emb, new_emb, idx1, idx2):
#     """检查原始嵌入对与新嵌入对的内积是否接近"""
#     orig_dot = torch.dot(orig_emb[idx1], orig_emb[idx2])
#     new_dot = torch.dot(new_emb[idx1], new_emb[idx2])
#     print(f"原始内积: {orig_dot:.4f}, 新内积: {new_dot:.4f}, 差异: {abs(orig_dot - new_dot):.6f}")
#
#
# def slerp(p0, p1, t, eps=1e-7):
#     """球面线性插值：在p0和p1之间按比例t插值"""
#     # 计算两向量的夹角余弦（内积除以模长乘积）
#     p0_norm = torch.norm(p0, dim=-1, keepdim=True)
#     p1_norm = torch.norm(p1, dim=-1, keepdim=True)
#     cos_theta = torch.sum(p0 * p1, dim=-1, keepdim=True) / (p0_norm * p1_norm + eps)
#     # 限制cos_theta在[-1,1]（避免数值误差）
#     cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
#     theta = torch.arccos(cos_theta)  # 夹角
#
#     # 计算插值系数
#     sin_theta = torch.sin(theta) + eps  # 避免除零
#     sin_t_theta = torch.sin(t * theta)
#     sin_1t_theta = torch.sin((1 - t) * theta)
#
#     # 生成插值向量
#     return (sin_1t_theta / sin_theta) * p0 + (sin_t_theta / sin_theta) * p1

def load_from_clip(
        name: str,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        jit: bool = False,
        download_root: str = None,
        run_finelip=False,
        use_ot: bool = False,  # 新增参数：是否使用带OT和融合的ALiBi模型
        fusion_num_heads: int = 8,  # 融合模块参数（use_ot=True时有效）
        ot_epsilon: float = 1e-2,  # OT损失参数（use_ot=True时有效）
        pos_embed_mode: str = "linear"  # 位置嵌入模式（use_ot=False时有效）
):
    """Load from CLIP model for fine-tuning, with option to use OT and cross-modal fusion"""
    _MODELS = {
        "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
        "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
        "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
        "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
        "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
        "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
        "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
        "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
    }

    def available_models() -> List[str]:
        return list(_MODELS.keys())

    def _download(url: str, root: str):
        os.makedirs(root, exist_ok=True)
        filename = os.path.basename(url)
        expected_sha256 = url.split("/")[-2]
        download_target = os.path.join(root, filename)
        if os.path.exists(download_target) and not os.path.isfile(download_target):
            raise RuntimeError(f"{download_target} exists and is not a regular file")
        if os.path.isfile(download_target):
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
                return download_target
            else:
                warnings.warn(f"{download_target} checksum mismatch; re-downloading")
        with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
            with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                      unit_divisor=1024) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break
                    output.write(buffer)
                    loop.update(len(buffer))
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
            raise RuntimeError("Model checksum mismatch")
        return download_target

    # ---------------------- 1. 下载/加载CLIP模型文件 ----------------------
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # 尝试加载JIT模型
            model_jit = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as state dict")
                jit = False
            # 加载为state dict（非JIT模式）
            state_dict = torch.load(opened_file, map_location="cpu")




    # ---------------------- 原始CLIP模型 ----------------------

    model = build_model(
        state_dict=state_dict or model_jit.state_dict(),
        load_from_clip=True,
        run_finelip=run_finelip
    ).to(device)
    #_______________________________________

    # ---------------------- 2. 正交变化插值 ----------------------




    # ---------------------- 保留原始位置嵌入扩展逻辑 ----------------------
    positional_embedding_pre = model.positional_embedding.type(model.dtype)
    length, dim = positional_embedding_pre.shape
    keep_len = 20
    posisitonal_embedding_new = torch.zeros([4 * length - 3 * keep_len, dim], dtype=model.dtype)

    # 位置嵌入插值扩展（原始逻辑不变）
    for i in range(keep_len):
        posisitonal_embedding_new[i] = positional_embedding_pre[i]
    for i in range(length - 1 - keep_len):
        posisitonal_embedding_new[4 * i + keep_len] = positional_embedding_pre[i + keep_len]
        posisitonal_embedding_new[4 * i + 1 + keep_len] = 3 * positional_embedding_pre[i + keep_len] / 4 + 1 * \
                                                          positional_embedding_pre[i + 1 + keep_len] / 4
        posisitonal_embedding_new[4 * i + 2 + keep_len] = 2 * positional_embedding_pre[i + keep_len] / 4 + 2 * \
                                                          positional_embedding_pre[i + 1 + keep_len] / 4
        posisitonal_embedding_new[4 * i + 3 + keep_len] = 1 * positional_embedding_pre[i + keep_len] / 4 + 3 * \
                                                          positional_embedding_pre[i + 1 + keep_len] / 4
    posisitonal_embedding_new[4 * length - 3 * keep_len - 4] = positional_embedding_pre[length - 1] + 0 * (
                positional_embedding_pre[length - 1] - positional_embedding_pre[length - 2]) / 4
    posisitonal_embedding_new[4 * length - 3 * keep_len - 3] = positional_embedding_pre[length - 1] + 1 * (
                positional_embedding_pre[length - 1] - positional_embedding_pre[length - 2]) / 4
    posisitonal_embedding_new[4 * length - 3 * keep_len - 2] = positional_embedding_pre[length - 1] + 2 * (
                positional_embedding_pre[length - 1] - positional_embedding_pre[length - 2]) / 4
    posisitonal_embedding_new[4 * length - 3 * keep_len - 1] = positional_embedding_pre[length - 1] + 3 * (
                positional_embedding_pre[length - 1] - positional_embedding_pre[length - 2]) / 4
    positional_embedding_res = posisitonal_embedding_new.clone()

    # 更新模型的位置嵌入（原始逻辑不变）
    model.positional_embedding = nn.Parameter(posisitonal_embedding_new, requires_grad=False)
    model.positional_embedding_res = nn.Parameter(positional_embedding_res, requires_grad=True)

    # ---------------------- 保留原始JIT设备和数据类型适配逻辑 ----------------------
    def _node_get(node: torch._C.Node, key: str):
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []
        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)
        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    device_node = torch.jit.trace(lambda: torch.ones([]).to(device),
                                                  example_inputs=[]).graph.findNode("prim::Constant")
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # CPU数据类型适配（原始逻辑不变）
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)
            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()

    # 返回原始模型和预处理
    return model, _transform(model.visual.input_resolution)




def tokenize(texts: Union[str, List[str]], context_length: int = 77*4-60, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


