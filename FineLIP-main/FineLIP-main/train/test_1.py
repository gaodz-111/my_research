import torch
from tqdm import tqdm
import sys
import os
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append("..")

from model import finelip
import torch.nn.functional as F  # 补充必要导入

from sharegpt4v import share4v_val_dataset
from eval_data import urban_dataset, coco_dataset, flickr_dataset, docci_dataset

import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class CLIP_Clean_Train():
    def __init__(self, local_rank, args):
        self.local_rank = local_rank
        self.test_data = args.test_data
        self.args = args

        self.model, self.preprocess = finelip.load(args.ckpt_path, device='cpu', run_finelip=args.run_finelip,
                                                   use_ot=args.use_ot)

        torch.cuda.set_device(device=f'cuda:{local_rank}')
        self.model = self.model.float().cuda()

        self.finegrain = args.finegrain
        self.max_valid_len = 248  # 与训练时一致
        self.feature_dim = 768  # 固定特征维度（与训练时一致）

    @torch.no_grad()
    def test_epoch(self):
        all_image_features, all_text_features = [], []
        self.all_finegrain_image_emb = []  # 3维：[N_total, 257, 768]
        self.all_finegrain_text_fused = []  # 3维：[N_total, 248, 768]
        self.all_lengths = []  # 1维：[N_total]

        for id, (images, text) in enumerate(tqdm(self.testloader)):
            images = images.cuda()
            B_img = images.shape[0]
            # 图像特征（3维）
            image_features_full = self.model.encode_image_full(images)
            # 多尺度图像特征（3维）
            img_low_global, img_high_global, _, _ = self.model.visual.extract_multi_scale_feats(
                images.type(self.model.dtype))
            img_low_seq = img_low_global.unsqueeze(1)
            img_high_seq = img_high_global.unsqueeze(1)

            # 基础检索：图像CLS特征（2维）
            image_features = image_features_full[:, 0, :]
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # 文本特征收集
            temp_text_features = []
            temp_text_features_full = []
            temp_lengths = []
            for cap_list in text:
                text_token = finelip.tokenize(cap_list, context_length=248, truncate=True).cuda()
                lengths = [torch.nonzero(text_token[i]).size(0) for i in range(text_token.shape[0])]
                # 文本特征：3维（[B_text, 248, 768]）
                text_features_full = self.model.encode_text_full(text_token) @ self.model.text_projection
                # 文本CLS特征：2维（[B_text, 768]）
                text_features = text_features_full[torch.arange(text_features_full.shape[0]), text_token.argmax(dim=-1)]
                text_features /= text_features.norm(dim=-1, keepdim=True)

                temp_text_features.append(text_features)
                temp_text_features_full.append(text_features_full)
                temp_lengths.append(torch.tensor(lengths, device=text_token.device))

            # 拼接文本特征（合并batch维度，确保3维/1维）
            batch_text_features = torch.cat(temp_text_features, dim=0)  # 2维：[B_total, 768]
            batch_text_features_full = torch.cat(temp_text_features_full, dim=0)  # 3维：[B_total, 248, 768]
            batch_lengths = torch.cat(temp_lengths, dim=0)  # 1维：[B_total]
            # 基础检索文本特征调整
            text_features = batch_text_features.reshape(-1, batch_text_features.shape[-1])  # 2维：[B_total, 768]

            if self.finegrain and self.model.run_finelip and self.model.support_multi_scale:
                # -------------------------- 核心修复：维度修正 --------------------------
                text_cls_included_list = []
                for i in range(batch_text_features_full.shape[0]):
                    valid_len = batch_lengths[i].item() if (i < len(batch_lengths) and batch_lengths.dim() > 0) else \
                    batch_text_features_full.shape[1]
                    # 1. 提取text_i并挤压冗余维度（4维→3维）
                    text_i = batch_text_features_full[i:i + 1, :valid_len, :]  # 可能是(1,1,248,768)→挤压后(1,248,768)

                    text_i = text_i.squeeze(1) if text_i.dim() == 4 else text_i  # 关键：删除冗余的第1维




                    # 2. 生成正确维度的pad（3维：[1, pad_len, 768]）
                    if text_i.shape[1] < self.max_valid_len:
                        pad_len = self.max_valid_len - text_i.shape[1]
                        pad = torch.zeros(
                            1,  # 与text_i第0维一致
                            pad_len,  # 需补充的长度
                            self.feature_dim,  # 与特征维度一致（768），而非序列长度（248）
                            device=text_i.device,
                            dtype=text_i.dtype
                        )
                        text_i = torch.cat([text_i, pad], dim=1)  # 3维+3维=3维，无报错

                    text_cls_included_list.append(text_i)
                text_cls_included = torch.cat(text_cls_included_list, dim=0)  # 3维：[B_total, 248, 768]

                # 文本融合（与训练一致，3维输入/输出）
                text_coarse = self.model.coarse_fusion[0](text_cls_included, img_low_seq.repeat_interleave(
                    text_cls_included.shape[0] // B_img, dim=0))
                text_coarse = self.model.coarse_fusion[1](text_coarse)
                text_coarse = F.normalize(text_coarse, dim=-1)

                text_fine = self.model.fine_fusion[0](text_coarse, img_high_seq.repeat_interleave(
                    text_cls_included.shape[0] // B_img, dim=0))
                text_fine = self.model.fine_fusion[1](text_fine)
                text_fine = F.normalize(text_fine, dim=-1)

                text_fused = self.model.fuse_text_features(text_cls_included, text_coarse,
                                                           text_fine)  # 3维：[B_total, 248, 768]

                # 存储细粒度特征（3维/1维）
                self.all_finegrain_image_emb.append(
                    image_features_full.repeat_interleave(text_cls_included.shape[0] // B_img, dim=0))
                self.all_finegrain_text_fused.append(text_fused.cpu())
                self.all_lengths.append(batch_lengths.cpu())

            # 存储基础检索特征
            all_image_features.append(image_features.repeat_interleave(batch_text_features.shape[0] // B_img, dim=0))
            all_text_features.append(text_features)

        # 拼接所有基础特征
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)
        self.logits_per_image = (all_image_features @ all_text_features.t()).detach().cpu()
        self.logits_per_text = self.logits_per_image.t()

        # 细粒度logits计算（3维输入）
        if self.finegrain and self.model.run_finelip and self.model.support_multi_scale:
            self.all_finegrain_image_emb = torch.cat(self.all_finegrain_image_emb, dim=0).cuda()
            self.all_finegrain_text_fused = torch.cat(self.all_finegrain_text_fused, dim=0).cuda()
            self.all_lengths = torch.cat(self.all_lengths, dim=0).cuda()

            finegrain_logits_per_image = torch.empty_like(self.logits_per_image).cuda()
            chunk_size = 100
            for i in tqdm(range(0, self.all_finegrain_text_fused.shape[0], chunk_size)):
                end_idx = min(i + chunk_size, self.all_finegrain_text_fused.shape[0])
                improved_sims = self.model.cross_net.forward_dual_aggr(
                    self.all_finegrain_image_emb,
                    self.all_finegrain_text_fused[i:end_idx],
                    self.all_lengths[i:end_idx]
                )
                finegrain_logits_per_image[:, i:end_idx] = improved_sims.detach()

            finegrain_logits_per_text = finegrain_logits_per_image.t()
            alpha = 0.8
            self.logits_per_image = (
                        alpha * self.logits_per_image.cuda() + (1 - alpha) * finegrain_logits_per_image).cpu()
            self.logits_per_text = (alpha * self.logits_per_text.cuda() + (1 - alpha) * finegrain_logits_per_text).cpu()

        # 计算指标
        if self.test_data in ['share', 'urban', 'docci']:
            self.ground_truth = torch.arange(len(all_text_features)).view(-1, 1)
            results = self.get_metrics()
        else:
            results = self.get_metrics_1v5()

        return results

    def test(self):
        self.model.eval()
        if self.test_data == 'share':
            testset = share4v_val_dataset(self.preprocess)
        elif self.test_data == 'urban':
            testset = urban_dataset(self.preprocess)
        elif self.test_data == 'coco':
            testset = coco_dataset(self.preprocess)
        elif self.test_data == 'flickr':
            testset = flickr_dataset(self.preprocess)
        elif self.test_data == 'docci':
            testset = docci_dataset(self.preprocess)

        self.testloader = DataLoader(testset, batch_size=200, num_workers=8, pin_memory=True)
        with torch.no_grad():
            metrics = self.test_epoch()
            print("=====================================")
            print(f"test mean of {self.test_data} retrieval")
            for k, v in metrics.items():
                if "@" in k:
                    print(f"{k} {format(v, '.4f')}")
            print("=====================================")

        return

    def save_logits(self, logits, name):
        if not os.path.exists(os.path.dirname(name)):
            os.makedirs(os.path.dirname(name))
        np.save(name, logits.numpy())

    def get_metrics(self):
        metrics = {}
        logits = {"image_to_text": self.logits_per_image, "text_to_image": self.logits_per_text}

        for name, logit in logits.items():
            ranking = torch.argsort(logit, descending=True)
            preds = torch.where(ranking == self.ground_truth)[1]
            preds = preds.detach().cpu().numpy()
            metrics[f"{name}_mean_rank"] = preds.mean() + 1
            metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
            for k in [1, 5, 10]:
                metrics[f"{name}_R@{k}"] = np.mean(preds < k)
        return metrics

    def get_metrics_1v5(self):
        metrics = {}

        if self.finegrain and self.model.run_finelip and self.model.support_multi_scale:
            self.all_finegrain_image_emb = torch.cat(self.all_finegrain_image_emb, dim=0).cuda()
            self.all_finegrain_text_fused = torch.cat(self.all_finegrain_text_fused, dim=0).cuda()
            self.all_lengths = torch.cat(self.all_lengths, dim=0).cuda()

            finegrain_logits_per_image = torch.empty_like(self.logits_per_image).cuda()
            chunk_size = 500
            for i in tqdm(range(0, self.all_finegrain_text_fused.shape[0], chunk_size)):
                end_idx = min(i + chunk_size, self.all_finegrain_text_fused.shape[0])
                improved_sims = self.model.cross_net.forward_dual_aggr(
                    self.all_finegrain_image_emb,
                    self.all_finegrain_text_fused[i:end_idx],
                    self.all_lengths[i:end_idx]
                )
                finegrain_logits_per_image[:, i:end_idx] = improved_sims.detach()

            finegrain_logits_per_text = finegrain_logits_per_image.t()
            alpha = 0.8
            self.logits_per_image = (
                        alpha * self.logits_per_image.cuda() + (1 - alpha) * finegrain_logits_per_image).cpu()
            self.logits_per_text = (alpha * self.logits_per_text.cuda() + (1 - alpha) * finegrain_logits_per_text).cpu()

        for k in [1, 5, 10]:
            pred_true = 0
            for i in range(self.logits_per_image.shape[0]):
                pred = self.logits_per_image[i]
                values, topk = pred.topk(k)
                for j in range(5):
                    true_index = 5 * i + j
                    if true_index in topk:
                        pred_true = pred_true + 1
                        break
            metrics[f"image_to_text_R@{k}"] = pred_true / self.logits_per_image.shape[0]

        for k in [1, 5, 10]:
            pred_true = 0
            for i in range(self.logits_per_text.shape[0]):
                pred = self.logits_per_text[i]
                values, topk = pred.topk(k)
                true_index = i // 5
                if true_index in topk:
                    pred_true = pred_true + 1
            metrics[f"text_to_image_R@{k}"] = pred_true / self.logits_per_text.shape[0]

        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--ckpt_path', default='work/checkpoints/finelip-B.pt', help="ckpt_path")
    parser.add_argument('--test_data', default='urban', help='docci,urban,coco,and flicker')
    parser.add_argument('--run_finelip', action='store_true', help='run finelip model')
    parser.add_argument('--finegrain', action='store_true', help='enable finegrain evaluation')

    parser.add_argument('--use_ot', action='store_true', help='whether use OT (same as training)')
    parser.add_argument('--pos_embed_mode', type=str, default='nonlinear',
                        help='position embed mode (same as training, e.g., nonlinear)')

    args = parser.parse_args()

    trainer = CLIP_Clean_Train(
        local_rank=0,
        args=args
    )
    trainer.test()