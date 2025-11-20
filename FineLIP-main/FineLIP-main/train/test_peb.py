import torch
from tqdm import tqdm
import sys
import os
from torch.utils.data import DataLoader
from torch.nn.functional import normalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append("..")

import torch.nn.functional as F
import torch.nn as nn

from model import finelip
import clip # for CLIP evaluation
# from open_clip_long import factory as open_clip # for bigG model

from sharegpt4v import share4v_val_dataset
from eval_data import urban_dataset, coco_dataset, flickr_dataset, docci_dataset ,docci_dataset_peb, urban_dataset_peb

import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")
device="cuda"

class CLIP_Clean_Train():
    def __init__(self, local_rank, args):
        self.local_rank = local_rank
        self.test_data = args.test_data
        self.device = device
        #
        self.model, self.preprocess = finelip.load_finegrained_clip(args.ckpt_path, device='cpu', run_finelip=args.run_finelip)
        self.word_feat_dict = self._load_word_feat_cache("/data2/gaodz/FineLIP-main/word_feat_cache.pt")  # 新增这行
        # 2. 获取词向量维度（用于后续处理）
        self.clip_dim = next(iter(self.word_feat_dict.values())).shape[0] if self.word_feat_dict else 768

        # self.model, self.preprocess = clip.load('ViT-L/14', device='cpu') # for CLIP evaluation
        # self.model, _, self.preprocess = open_clip.create_model_and_transforms(
        #     'ViT-bigG-14',
        #     pretrained="experiments/open_clip_pytorch_model.bin",
        #     text_cfg={'context_length': 77, 'vocab_size': 49408, "width": 1280, "heads": 20, "layers": 32},
        # )
        torch.cuda.set_device(device=f'cuda:{local_rank}')
        self.model = self.model.float().cuda()

        self.finegrain = args.finegrain
        self.model = self.model.float().to(self.device)

        self.all_finegrain_image_features = []
        self.all_finegrain_text_features = []
        self.all_rope_features = []  # 新增：收集 PEB 输出的 rope_position_features
        self.lengths = []


    def _load_word_feat_cache(self, cache_path):
        """加载预计算的词向量缓存"""
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"词向量缓存文件不存在：{cache_path}")
        print(f"加载词向量缓存：{cache_path}")
        return torch.load(cache_path, map_location='cpu')  # 加载到CPU，后续按需转移到GPU

    @torch.no_grad()
    def test_epoch(self):
        all_image_features, all_text_features = [], []
        self.all_finegrain_image_features, self.all_finegrain_text_features, self.all_rope_features = [], [] , []
        self.lengths = []

        for id, batch in enumerate(tqdm(self.testloader)):
            images, captions, caption_shorts, indices, word_feats, word_centers = batch
            images = images.to(self.device, non_blocking=True)
            word_feats = word_feats.to(self.device, non_blocking=True)
            word_centers = word_centers.to(self.device, non_blocking=True)



            # ========== 图像特征（直接调用封装好的方法） ==========
            img_embs = self.model.encode_image_for_crossnet(images)  # [B, N_patches+1, D]
            image_features = F.normalize(img_embs[:, 0, :], dim=-1)  # 普通CLIP特征（CLS）

            # ========== 文本特征 ==========
            text_token = finelip.tokenize(captions, truncate=True).to(self.device)  # [B, 248]
            cap_lens = torch.sum(text_token != 0, dim=1).cpu().tolist()
            self.lengths.extend(cap_lens)

            # PEB位置嵌入
            clip_style_pos_emb, _, _,rope_features = self.model.peb(
                word_feat=word_feats,
                word_pos_248=word_centers,
                text_feat_full=self.model.base_clip.encode_text_full(text_token)
            )

            # 调用封装好的文本特征提取方法
            cap_embs = self.model.encode_text_for_crossnet(
                texts=text_token,
                clip_style_pos_emb=clip_style_pos_emb
            )  # [B, 248, D]

            # 普通文本特征（mean pool）
            text_features = F.normalize(cap_embs.mean(dim=1), dim=-1)

            # ========== 收集特征 ==========
            if self.finegrain:
                # 细粒度特征直接用 cross_net 输入格式
                self.all_finegrain_image_features.append(img_embs.cpu())
                self.all_finegrain_text_features.append(cap_embs.cpu())
                self.all_rope_features.append(rope_features.cpu())
            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())

        # 拼接普通特征并计算基础logits
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)
        self.logits_per_image = (all_image_features @ all_text_features.t()).detach()
        self.logits_per_text = self.logits_per_image.t()

        # 指标计算
        if self.test_data in ['share', 'urban', 'docci']:
            self.ground_truth = torch.arange(len(all_text_features)).view(-1, 1)
            results = self.get_metrics()
        else:
            results = self.get_metrics_1v5()

        return results



    def custom_collate_fn(self, batch):
        # 初始化列表，对应6个返回值
        images = []
        captions = []  # 完整caption
        caption_shorts = []  # 短caption
        indices = []  # index
        words_list = []  # 词列表
        word_centers_list = []  # 词中心位置
        word_feats_list = []  # 词向量

        for item in batch:
            # 解包6个属性（与数据集返回顺序严格一致！）
            image, caption, caption_short, idx, words, word_centers = item

            # 收集基础信息
            images.append(image)
            captions.append(caption)
            caption_shorts.append(caption_short)
            indices.append(idx)
            words_list.append(words)
            word_centers_list.append(word_centers)

            # 加载词向量（从缓存查表，与训练一致）
            word_feats = []
            for word in words:
                if word in self.word_feat_dict:
                    word_feats.append(self.word_feat_dict[word].clone())
                else:
                    word_feats.append(torch.zeros(self.clip_dim, dtype=torch.float32))
            # 处理空词列表情况
            if not word_feats:
                word_feats = torch.zeros(0, self.clip_dim)
            else:
                word_feats = torch.stack(word_feats)
            word_feats_list.append(word_feats)

        # 1. 拼接图像（转为tensor）
        images = torch.stack(images, dim=0)  # [B, 3, 224, 224]

        # 2. 处理词向量padding（与训练一致）
        max_words = max([wf.shape[0] for wf in word_feats_list], default=0)
        padded_word_feats = []
        for wf in word_feats_list:
            if wf.shape[0] < max_words:
                pad_len = max_words - wf.shape[0]
                padded = torch.nn.functional.pad(wf, (0, 0, 0, pad_len), mode='constant', value=0)
            else:
                padded = wf[:max_words]  # 截断过长的词列表
            padded_word_feats.append(padded)
        padded_word_feats = torch.stack(padded_word_feats, dim=0)  # [B, max_words, D]

        # 3. 处理词中心位置padding（无效位置用-1）
        padded_word_centers = []
        for wc in word_centers_list:
            if len(wc) < max_words:
                pad_len = max_words - len(wc)
                padded = torch.cat([torch.tensor(wc, dtype=torch.long),
                                    torch.full((pad_len,), -1, dtype=torch.long)], dim=0)
            else:
                padded = torch.tensor(wc[:max_words], dtype=torch.long)
            padded_word_centers.append(padded)
        padded_word_centers = torch.stack(padded_word_centers, dim=0)  # [B, max_words]

        # 4. 返回6个字段（顺序与解包一致，供test_epoch使用）
        return (images, captions, caption_shorts, indices,
                padded_word_feats, padded_word_centers)
    def test(self):
        self.model.eval()
        if self.test_data == 'share':
            testset = share4v_val_dataset(self.preprocess,word_feat_dict=self.word_feat_dict)
        elif self.test_data == 'urban':
            testset = urban_dataset_peb(self.preprocess,word_feat_dict=self.word_feat_dict)
        elif self.test_data == 'coco':
            testset = coco_dataset(self.preprocess)
        elif self.test_data == 'flickr':
            testset = flickr_dataset(self.preprocess)
        elif self.test_data == 'docci':
            testset = docci_dataset_peb(self.preprocess,word_feat_dict=self.word_feat_dict)

        self.testloader = DataLoader(testset, batch_size=200, num_workers=8,
                                     pin_memory=True,collate_fn=self.custom_collate_fn)  # changed batch size from 1000 to 500 due to OOM
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

        if self.finegrain:
            # 1. 拼接收集到的细粒度特征
            img_feats = torch.cat(self.all_finegrain_image_features, dim=0).to(self.device)  # [M, N_patches, D]
            text_feats = torch.cat(self.all_finegrain_text_features, dim=0).to(self.device)  # [N, L_text, D]

            # 2. 初始化细粒度logits矩阵
            finegrain_logits_per_image = torch.empty_like(self.logits_per_image)

            # 3. 分块计算细粒度相似度
            chunk_size = 100  # 可调，避免显存爆炸
            for i in tqdm(range(0, text_feats.shape[0], chunk_size)):
                # 取出文本块和对应的长度
                text_chunk = text_feats[i:i + chunk_size]
                cap_lens_chunk = self.lengths[i:i + chunk_size]

                # 调用 cross_net.forward_dual_aggr 计算相似度
                sim_chunk= self.model.cross_net.forward_dual_aggr(
                    img_embs=img_feats,
                    cap_embs=text_chunk,
                    cap_lens=cap_lens_chunk,
                )

                # 存到结果矩阵
                finegrain_logits_per_image[:, i:i + chunk_size] = sim_chunk.detach().cpu()

            # 4. 转置得到文本->图像的logits
            finegrain_logits_per_text = finegrain_logits_per_image.t()

            # 5. 融合原始logits和细粒度logits
            alpha = 0.8
            self.logits_per_image = alpha * self.logits_per_image + (1 - alpha) * finegrain_logits_per_image
            self.logits_per_text = alpha * self.logits_per_text + (1 - alpha) * finegrain_logits_per_text

        # 6. 计算排名指标
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

        # self.save_logits(self.logits_per_image, f"{os.path.dirname(args.ckpt_path)}/{args.test_data}/logits_per_image.npy")
        if self.finegrain:
            self.all_finegrain_image_features = torch.cat(self.all_finegrain_image_features, dim=0)
            self.all_finegrain_text_features = torch.cat(self.all_finegrain_text_features, dim=0)

            finegrain_logits_per_image = torch.empty_like(self.logits_per_image)
            finegrain_logits_per_text = torch.empty_like(self.logits_per_text)
            chunk_size = 500
            for i in tqdm(range(0, self.all_finegrain_text_features.shape[0], chunk_size)):
                finegrain_logits_per_image[:, i:i + chunk_size] = self.model.cross_net.forward_dual_aggr(
                    self.all_finegrain_image_features, self.all_finegrain_text_features[i:i + chunk_size].cuda(),
                    self.lengths[i:i + chunk_size]).detach().cpu()

            finegrain_logits_per_text = finegrain_logits_per_image.t()

            alpha = 0.8  # [0.8, 0.2]
            # self.save_logits(finegrain_logits_per_image, f"{os.path.dirname(args.ckpt_path)}/{args.test_data}/finegrain_logits_per_image.npy")
            self.logits_per_image = alpha * self.logits_per_image + (1 - alpha) * finegrain_logits_per_image
            self.logits_per_text = alpha * self.logits_per_text + (1 - alpha) * finegrain_logits_per_text

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
    args = parser.parse_args()

    trainer = CLIP_Clean_Train(
        local_rank=0,
        args=args
    )
    trainer.test()
