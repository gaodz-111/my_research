import torch
import numpy as np
import json
import copy

# def build_target_attention_distribution(
#         question_tokens,
#         filter_tokens,
#         mark_area,
#         img_token_num=577,
#         img_token_side=24,
#         image_size=336
# ):
#     """
#     question_tokens: 原始分词结果
#     filter_tokens: 需要计算attention的关键词列表
#     mark_area: 每个filter token对应的bbox list
#     返回: [L_question, img_token_num] 的attention分布
#     """
#     L_question = len(question_tokens)
#
#     # 1. patch token 网格中心坐标
#     stride = image_size / img_token_side
#     grid_x = (np.arange(img_token_side) + 0.5) * stride
#     grid_y = (np.arange(img_token_side) + 0.5) * stride
#     grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
#     grid_xx = grid_xx.flatten()
#     grid_yy = grid_yy.flatten()
#
#     # 2. 解析 filter tokens
#     filter_tokens_list = eval(filter_tokens) if isinstance(filter_tokens, str) else filter_tokens
#
#     # 3. 每个 filter token 的空间 mask
#     token_img_masks = []
#     for idx, token in enumerate(filter_tokens_list):
#         mask = np.zeros(img_token_num - 1, dtype=np.float32)  # 去掉 CLS
#         bboxes = mark_area[idx]
#         if bboxes is None or bboxes == "null":
#             token_img_masks.append(mask)
#             continue
#         for bbox_info in bboxes:
#             x1, y1, x2, y2 = bbox_info['bbox_2d']
#             inside = (grid_xx >= x1) & (grid_xx <= x2) & (grid_yy >= y1) & (grid_yy <= y2)
#             mask = np.logical_or(mask, inside)
#         mask = mask.astype(np.float32)
#         if mask.sum() > 0:
#             mask /= mask.sum()
#         token_img_masks.append(mask)
#
#     # 4. question token 与 filter token 对齐
#     P = np.zeros((L_question, img_token_num), dtype=np.float32)
#     qt_lower = [qt.lower() for qt in question_tokens]
#     visited = [False] * L_question  # 标记已赋值的 token
#
#     for i in range(L_question):
#         if visited[i]:
#             continue
#         matched = False
#         for ft_idx, ft in enumerate(filter_tokens_list):
#             ft_lower = ft.lower()
#             # 尝试从当前位置向后拼接 token，长度不超过 ft 的长度
#             max_span = min(L_question - i, len(ft_lower) + 5)  # 多容忍几个字符
#             for span in range(1, max_span + 1):
#                 concat_tokens = "".join(qt_lower[i:i + span])
#                 if concat_tokens == ft_lower:
#                     # 匹配成功，赋值 mask
#                     for j in range(i, i + span):
#                         P[j, 1:] = token_img_masks[ft_idx]
#                         visited[j] = True
#                     matched = True
#                     break
#             if matched:
#                 break
#         # 如果没匹配上，就保持全0
#
#     return torch.from_numpy(P)
#
#
# # ====== 数据处理 ======
# data = []
# with open("D:/pycharmProject/Re_align/re_data/combined_data_token_filter_mark_t1.1.json", "r", encoding="utf-8") as f:
#     for line in f:
#         data.append(json.loads(line))
#
# new_data = []
# for item in data:
#     item_t = copy.deepcopy(item)
#     P = build_target_attention_distribution(
#         question_tokens=item["question_tokens"],
#         filter_tokens=item["filter_tokens"],
#         mark_area=item["mark_area"],
#         img_token_num=577,
#         img_token_side=24,
#         image_size=336
#     )
#     Index = P.tolist()
#     q_idx_list = [0 for _ in range(len(item["question_tokens"]))]
#
#     for i in range(len(Index)):
#         if sum(Index[i]) != 0:
#             q_idx_list[i] = 1
#     res = []
#     for i in range(len(q_idx_list)):
#         if q_idx_list[i] == 1:
#            res.append(i)
#     item_t["important_q_idx"] = res
#     new_data.append(item_t)
#
# with open("D:/pycharmProject/Re_align/re_data/combined_data_token_filter_mark_t2_asa.json", "w", encoding="utf-8") as f:
#     for item in new_data:
#         json.dump(item, f, ensure_ascii=False)
#         f.write("\n")
#
# # import torch
# # import numpy as np
# # import json
# # import copy
# #
# #
# # def build_target_attention_distribution_soft_weighted(
# #         question_tokens,
# #         filter_tokens,
# #         mark_area,
# #         img_token_num=577,
# #         img_token_side=24,
# #         image_size=336,
# #         sigma_ratio=0.5,
# #         special_ratio=0.8  # 特殊区域总概率
# # ):
# #     """
# #     构建基于 bbox 的“软掩码”目标注意力分布：
# #       - 特殊 token 对应区域分配 special_ratio 的总概率
# #       - 其他区域均匀分配 (1 - special_ratio)
# #       - 特殊区域内部按距离做加权
# #     """
# #     L_question = len(question_tokens)
# #
# #     # 1. patch token 网格中心坐标
# #     stride = image_size / img_token_side
# #     grid_x = (np.arange(img_token_side) + 0.5) * stride
# #     grid_y = (np.arange(img_token_side) + 0.5) * stride
# #     grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
# #     grid_points = np.stack([grid_xx.flatten(), grid_yy.flatten()], axis=-1)  # [L_img-1, 2]
# #
# #     # 2. 解析 filter tokens
# #     filter_tokens_list = eval(filter_tokens) if isinstance(filter_tokens, str) else filter_tokens
# #
# #     # 3. 每个 filter token 的加权掩码
# #     token_img_masks = []
# #     for idx, token in enumerate(filter_tokens_list):
# #         mask = np.zeros(img_token_num - 1, dtype=np.float32)  # 去掉 CLS
# #         bboxes = mark_area[idx]
# #         if bboxes is None or bboxes == "null":
# #             token_img_masks.append(mask)
# #             continue
# #
# #         for bbox_info in bboxes:
# #             x1, y1, x2, y2 = bbox_info['bbox_2d']
# #
# #             # bbox 中心 & 尺寸
# #             bbox_cx = (x1 + x2) / 2
# #             bbox_cy = (y1 + y2) / 2
# #             bbox_w = max(x2 - x1, 1e-6)
# #             bbox_h = max(y2 - y1, 1e-6)
# #
# #             # 高斯 σ
# #             sigma_x = bbox_w * sigma_ratio
# #             sigma_y = bbox_h * sigma_ratio
# #
# #             # 高斯权重
# #             dx = (grid_points[:, 0] - bbox_cx) / sigma_x
# #             dy = (grid_points[:, 1] - bbox_cy) / sigma_y
# #             dist2 = dx ** 2 + dy ** 2
# #             gauss_weight = np.exp(-0.5 * dist2)
# #
# #             # 只保留 bbox 内 patch
# #             inside = (grid_points[:, 0] >= x1) & (grid_points[:, 0] <= x2) & \
# #                      (grid_points[:, 1] >= y1) & (grid_points[:, 1] <= y2)
# #             gauss_weight = gauss_weight * inside.astype(np.float32)
# #
# #             mask = np.maximum(mask, gauss_weight)
# #
# #         # 如果有值，则归一化到特殊区域概率 special_ratio
# #         if mask.sum() > 0:
# #             mask = mask / mask.sum() * special_ratio
# #
# #         token_img_masks.append(mask)
# #
# #     # 4. question token 与 filter token 对齐
# #     P = np.zeros((L_question, img_token_num), dtype=np.float32)
# #     qt_lower = [qt.lower() for qt in question_tokens]
# #     visited = [False] * L_question
# #
# #     for i in range(L_question):
# #         if visited[i]:
# #             continue
# #         matched = False
# #         for ft_idx, ft in enumerate(filter_tokens_list):
# #             ft_lower = ft.lower()
# #             max_span = min(L_question - i, len(ft_lower) + 5)
# #             for span in range(1, max_span + 1):
# #                 concat_tokens = "".join(qt_lower[i:i + span])
# #                 if concat_tokens == ft_lower:
# #                     for j in range(i, i + span):
# #                         special_mask = token_img_masks[ft_idx]
# #
# #                         # 其他区域均分剩余概率
# #                         if special_mask.sum() > 0:
# #                             other_mask = (special_mask == 0).astype(np.float32)
# #                             if other_mask.sum() > 0:
# #                                 other_mask = other_mask / other_mask.sum() * (1 - special_ratio)
# #                             full_mask = special_mask + other_mask
# #                         else:
# #                             # 如果没有标注，直接均匀分布
# #                             full_mask = np.ones(img_token_num - 1, dtype=np.float32)
# #                             full_mask /= full_mask.sum()
# #
# #                         P[j, 1:] = full_mask
# #                         visited[j] = True
# #                     matched = True
# #                     break
# #             if matched:
# #                 break
# #
# #     return torch.from_numpy(P)
# #
# #
# # # ====== 数据处理 ======
# # data = []
# # with open("D:/pycharmProject/Re_align/re_data/combined_data_token_filter_mark_t1.1.json", "r", encoding="utf-8") as f:
# #     for line in f:
# #         data.append(json.loads(line))
# #
# # new_data = []
# # for item in data:
# #     item_t = copy.deepcopy(item)
# #     P = build_target_attention_distribution_soft_weighted(
# #         question_tokens=item["question_tokens"],
# #         filter_tokens=item["filter_tokens"],
# #         mark_area=item["mark_area"],
# #         img_token_num=577,
# #         img_token_side=24,
# #         image_size=336,
# #         sigma_ratio=0.5,
# #         special_ratio=0.8
# #     )
# #     item_t["attention_distribution"] = P.tolist()
# #     new_data.append(item_t)
# #
# # with open("D:/pycharmProject/Re_align/re_data/combined_data_token_filter_mark_t2_soft.json", "w", encoding="utf-8") as f:
# #     for item in new_data:
# #         json.dump(item, f, ensure_ascii=False)
# #         f.write("\n")
#
#

def build_target_attention_distribution_with_idx(
        question_tokens,
        filter_tokens,
        mark_area,
        img_token_num=577,
        img_token_side=24,
        image_size=336
):
    """
    question_tokens: 原始分词结果
    filter_tokens: 需要计算attention的关键词列表
    mark_area: 每个filter token对应的bbox list
    返回:
        P: [L_question, img_token_num] 的attention分布
        important_idx: 重要 token 下标列表
    """
    L_question = len(question_tokens)

    # 1. patch token 网格中心坐标
    stride = image_size / img_token_side
    grid_x = (np.arange(img_token_side) + 0.5) * stride
    grid_y = (np.arange(img_token_side) + 0.5) * stride
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    grid_xx = grid_xx.flatten()
    grid_yy = grid_yy.flatten()

    # 2. 解析 filter tokens
    filter_tokens_list = eval(filter_tokens) if isinstance(filter_tokens, str) else filter_tokens

    # 3. 每个 filter token 的空间 mask
    token_img_masks = []
    for idx, token in enumerate(filter_tokens_list):
        mask = np.zeros(img_token_num - 1, dtype=np.float32)  # 去掉 CLS
        bboxes = mark_area[idx]
        if bboxes is None or bboxes == "null":
            token_img_masks.append(mask)
            continue
        for bbox_info in bboxes:
            x1, y1, x2, y2 = bbox_info['bbox_2d']
            inside = (grid_xx >= x1) & (grid_xx <= x2) & (grid_yy >= y1) & (grid_yy <= y2)
            mask = np.logical_or(mask, inside)
        mask = mask.astype(np.float32)
        if mask.sum() > 0:
            mask /= mask.sum()
        token_img_masks.append(mask)

    # 4. question token 与 filter token 对齐，生成 P 和 important_idx
    P = np.zeros((L_question, img_token_num), dtype=np.float32)
    qt_lower = [qt.lower() for qt in question_tokens]
    visited = [False] * L_question
    important_idx = []

    for i in range(L_question):
        if visited[i]:
            continue
        matched = False
        for ft_idx, ft in enumerate(filter_tokens_list):
            ft_lower = ft.lower()
            max_span = min(L_question - i, len(ft_lower) + 5)
            for span in range(1, max_span + 1):
                concat_tokens = "".join(qt_lower[i:i + span])
                if concat_tokens == ft_lower:
                    # 匹配成功
                    for j in range(i, i + span):
                        P[j, 1:] = token_img_masks[ft_idx]
                        visited[j] = True
                        important_idx.append(j)
                    matched = True
                    break
            if matched:
                break
    return torch.from_numpy(P), important_idx






data = []
with open("D:/pycharmProject/Re_align/re_data/combined_data_token_filter_mark_t1.1.json", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))


new_data = []
for item in data:
    item_t = copy.deepcopy(item)

    # ===== 数据清理 =====
    filter_tokens = item_t["filter_tokens"]
    mark_area = item_t["mark_area"]

    # 如果 filter_tokens 是字符串，就转为 list
    if isinstance(filter_tokens, str):
        try:
            filter_tokens = eval(filter_tokens)  # 如果格式是 Python list 字符串
        except:
            import ast
            filter_tokens = ast.literal_eval(filter_tokens)

    new_filter_tokens = []
    new_mark_area = []

    for tok, area in zip(filter_tokens, mark_area):
        if isinstance(area, list) and area != "null":
            # 过滤掉 area 中不是 dict 或没有 label 的部分
            clean_area = [r for r in area if isinstance(r, dict) and "label" in r]
            if clean_area:
                new_filter_tokens.append(tok)
                new_mark_area.append(clean_area)

    item_t["filter_tokens"] = new_filter_tokens
    item_t["mark_area"] = new_mark_area
    # ===== 清理结束 =====

    # 2. 获得 P 和重要 token idx
    P, important_idx = build_target_attention_distribution_with_idx(
        question_tokens=item_t["question_tokens"],
        filter_tokens=item_t["filter_tokens"],
        mark_area=item_t["mark_area"],
        img_token_num=577,
        img_token_side=24,
        image_size=336
    )

    item_t["important_q_idx"] = important_idx
    new_data.append(item_t)

# 3. 保存结果
with open("D:/pycharmProject/Re_align/re_data/combined_data_token_filter_mark_t2_asa.json", "w", encoding="utf-8") as f:
    for item in new_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")
