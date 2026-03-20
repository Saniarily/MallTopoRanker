import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

@dataclass
class PairSample:
    q: np.ndarray
    m_pos: np.ndarray
    m_neg: np.ndarray
    sid_pos: str
    sid_neg: str
    y_pos: float
    y_neg: float

class PairwiseRankDataset(Dataset):
    def __init__(self, pairs: List[PairSample], graph_cache_dir: str):
        self.pairs = pairs
        self.graph_cache_dir = graph_cache_dir

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        # 图对象在collate里再读也可以；这里直接返回路径/ID，减少IO
        return {
            "q": torch.tensor(p.q, dtype=torch.float32),
            "m_pos": torch.tensor(p.m_pos, dtype=torch.float32),
            "m_neg": torch.tensor(p.m_neg, dtype=torch.float32),
            "sid_pos": p.sid_pos,
            "sid_neg": p.sid_neg,
            "y_pos": torch.tensor(p.y_pos, dtype=torch.float32),
            "y_neg": torch.tensor(p.y_neg, dtype=torch.float32),
        }

def compute_bucket_query_map(
    df: pd.DataFrame,
    bucket_col: str,
    q_mat: np.ndarray,
) -> Dict[Any, np.ndarray]:
    """
    为每个 bucket 计算代表性 query（该 bucket 内所有样本的平均 query）。
    这样训练分布与推理分布一致（推理时使用外部 query，训练时使用 bucket-level 代表性 query）。

    Args:
        df: 包含 bucket_col 的数据框
        bucket_col: bucket 列名（例如 "bucket_id"）
        q_mat: [N, Fq] query features 矩阵

    Returns:
        Dict[bucket_id -> representative_query],其中 representative_query 是该 bucket 的平均 query 向量
    """
    bucket_query_map = {}
    for bucket_id, g in df.groupby(bucket_col):
        idxs = g.index.to_list()
        if len(idxs) > 0:
            # 计算该 bucket 的平均 query
            bucket_query_map[bucket_id] = np.mean(q_mat[idxs], axis=0)
    return bucket_query_map


def sample_pairs_within_bucket(
    df: pd.DataFrame,
    bucket_col: str,
    id_col: str,
    q_mat: np.ndarray,
    m_mat: np.ndarray,
    y: np.ndarray,
    bucket_query_map: Dict[Any, np.ndarray] = None,
    pairs_per_bucket: int = 2000,
    seed: int = 42,
    enable_hard_negatives: bool = False,
    hard_negative_ratio: float = 0.3
) -> List[PairSample]:
    """
    采样 pairwise 样本用于训练。

    Args:
        df: 训练数据框
        bucket_col: bucket 列名
        id_col: 样本 id 列名
        q_mat: query features 矩阵
        m_mat: metric features 矩阵
        y: 标签向量（排名分数）
        bucket_query_map: 可选的 bucket 代表性 query 映射。如果提供，使用该 query；否则使用 pos 样本的 query（旧行为）
        pairs_per_bucket: 每个 bucket 采样的 pair 数
        seed: 随机种子
        enable_hard_negatives: 是否启用困难负样本采样（选择排名较高但标签较低的负样本）
        hard_negative_ratio: 困难负样本的比例

    Returns:
        PairSample 列表
    """
    rnd = random.Random(seed)
    pairs: List[PairSample] = []

    # 以 bucket 分组
    for bucket_id, g in df.groupby(bucket_col):
        idxs = g.index.to_list()
        if len(idxs) < 2:
            continue

        # 排序后方便硬负采样
        idxs_sorted = sorted(idxs, key=lambda i: y[i])

        # 确定 query
        if bucket_query_map is not None and bucket_id in bucket_query_map:
            bucket_q = bucket_query_map[bucket_id]
        else:
            # 回退旧行为：等采样时再从 pos 提取
            bucket_q = None

        # 采样 pair
        num_pairs_target = min(pairs_per_bucket, len(idxs) * 10)
        for _ in range(num_pairs_target):
            # 随机选两个不同样本
            i, j = rnd.sample(idxs_sorted, 2)
            if y[i] == y[j]:
                continue

            pos, neg = (i, j) if y[i] > y[j] else (j, i)

            # 困难负样本采样：按一定比例选择排名较高但标签较低的负样本
            if enable_hard_negatives and rnd.random() < hard_negative_ratio:
                # 从与 pos 接近排名的样本中选负样本（高难度）
                pos_rank = sorted([y[idx] for idx in idxs], reverse=True).index(y[pos]) if y[pos] in [y[idx] for idx in idxs] else 0
                # 简化：从排名更高但标签低于 pos 的候选中采样
                hard_neg_candidates = [idx for idx in idxs if y[idx] < y[pos] and y[idx] > np.percentile([y[i] for i in idxs], 50)]
                if len(hard_neg_candidates) > 0:
                    neg = rnd.choice(hard_neg_candidates)

            # 确定使用的 query
            if bucket_q is not None:
                query_to_use = bucket_q
            else:
                # 旧行为：使用 pos 的 query
                query_to_use = q_mat[pos]

            pairs.append(PairSample(
                q=query_to_use,
                m_pos=m_mat[pos],
                m_neg=m_mat[neg],
                sid_pos=str(df.loc[pos, id_col]),
                sid_neg=str(df.loc[neg, id_col]),
                y_pos=float(y[pos]),
                y_neg=float(y[neg]),
            ))
            if len(pairs) >= pairs_per_bucket * df[bucket_col].nunique():
                break

    rnd.shuffle(pairs)
    return pairs
@dataclass
class PointSample:
    q: np.ndarray
    m: np.ndarray
    sid: str
    y: float
    bucket_id: str

class PointwiseRankDataset(Dataset):
    def __init__(self, samples: List[PointSample], graph_cache_dir: str):
        self.samples = samples
        self.graph_cache_dir = graph_cache_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        p = self.samples[idx]
        return {
            "q": torch.tensor(p.q, dtype=torch.float32),
            "m": torch.tensor(p.m, dtype=torch.float32),
            "sid": p.sid,
            "y": torch.tensor(p.y, dtype=torch.float32),
            "bucket_id": p.bucket_id
        }
