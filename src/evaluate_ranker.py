# src/evaluate_ranker.py
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from .infer_ranker import RankerService
from .utils_bucket import make_bucket_id, area_bin


def split_by_mall_id(df: pd.DataFrame, mall_col: str, seed: int, test_ratio: float, val_ratio: float):
    rng = np.random.RandomState(seed)
    malls = df[mall_col].dropna().astype(str).unique().tolist()
    rng.shuffle(malls)

    n = len(malls)
    test_n = int(n * test_ratio)
    val_n = int(n * val_ratio)

    test_malls = set(malls[:test_n])
    val_malls = set(malls[test_n:test_n + val_n])
    train_malls = set(malls[test_n + val_n:])

    train_df = df[df[mall_col].astype(str).isin(train_malls)].reset_index(drop=True)
    val_df = df[df[mall_col].astype(str).isin(val_malls)].reset_index(drop=True)
    test_df = df[df[mall_col].astype(str).isin(test_malls)].reset_index(drop=True)
    return train_df, val_df, test_df


def dcg_at_k(rels: np.ndarray, k: int) -> float:
    rels = rels[:k]
    if len(rels) == 0:
        return 0.0
    denom = np.log2(np.arange(2, len(rels) + 2))
    return float(np.sum((2 ** rels - 1) / denom))


def ndcg_at_k(rels_pred_order: np.ndarray, rels_true_sorted: np.ndarray, k: int) -> float:
    # rels_pred_order: relevance array ordered by model rank (desc)
    # rels_true_sorted: relevance array ordered by ideal rank (desc)
    dcg = dcg_at_k(rels_pred_order, k)
    idcg = dcg_at_k(rels_true_sorted, k)
    return float(dcg / idcg) if idcg > 0 else 0.0


def minmax01(x: np.ndarray) -> np.ndarray:
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-9:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)


def evaluate_one_query(
    svc: RankerService,
    query_row: pd.Series,
    cand_df: pd.DataFrame,
    city_cluster: int,
    topk_list: List[int],
    binary_top_pct: float = 0.10,
    max_candidates: Optional[int] = None
) -> Dict[str, Any]:
    """
    A：连续相关性：用 total_score 做 min-max 后当 graded relevance
    B：二值相关性：bucket 内 top X% total_score 视为 relevant
    """
    # query features dict
    q_cols = svc.q_cols
    q = {c: float(query_row[c]) for c in q_cols}

    # candidates: 同 bucket（已在外部过滤），排除自身
    cand_df = cand_df[cand_df[svc.id_col].astype(str) != str(query_row[svc.id_col])].reset_index(drop=True)
    if len(cand_df) == 0:
        return {"skip": True}

    if max_candidates is not None and len(cand_df) > max_candidates:
        cand_df = cand_df.sample(n=max_candidates, random_state=0).reset_index(drop=True)

    # 用 RankerService 打分（hard_filter=False 因为我们已经手动筛好候选集）
    # 为避免重复实现打分，这里临时复用 recommend()，并让它不再二次过滤：hard_filter=False
    # 做法：直接传入 city_cluster，但 hard_filter=False
    res = svc.recommend(q, city_cluster=city_cluster, topk=len(cand_df),
                        hard_filter=False, soft_relax=False, max_candidates=None, return_attention=False)

    # svc.recommend 在 hard_filter=False 时，会对全 df 打分，所以这里不用它的候选；
    # 为保证候选一致，下面我们改为：直接用 svc 的模型对 cand_df 打分（更严谨）
    # ——因此，实际排序我们自己算：
    # 轻量实现：对 cand_df 逐条调用 recommend 太慢，这里用 svc 内部模型批量打分
    # 为简洁与稳定，我们直接写一个内部批量 scorer：
    scores = batch_score_candidates(svc, q, cand_df)

    y_true = cand_df[svc.label_col].fillna(0).astype(float).values
    # A: continuous graded relevance (0..1)
    rel_cont = minmax01(y_true)
    # ideal
    ideal_order = np.argsort(-rel_cont)
    rel_cont_ideal = rel_cont[ideal_order]

    # B: binary relevance
    thr = np.quantile(y_true, 1.0 - binary_top_pct) if len(y_true) >= 5 else np.max(y_true)
    rel_bin = (y_true >= thr).astype(int)
    rel_bin_ideal = rel_bin[np.argsort(-rel_bin)]

    pred_order = np.argsort(-scores)
    rel_cont_pred = rel_cont[pred_order]
    rel_bin_pred = rel_bin[pred_order]

    out = {"skip": False}
    for k in topk_list:
        out[f"ndcg_cont@{k}"] = ndcg_at_k(rel_cont_pred, rel_cont_ideal, k)
        # precision/recall on binary
        top_rel = rel_bin_pred[:k]
        out[f"precision_bin@{k}"] = float(np.mean(top_rel)) if len(top_rel) else 0.0
        out[f"recall_bin@{k}"] = float(np.sum(top_rel) / max(1, np.sum(rel_bin)))  # recall against all relevant in cand list

    # Spearman between scores and y_true (monotonic correlation)
    if len(y_true) >= 3:
        out["spearman"] = float(spearmanr(scores, y_true).correlation)
    else:
        out["spearman"] = 0.0

    return out


def batch_score_candidates(svc: RankerService, q: Dict[str, float], cand_df: pd.DataFrame) -> np.ndarray:
    """
    用 svc.model 对给定 cand_df 批量打分（query固定，候选变化）。
    """
    import torch
    from torch_geometric.data import Batch

    # query tensor
    q_raw = np.array([float(q[c]) for c in svc.q_cols], dtype=float).reshape(1, -1)
    q_std = svc.scaler.transform_q(q_raw)
    q_t = torch.tensor(q_std, dtype=torch.float32).to(svc.device)  # [1,10]

    # candidates tensors
    Qcand = cand_df[svc.q_cols].fillna(0).astype(float).values
    Mcand = cand_df[svc.m_cols].fillna(0).astype(float).values
    Qcand = svc.scaler.transform_q(Qcand)
    Mcand = svc.scaler.transform_m(Mcand)

    sids = cand_df[svc.id_col].astype(str).tolist()

    scores = []
    bs = 64
    for st in range(0, len(cand_df), bs):
        ed = min(st + bs, len(cand_df))
        # 注意：这里 bq 用候选的 Query 特征还是用真实 Query？
        # 对于检索推荐，Query 应当固定为用户输入；候选自身 Query 不应替代 Query。
        # 因此 bq 全部用 q_t 复制到batch大小。
        bq = q_t.repeat(ed - st, 1)  # [B,10]
        bm = torch.tensor(Mcand[st:ed], dtype=torch.float32).to(svc.device)

        glist = [_load_graph_cached(svc, sid) for sid in sids[st:ed]]
        gb = Batch.from_data_list(glist).to(svc.device)

        with torch.no_grad():
            s, _, _ = svc.model(bq, bm, gb.x, gb.edge_index, gb.batch)
        scores.append(s.detach().cpu().numpy())

    return np.concatenate(scores, axis=0)


def _load_graph_cached(svc: RankerService, sid: str):
    # 简单内存缓存（评估时加速）
    if not hasattr(svc, "_graph_mem_cache"):
        svc._graph_mem_cache = {}
    cache = svc._graph_mem_cache
    if sid in cache:
        return cache[sid]
    g = torch.load(svc.cache_dir / f"{sid}.pt")
    cache[sid] = g
    return g


def main(config_path: str = "./config.yaml", checkpoint_path: str = "./outputs/checkpoints/latest.pt"):
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    out_dir = Path("./outputs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    svc = RankerService(config_path=config_path, checkpoint_path=checkpoint_path)
    df = svc.df.copy()

    mall_col = cfg.get("features", {}).get("mall_id_col", "mall_id")
    if mall_col not in df.columns:
        raise ValueError(f"mall_id_col='{mall_col}' not found in main_table columns.")

    # bucket id
    t1, t2 = cfg["bucket"]["area_thresholds"]
    df["bucket_id"] = make_bucket_id(df, svc.city_col, svc.area_col, t1, t2)

    # only valid rows
    df = df.dropna(subset=[svc.id_col, svc.label_col, svc.city_col, svc.area_col, "bucket_id"]).reset_index(drop=True)
    # ensure graph exists
    df["has_graph"] = df[svc.id_col].astype(str).apply(lambda sid: (svc.cache_dir / f"{sid}.pt").exists())
    df = df[df["has_graph"]].reset_index(drop=True)

    # split by mall_id
    seed = cfg["train"]["seed"]
    train_df, val_df, test_df = split_by_mall_id(df, mall_col, seed, cfg["train"]["test_ratio"], cfg["train"]["val_ratio"])

    # evaluate on test
    topk_list = [5, 10, 20]
    binary_top_pct = 0.10
    max_queries_per_bucket = 50     # 防止评估太慢；论文可提高/全量
    max_candidates_per_query = 300  # 控制复杂度；论文可提高/全量

    metrics_rows = []
    attn_accum = []  # condition->metric attention 累积（用于热力图）
    attn_count = 0

    # 预先按 bucket 建索引（test 集内部排序）
    test_groups = {bid: g.reset_index(drop=True) for bid, g in test_df.groupby("bucket_id")}

    for bid, g in tqdm(test_groups.items(), desc="Evaluate by bucket"):
        if len(g) < 6:
            continue

        # 取部分 query（可全量）
        q_g = g.sample(n=min(max_queries_per_bucket, len(g)), random_state=0).reset_index(drop=True)

        for _, qrow in q_g.iterrows():
            # 候选列表：同 bucket 的 test 样本
            cand_df = g
            city_cluster = int(qrow[svc.city_col])

            m = evaluate_one_query(
                svc=svc,
                query_row=qrow,
                cand_df=cand_df,
                city_cluster=city_cluster,
                topk_list=topk_list,
                binary_top_pct=binary_top_pct,
                max_candidates=max_candidates_per_query
            )
            if m.get("skip", False):
                continue

            m.update({
                "bucket_id": bid,
                "city_cluster": int(qrow[svc.city_col]),
                "area_bin": area_bin(float(qrow[svc.area_col]), t1, t2),
            })
            metrics_rows.append(m)

            # 同时累积“条件→指标注意力”（抽样少量 query 统计即可）
            if attn_count < 500:  # 控制运行时间
                q_feats = {c: float(qrow[c]) for c in svc.q_cols}
                # 随机取一个候选做 attention（为了得到条件→指标的平均偏好）
                one_cand = cand_df.sample(n=1, random_state=attn_count).iloc[0]
                # 复用 svc.recommend 的解释（只取Top1即可）
                res = svc.recommend(q_feats, city_cluster=city_cluster, topk=1, hard_filter=False, return_attention=True, max_candidates=200)
                if res["topk"]:
                    expl = res["topk"][0].get("explanations", {})
                    # 取 condition_to_metric 的权重近似（更严格可直接从 attn 输出原矩阵；这里用已整理的 top metrics）
                    # 为论文热力图更稳定，建议你后续把原 attention 矩阵也存下来做平均。
                    # 这里给一个“可运行的简化版”：按每个 condition 的 top_metrics 权重累加到矩阵中。
                    cond_names = svc.q_cols
                    met_names = svc.m_cols
                    mat = np.zeros((len(cond_names), len(met_names)), dtype=float)
                    for d in expl.get("condition_to_metric", []):
                        c = d["condition"]
                        if c not in cond_names:
                            continue
                        i = cond_names.index(c)
                        for tm in d["top_metrics"]:
                            mn = tm["name"]
                            if mn in met_names:
                                j = met_names.index(mn)
                                mat[i, j] += float(tm["weight"])
                    attn_accum.append(mat)
                    attn_count += 1

    met_df = pd.DataFrame(metrics_rows)
    if len(met_df) == 0:
        raise RuntimeError("No evaluation records generated. Check your test split or bucket sizes.")

    met_df.to_csv(out_dir / "metrics_per_query.csv", index=False, encoding="utf-8-sig")

    # summary
    summary = {}
    for col in met_df.columns:
        if any(col.startswith(x) for x in ["ndcg_cont@", "precision_bin@", "recall_bin@", "spearman"]):
            summary[col] = {
                "mean": float(met_df[col].mean()),
                "std": float(met_df[col].std()),
                "median": float(met_df[col].median())
            }

    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # by bucket
    agg_cols = [c for c in met_df.columns if c.startswith("ndcg_cont@") or c.startswith("precision_bin@") or c.startswith("recall_bin@") or c == "spearman"]
    by_bucket = met_df.groupby("bucket_id")[agg_cols].mean().reset_index()
    by_bucket.to_csv(out_dir / "metrics_by_bucket.csv", index=False, encoding="utf-8-sig")

    # attention mean matrix (for paper heatmap)
    if len(attn_accum) > 0:
        attn_mean = np.mean(np.stack(attn_accum, axis=0), axis=0)
        attn_df = pd.DataFrame(attn_mean, index=svc.q_cols, columns=svc.m_cols)
        attn_df.to_csv(out_dir / "attention_condition_metric_mean.csv", encoding="utf-8-sig")

    print("Saved:")
    print(" - outputs/eval/metrics_per_query.csv")
    print(" - outputs/eval/metrics_summary.json")
    print(" - outputs/eval/metrics_by_bucket.csv")
    print(" - outputs/eval/attention_condition_metric_mean.csv (if generated)")


if __name__ == "__main__":
    main()
