# src/infer_ranker.py
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch

from .utils_bucket import area_bin
from .utils_scaler import DualScaler
from .model_ranker import GraphMatchRanker
from .explain import summarize_condition_metric_attention


def _select_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def _load_graph(graph_cache_dir: str, sid: str):
    p = Path(graph_cache_dir) / f"{sid}.pt"
    return torch.load(p)


class RankerService:
    """
    封装：加载模型/数据/归一化器；提供 recommend() 供评估/案例复用。
    """
    def __init__(self, config_path: str = "./config.yaml", checkpoint_path: str = "./outputs/checkpoints/latest.pt"):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.cfg = ckpt["cfg"] if isinstance(ckpt, dict) and "cfg" in ckpt else yaml.safe_load(open(config_path, "r", encoding="utf-8"))
        self.state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        self.id_col = self.cfg["features"].get("id_col", "floor_id")
        self.city_col = self.cfg["features"].get("city_cluster_col", "city_cluster")
        self.area_col = self.cfg["features"].get("total_area_col", "total_area")
        self.label_col = self.cfg["features"].get("label_col", "total_score")

        self.q_cols = self.cfg["features"]["query_cols"]
        self.m_cols = self.cfg["features"]["metric_cols"]

        self.scaler = DualScaler.load(self.cfg["cache"]["scaler_path"])
        self.df = pd.read_csv(self.cfg["data"]["main_table_csv"])
        self.cache_dir = Path(self.cfg["cache"]["graph_cache_dir"])

        self.device = _select_device()

        self.model = GraphMatchRanker(
            q_features=len(self.q_cols),
            metric_features=len(self.m_cols),
            node_in_dim=3,
            d_model=self.cfg["model"]["d_model"],
            gnn_layers=self.cfg["model"]["gnn_layers"],
            cross_heads=self.cfg["model"]["cross_heads"],
            dropout=self.cfg["model"]["dropout"],
        ).to(self.device)
        self.model.load_state_dict(self.state_dict)
        self.model.eval()

        self.t1, self.t2 = self.cfg["bucket"]["area_thresholds"]

    def recommend(
        self,
        query_features: Dict[str, float],
        city_cluster: int,
        topk: int = 10,
        hard_filter: bool = True,
        soft_relax: bool = False,
        relax_to_adjacent_area_bin: bool = True,
        max_candidates: Optional[int] = None,
        return_attention: bool = True
    ) -> Dict[str, Any]:
        """
        query_features: 必须包含 self.q_cols 所有字段（10维），尤其 total_area、Tx
        city_cluster: 单独传入（你硬约束字段）
        """
        # ---- build query tensor ----
        q_raw = np.array([float(query_features[c]) for c in self.q_cols], dtype=float).reshape(1, -1)
        q_std = self.scaler.transform_q(q_raw)
        q = torch.tensor(q_std, dtype=torch.float32).to(self.device)  # [1,10]

        q_area_bin = area_bin(float(query_features[self.area_col]), self.t1, self.t2)

        # ---- candidate filtering ----
        cand = self.df.copy()
        cand = cand.dropna(subset=[self.id_col]).reset_index(drop=True)

        # 仅保留有图缓存的候选
        cand["has_graph"] = cand[self.id_col].astype(str).apply(lambda sid: (self.cache_dir / f"{sid}.pt").exists())
        cand = cand[cand["has_graph"]].reset_index(drop=True)

        constraints = []

        if hard_filter:
            cand = cand[cand[self.city_col] == city_cluster]
            constraints.append(f"city_cluster={city_cluster} 命中")

            cand_area_bin = cand[self.area_col].apply(lambda x: area_bin(x, self.t1, self.t2))
            cand = cand[cand_area_bin == q_area_bin]
            constraints.append(f"total_area 档位={q_area_bin}（阈值 {self.t1}/{self.t2}）命中")

        # soft relax（可选）：如果硬筛后为空或太少，放宽到相邻面积档
        if soft_relax and len(cand) < 5 and relax_to_adjacent_area_bin:
            cand2 = self.df.copy().dropna(subset=[self.id_col]).reset_index(drop=True)
            cand2["has_graph"] = cand2[self.id_col].astype(str).apply(lambda sid: (self.cache_dir / f"{sid}.pt").exists())
            cand2 = cand2[cand2["has_graph"]]
            cand2 = cand2[cand2[self.city_col] == city_cluster]

            cand2_area_bin = cand2[self.area_col].apply(lambda x: area_bin(x, self.t1, self.t2))
            adj_bins = [b for b in [q_area_bin - 1, q_area_bin + 1] if b in [0, 1, 2]]
            cand_relaxed = cand2[cand2_area_bin.isin([q_area_bin] + adj_bins)].reset_index(drop=True)
            if len(cand_relaxed) > len(cand):
                cand = cand_relaxed
                constraints.append(f"已放宽：允许相邻面积档位 {adj_bins}")

        if len(cand) == 0:
            return {
                "query": query_features,
                "city_cluster": city_cluster,
                "constraints": constraints,
                "topk": [],
                "note": "候选集为空（请开启 soft_relax 或检查 city_cluster/total_area 档位）"
            }

        if max_candidates is not None and len(cand) > max_candidates:
            cand = cand.sample(n=max_candidates, random_state=0).reset_index(drop=True)

        # ---- batch score ----
        Qcand = cand[self.q_cols].fillna(0).astype(float).values
        Mcand = cand[self.m_cols].fillna(0).astype(float).values
        Qcand = self.scaler.transform_q(Qcand)
        Mcand = self.scaler.transform_m(Mcand)

        sids = cand[self.id_col].astype(str).tolist()

        scores = []
        bs = 64
        for st in range(0, len(cand), bs):
            ed = min(st + bs, len(cand))
            bq = torch.tensor(Qcand[st:ed], dtype=torch.float32).to(self.device)
            bm = torch.tensor(Mcand[st:ed], dtype=torch.float32).to(self.device)

            glist = [_load_graph(str(self.cache_dir), sid) for sid in sids[st:ed]]
            gb = Batch.from_data_list(glist).to(self.device)

            with torch.no_grad():
                s, _, _ = self.model(bq, bm, gb.x, gb.edge_index, gb.batch)
            scores.append(s.detach().cpu().numpy())
        scores = np.concatenate(scores, axis=0)

        order = np.argsort(-scores)[:topk]
        top_items = []

        # ---- per-item explanation (TopK) ----
        for rank, idx in enumerate(order, start=1):
            sid = sids[idx]
            row = cand.iloc[idx]

            item = {
                "rank": rank,
                "sample_id": sid,
                "score": float(scores[idx]),
                "total_score_true": float(row[self.label_col]) if self.label_col in row else None,
                "metrics": {m: float(row[m]) if m in row and pd.notna(row[m]) else None for m in self.m_cols},
            }

            if return_attention:
                # 单条再跑一次：得到 attention
                bm1 = torch.tensor(
                    self.scaler.transform_m(row[self.m_cols].fillna(0).astype(float).values.reshape(1, -1)),
                    dtype=torch.float32
                ).to(self.device)
                g1 = Batch.from_data_list([_load_graph(str(self.cache_dir), sid)]).to(self.device)

                with torch.no_grad():
                    s1, attn1, M = self.model(q, bm1, g1.x, g1.edge_index, g1.batch)

                cond_imp, metric_imp, cond_to_metric = summarize_condition_metric_attention(
                    attn1, M=M, query_feature_names=self.q_cols, metric_names=self.m_cols
                )

                # Top metrics
                m_order = np.argsort(-metric_imp)
                top_metrics = [{"name": self.m_cols[i], "contribution": float(metric_imp[i])} for i in m_order]

                # Top driving conditions（用条件对所有指标注意力求和）
                cond_scores = {c: float(sum(w for _, w in cond_to_metric[c])) for c in cond_to_metric.keys()}
                top_conds = sorted(cond_scores.items(), key=lambda x: -x[1])[:3]
                drives = []
                for c, _ in top_conds:
                    drives.append({
                        "condition": c,
                        "top_metrics": [{"name": mn, "weight": float(mw)} for mn, mw in cond_to_metric[c][:3]]
                    })

                item["explanations"] = {
                    "constraints": constraints,
                    "top_metrics": top_metrics[:4],
                    "condition_to_metric": drives
                }

            top_items.append(item)

        return {
            "query": query_features,
            "city_cluster": city_cluster,
            "constraints": constraints,
            "topk": top_items
        }


def load_query_json(path: str) -> Tuple[Dict[str, float], int]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    city = int(obj["city_cluster"])
    feats = obj["query_features"]
    return feats, city


if __name__ == "__main__":
    # 简单命令行：python -m src.infer_ranker --query_json xxx.json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.yaml")
    parser.add_argument("--ckpt", default="./outputs/checkpoints/latest.pt")
    parser.add_argument("--query_json", type=str, default=None)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    svc = RankerService(config_path=args.config, checkpoint_path=args.ckpt)

    if args.query_json is None:
        # 若不传query，使用中位数做示例（仅用于 sanity check）
        df = svc.df
        q = {c: float(df[c].fillna(0).median()) for c in svc.q_cols}
        city_cluster = int(df[svc.city_col].mode().iloc[0])
    else:
        q, city_cluster = load_query_json(args.query_json)

    res = svc.recommend(q, city_cluster=city_cluster, topk=args.topk, hard_filter=True, soft_relax=False)
    Path("./outputs/infer").mkdir(parents=True, exist_ok=True)
    with open("./outputs/infer/topk_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("Saved: outputs/infer/topk_results.json")
