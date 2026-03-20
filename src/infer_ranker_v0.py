import json
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Batch
# Optional: improve determinism/stability on Apple Silicon
torch.set_float32_matmul_precision("high")

from .utils_bucket import area_bin
from .utils_scaler import DualScaler
from .model_ranker import GraphMatchRanker
from .explain import summarize_condition_metric_attention, format_explanation_text

def load_graph(graph_cache_dir: str, sid: str):
    return torch.load(Path(graph_cache_dir) / f"{sid}.pt")

def main(config_path: str = "./config.yaml", checkpoint_path: str = "./outputs/checkpoints/latest.pt"):
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["cfg"]  # 保证一致

    df = pd.read_csv(cfg["data"]["main_table_csv"])
    id_col = cfg["features"]["id_col"]
    q_cols = cfg["features"]["query_cols"]
    m_cols = cfg["features"]["metric_cols"]
    label_col = cfg["features"]["label_col"]

    # scaler
    scaler = DualScaler.load(cfg["cache"]["scaler_path"])

    # Device priority: CUDA (if any) -> Apple Silicon MPS -> CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[Device] Using: {device}")
    model = GraphMatchRanker(
        q_features=len(q_cols),
        metric_features=len(m_cols),
        node_in_dim=3,
        d_model=cfg["model"]["d_model"],
        gnn_layers=cfg["model"]["gnn_layers"],
        cross_heads=cfg["model"]["cross_heads"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ======= 这里模拟一个Query：你实际系统中由前端/配置输入 =======
    # 你可以把这段替换为真实输入
    query_dict = {c: float(df[c].fillna(0).median()) for c in q_cols}  # 示例：用中位数
    q_raw = np.array([query_dict[c] for c in q_cols], dtype=float).reshape(1, -1)
    q = torch.tensor(scaler.transform_q(q_raw), dtype=torch.float32).to(device)  # [1,10]

    # 硬约束过滤：同 city_cluster + 同 total_area 档位
    t1, t2 = cfg["bucket"]["area_thresholds"]
    # 这里 query 的 city_cluster 没在q_cols里，你实际输入时应单独传；示例用df众数
    q_city_cluster = int(df[cfg["features"]["city_cluster_col"]].mode().iloc[0])
    q_area_bin = area_bin(query_dict["total_area"], t1, t2)

    cand = df.copy()
    cand = cand[cand[cfg["features"]["city_cluster_col"]] == q_city_cluster]
    cand = cand[cand[cfg["features"]["total_area_col"]].apply(lambda x: area_bin(x, t1, t2) == q_area_bin)]
    cand = cand.dropna(subset=[id_col]).reset_index(drop=True)

    # 只保留有图缓存的候选
    cache_dir = Path(cfg["cache"]["graph_cache_dir"])
    cand["has_graph"] = cand[id_col].astype(str).apply(lambda sid: (cache_dir / f"{sid}.pt").exists())
    cand = cand[cand["has_graph"]].reset_index(drop=True)

    if len(cand) == 0:
        raise RuntimeError("No candidates after hard filtering. Consider soft relaxation logic.")

    # 批量打分
    Qcand = cand[q_cols].fillna(0).astype(float).values
    Mcand = cand[m_cols].fillna(0).astype(float).values
    Qcand = scaler.transform_q(Qcand)
    Mcand = scaler.transform_m(Mcand)

    scores = []
    attns = []
    sids = cand[id_col].astype(str).tolist()

    bs = 64
    for st in range(0, len(cand), bs):
        ed = min(st + bs, len(cand))
        bq = torch.tensor(Qcand[st:ed], dtype=torch.float32).to(device)
        bm = torch.tensor(Mcand[st:ed], dtype=torch.float32).to(device)

        glist = [load_graph(str(cache_dir), sid) for sid in sids[st:ed]]
        gb = Batch.from_data_list(glist).to(device)

        with torch.no_grad():
            s, attn, M = model(bq, bm, gb.x, gb.edge_index, gb.batch)
        scores.append(s.detach().cpu().numpy())
        attns.append(attn.detach().cpu())
    scores = np.concatenate(scores, axis=0)

    topk = cfg["infer"]["topk"]
    order = np.argsort(-scores)[:topk]

    out = []
    explain_md = ["# Top-K 推荐结果与解释\n"]
    query_show = {k: query_dict[k] for k in q_cols}

    for rank, idx in enumerate(order, start=1):
        sid = sids[idx]
        row = cand.iloc[idx]
        # 取该样本对应的attention（所在batch位置需要重取，简化：推理阶段仅对TopK再单独跑一次以拿attn）
        # 为了稳妥：单样本再算一遍，拿精确attn
        bq = q
        bm = torch.tensor(scaler.transform_m(row[m_cols].fillna(0).astype(float).values.reshape(1, -1)),
                          dtype=torch.float32).to(device)
        g = Batch.from_data_list([load_graph(str(cache_dir), sid)]).to(device)

        with torch.no_grad():
            s1, attn1, M = model(bq, bm, g.x, g.edge_index, g.batch)

        cond_imp, metric_imp, cond_to_metric = summarize_condition_metric_attention(
            attn1, M=M,
            query_feature_names=q_cols,
            metric_names=m_cols
        )

        constraints = [
            f"city_cluster={q_city_cluster} 命中",
            f"total_area 档位={q_area_bin}（阈值 {t1}/{t2}）命中"
        ]

        exp_text, top_metrics = format_explanation_text(
            query_dict=query_show,
            constraints_text=constraints,
            metric_imp=metric_imp,
            metric_names=m_cols,
            cond_to_metric=cond_to_metric
        )

        item = {
            "rank": rank,
            "sample_id": sid,
            "score": float(s1.item()),
            "total_score_true": float(row[label_col]) if label_col in row else None,
            "top_metrics": [{"name": n, "contribution": w} for n, w in top_metrics],
        }
        out.append(item)

        explain_md.append(f"## Rank {rank}: {sid}\n")
        explain_md.append(exp_text)
        explain_md.append("\n---\n")

    Path("./outputs/infer").mkdir(parents=True, exist_ok=True)
    with open("./outputs/infer/topk_results.json", "w", encoding="utf-8") as f:
        json.dump({"query": query_show, "topk": out}, f, ensure_ascii=False, indent=2)

    with open("./outputs/infer/explanations.md", "w", encoding="utf-8") as f:
        f.write("\n".join(explain_md))

    print("Saved:")
    print(" - outputs/infer/topk_results.json")
    print(" - outputs/infer/explanations.md")

if __name__ == "__main__":
    main()