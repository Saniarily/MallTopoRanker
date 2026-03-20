# src/train_ranker.py
import os
import argparse
from functools import lru_cache
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .paths import ensure_dir
from .utils_seed import set_seed
from .utils_bucket import make_bucket_id, area_bin, fit_query_bin_edges, make_multidim_profile_bucket_id
from .utils_scaler import DualScaler
from .utils_graph_io import build_graph_cache
from .dataset_pairs import compute_bucket_query_map, PointSample, PointwiseRankDataset
from .model_ranker import GraphMatchRanker

from torch_geometric.data import Batch


def _select_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using: {device}")
    return device


def _load_graph_pt(graph_cache_dir: str, sample_id: str):
    return _load_graph_pt_cached(graph_cache_dir, sample_id)


@lru_cache(maxsize=20000)
def _load_graph_pt_cached(graph_cache_dir: str, sample_id: str):
    p = Path(graph_cache_dir) / f"{sample_id}.pt"
    # Explicit weights_only avoids repeated FutureWarning noise in tight loops.
    return torch.load(p, weights_only=False)


def collate_fn_pointwise(batch, graph_cache_dir: str):
    q = torch.stack([b["q"] for b in batch], dim=0)
    m = torch.stack([b["m"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    bucket_ids = [str(b["bucket_id"]) for b in batch]

    sids = [b["sid"] for b in batch]
    g_list = [_load_graph_pt(graph_cache_dir, sid) for sid in sids]
    g = Batch.from_data_list(g_list)

    return q, m, g, y, bucket_ids


def _bucket_ids_to_tensor(bucket_ids, device: torch.device) -> torch.Tensor:
    mapping = {}
    out = []
    for b in bucket_ids:
        if b not in mapping:
            mapping[b] = len(mapping)
        out.append(mapping[b])
    return torch.tensor(out, dtype=torch.long, device=device)


def inbatch_pairwise_logistic_loss(scores: torch.Tensor, labels: torch.Tensor, bucket_ids: list):
    """
    Pairwise logistic loss built inside a pointwise batch.
    Only compare samples from the same bucket and with different labels.
    """
    bucket_idx = _bucket_ids_to_tensor(bucket_ids, device=scores.device)
    same_bucket = bucket_idx.unsqueeze(1).eq(bucket_idx.unsqueeze(0))

    score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    label_diff = labels.unsqueeze(1) - labels.unsqueeze(0)

    valid_pairs = (label_diff > 0) & same_bucket
    num_pairs = int(valid_pairs.sum().item())
    if num_pairs == 0:
        return None, 0

    loss = torch.nn.functional.softplus(-score_diff[valid_pairs]).mean()
    return loss, num_pairs


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


def _minmax01(x: np.ndarray) -> np.ndarray:
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-9:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)


def _dcg_at_k(rels: np.ndarray, k: int) -> float:
    rels = rels[:k]
    if len(rels) == 0:
        return 0.0
    denom = np.log2(np.arange(2, len(rels) + 2))
    return float(np.sum((2 ** rels - 1) / denom))


def _ndcg_at_k(rels_pred_order: np.ndarray, rels_true_sorted: np.ndarray, k: int) -> float:
    dcg = _dcg_at_k(rels_pred_order, k)
    idcg = _dcg_at_k(rels_true_sorted, k)
    return float(dcg / idcg) if idcg > 0 else 0.0


def _batch_score_candidates(model, device, scaler, graph_cache_dir: str,
                            q_vec_std: np.ndarray,
                            cand_metric_std: np.ndarray,
                            cand_ids: list,
                            batch_size: int = 64):
    """
    固定一个 query（已标准化），对一批候选图（metric_std + graph）打分。
    q_vec_std: shape [Fq] 标准化后
    cand_metric_std: shape [Ncand, Fm]
    """
    scores = []
    q_t = torch.tensor(q_vec_std, dtype=torch.float32, device=device).view(1, -1)  # [1,Fq]

    for st in range(0, len(cand_ids), batch_size):
        ed = min(st + batch_size, len(cand_ids))
        B = ed - st

        bq = q_t.repeat(B, 1)  # [B,Fq]
        bm = torch.tensor(cand_metric_std[st:ed], dtype=torch.float32, device=device)

        glist = [_load_graph_pt(graph_cache_dir, sid) for sid in cand_ids[st:ed]]
        gb = Batch.from_data_list(glist).to(device)

        with torch.no_grad():
            s, _, _ = model(bq, bm, gb.x, gb.edge_index, gb.batch)
        scores.append(s.detach().cpu().numpy())

    return np.concatenate(scores, axis=0)


def evaluate_inbatch_pairwise_accuracy(model, device, data_loader, max_batches: int = 50):
    """
    Evaluate pairwise accuracy from pointwise batches (same-bucket comparisons only).
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for bi, (q, m, g, y, bucket_ids) in enumerate(data_loader):
            if bi >= max_batches:
                break
            q = q.to(device)
            m = m.to(device)
            g = g.to(device)
            y = y.to(device)

            s, _, _ = model(q, m, g.x, g.edge_index, g.batch)

            bucket_idx = _bucket_ids_to_tensor(bucket_ids, device=device)
            same_bucket = bucket_idx.unsqueeze(1).eq(bucket_idx.unsqueeze(0))

            score_diff = s.unsqueeze(1) - s.unsqueeze(0)
            label_diff = y.unsqueeze(1) - y.unsqueeze(0)

            valid_pairs = (label_diff != 0) & same_bucket
            if valid_pairs.any():
                pred_order = score_diff[valid_pairs] > 0
                true_order = label_diff[valid_pairs] > 0
                correct += int((pred_order == true_order).sum().item())
                total += int(valid_pairs.sum().item())
    return float(correct / total) if total > 0 else 0.0


def evaluate_val_ndcg(
    model, device, scaler, graph_cache_dir: str,
    val_df: pd.DataFrame,
    q_cols: list, m_cols: list,
    bucket_col: str, id_col: str, label_col: str,
    topk_list=(10, 20),
    max_queries_per_bucket: int = 20,
    max_candidates_per_query: int = 300,
    seed: int = 0
):
    """
    listwise 验证：在 val 集内，每个 bucket 抽样若干 query，
    候选为同 bucket 的其他样本（抽样候选），计算 NDCG@K（连续相关性）。
    """
    rng = np.random.RandomState(seed)
    model.eval()

    ndcg_sums = {k: 0.0 for k in topk_list}
    ndcg_cnt = 0

    # 预先按 bucket 分组（只用 val 内部）
    groups = {bid: g.reset_index(drop=True) for bid, g in val_df.groupby(bucket_col)}

    for bid, g in groups.items():
        if len(g) < 6:
            continue

        qg = g.sample(n=min(max_queries_per_bucket, len(g)), random_state=int(seed)).reset_index(drop=True)

        for _, qrow in qg.iterrows():
            qid = str(qrow[id_col])

            # candidates in same bucket, exclude itself
            cand = g[g[id_col].astype(str) != qid].reset_index(drop=True)
            if len(cand) < 5:
                continue

            if len(cand) > max_candidates_per_query:
                cand = cand.sample(n=max_candidates_per_query, random_state=int(seed)).reset_index(drop=True)

            # build query vector (raw -> std)
            q_raw = np.array([float(qrow[c]) for c in q_cols], dtype=float).reshape(1, -1)
            q_std = scaler.transform_q(q_raw).reshape(-1)  # [Fq]

            # candidate metric std
            m_raw = cand[m_cols].fillna(0).astype(float).values
            m_std = scaler.transform_m(m_raw)

            cand_ids = cand[id_col].astype(str).tolist()
            scores = _batch_score_candidates(model, device, scaler, graph_cache_dir, q_std, m_std, cand_ids, batch_size=64)

            y_true = cand[label_col].fillna(0).astype(float).values
            rel = _minmax01(y_true)  # 0..1 graded relevance
            ideal = rel[np.argsort(-rel)]

            pred_order = np.argsort(-scores)
            rel_pred = rel[pred_order]

            for k in topk_list:
                ndcg_sums[k] += _ndcg_at_k(rel_pred, ideal, k)
            ndcg_cnt += 1

    if ndcg_cnt == 0:
        return {f"ndcg@{k}": 0.0 for k in topk_list}

    return {f"ndcg@{k}": float(ndcg_sums[k] / ndcg_cnt) for k in topk_list}


def main(config_path: str = "./config.yaml"):
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    # Seed protocol (for reproducible and comparable experiments):
    # - run_seed controls model/init/shuffle randomness per run
    # - split/pair/val_eval seeds are fixed by default to avoid data protocol drift across runs
    run_seed = int(cfg["train"]["seed"])
    split_seed = int(cfg["train"].get("split_seed", 42))
    pair_train_seed = int(cfg["train"].get("pair_train_seed", 43))
    pair_val_seed = int(cfg["train"].get("pair_val_seed", 44))

    set_seed(run_seed)

    df = pd.read_csv(cfg["data"]["main_table_csv"])
    id_col = cfg["features"].get("id_col", "floor_id")
    label_col = cfg["features"].get("label_col", "total_score")
    mall_col = cfg.get("features", {}).get("mall_id_col", "mall_id")

    if mall_col not in df.columns:
        raise ValueError(f"mall_id_col='{mall_col}' not found in main_table columns.")

    # bucket
    t1, t2 = cfg["bucket"]["area_thresholds"]
    df["bucket_id"] = make_bucket_id(df, cfg["features"]["city_cluster_col"], cfg["features"]["total_area_col"], t1, t2)

    # basic clean
    df = df.dropna(subset=[id_col, label_col, "bucket_id"]).reset_index(drop=True)

    q_cols = cfg["features"]["query_cols"]
    m_cols = cfg["features"]["metric_cols"]

    # build graph cache (for all)
    ensure_dir(cfg["cache"]["graph_cache_dir"])
    missing = build_graph_cache(
        sample_ids=df[id_col].astype(str).tolist(),
        graph_dir=cfg["data"]["graph_dir"],
        edge_suffix=cfg["data"]["graph_suffix_edge"],
        node_suffix=cfg["data"]["graph_suffix_node"],
        out_dir=cfg["cache"]["graph_cache_dir"]
    )
    print(f"[GraphCache] missing files: {missing}")

    # filter rows whose graph cache missing
    cache_dir = Path(cfg["cache"]["graph_cache_dir"])
    df["has_graph"] = df[id_col].astype(str).apply(lambda sid: (cache_dir / f"{sid}.pt").exists())
    df = df[df["has_graph"]].reset_index(drop=True)

    # split by mall_id
    train_df, val_df, test_df = split_by_mall_id(
        df, mall_col=mall_col, seed=split_seed,
        test_ratio=cfg["train"]["test_ratio"], val_ratio=cfg["train"]["val_ratio"]
    )
    print(f"[Split] train={len(train_df)} val={len(val_df)} test={len(test_df)} (by {mall_col})")

    # matrices for scaler fit
    Q_train_raw = train_df[q_cols].fillna(0).astype(float).values
    M_train_raw = train_df[m_cols].fillna(0).astype(float).values

    scaler = DualScaler()
    scaler.fit(Q_train_raw, M_train_raw)
    ensure_dir(str(Path(cfg["cache"]["scaler_path"]).parent))
    scaler.save(cfg["cache"]["scaler_path"])

    # Keep area bucket mode for this experiment.
    pair_bucket_mode = str(cfg.get("train", {}).get("pair_bucket_mode", "area")).lower()
    city_col = cfg["features"]["city_cluster_col"]

    train_df_reset = train_df.reset_index(drop=True)
    val_df_reset = val_df.reset_index(drop=True)
    test_df_reset = test_df.reset_index(drop=True)

    if pair_bucket_mode == "multi_query":
        query_bin_specs = cfg.get("bucket", {}).get("query_bin_specs", {})
        if not query_bin_specs:
            raise ValueError("train.pair_bucket_mode='multi_query' requires bucket.query_bin_specs in config")

        edge_map = fit_query_bin_edges(train_df_reset, query_bin_specs)
        train_df_reset["pair_bucket_id"] = make_multidim_profile_bucket_id(train_df_reset, city_col, edge_map)
        val_df_reset["pair_bucket_id"] = make_multidim_profile_bucket_id(val_df_reset, city_col, edge_map)
        test_df_reset["pair_bucket_id"] = make_multidim_profile_bucket_id(test_df_reset, city_col, edge_map)
    else:
        train_df_reset["pair_bucket_id"] = train_df_reset["bucket_id"]
        val_df_reset["pair_bucket_id"] = val_df_reset["bucket_id"]
        test_df_reset["pair_bucket_id"] = test_df_reset["bucket_id"]

    # standardized arrays for pair sampling
    Q_train = scaler.transform_q(Q_train_raw)
    M_train = scaler.transform_m(M_train_raw)
    y_train = train_df[label_col].fillna(0).astype(float).values

    Q_val = scaler.transform_q(val_df[q_cols].fillna(0).astype(float).values)
    M_val = scaler.transform_m(val_df[m_cols].fillna(0).astype(float).values)
    y_val = val_df[label_col].fillna(0).astype(float).values

    Q_test = scaler.transform_q(test_df[q_cols].fillna(0).astype(float).values)
    M_test = scaler.transform_m(test_df[m_cols].fillna(0).astype(float).values)
    y_test = test_df[label_col].fillna(0).astype(float).values

    # ---- Query strategy for pair sampling ----
    # use_bucket_query=True: use one representative query per bucket (recommended).
    # use_bucket_query=False: fallback to legacy behavior (query from positive sample).
    use_bucket_query = bool(cfg.get("train", {}).get("use_bucket_query", True))

    # ---- Compute bucket-level representative queries ----
    # This ensures training distribution matches inference (external query vs. samples)
    bucket_query_map_train = compute_bucket_query_map(train_df_reset, "pair_bucket_id", Q_train) if use_bucket_query else None
    bucket_query_map_val = compute_bucket_query_map(val_df_reset, "pair_bucket_id", Q_val) if use_bucket_query else None
    bucket_query_map_test = compute_bucket_query_map(test_df_reset, "pair_bucket_id", Q_test) if use_bucket_query else None

    def _build_point_samples(df_reset, q_mat, m_mat, y_vec, bucket_query_map):
        points = []
        for i in range(len(df_reset)):
            b = str(df_reset.loc[i, "pair_bucket_id"])
            if bucket_query_map is not None and b in bucket_query_map:
                q_use = bucket_query_map[b]
            else:
                q_use = q_mat[i]
            points.append(PointSample(
                q=q_use,
                m=m_mat[i],
                sid=str(df_reset.loc[i, id_col]),
                y=float(y_vec[i]),
                bucket_id=b,
            ))
        return points

    print(f"[PointSampling] building pointwise datasets with bucket mode='{pair_bucket_mode}'")
    points_train = _build_point_samples(train_df_reset, Q_train, M_train, y_train, bucket_query_map_train)
    points_val = _build_point_samples(val_df_reset, Q_val, M_val, y_val, bucket_query_map_val)
    points_test = _build_point_samples(test_df_reset, Q_test, M_test, y_test, bucket_query_map_test)

    dataset_train = PointwiseRankDataset(points_train, cfg["cache"]["graph_cache_dir"])
    dataset_val = PointwiseRankDataset(points_val, cfg["cache"]["graph_cache_dir"])
    dataset_test = PointwiseRankDataset(points_test, cfg["cache"]["graph_cache_dir"])

    loader_train = DataLoader(
        dataset_train,
        batch_size=cfg["train"]["batch_size_pairs"],
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_fn_pointwise(b, cfg["cache"]["graph_cache_dir"])
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=cfg["train"]["batch_size_pairs"],
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn_pointwise(b, cfg["cache"]["graph_cache_dir"])
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=cfg["train"]["batch_size_pairs"],
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn_pointwise(b, cfg["cache"]["graph_cache_dir"])
    )
    print(f"[Loader] train_batches={len(loader_train)} val_batches={len(loader_val)} test_batches={len(loader_test)}")

    device = _select_device()

    reg_cfg = cfg.get("regularization", {})
    effective_dropout = max(float(cfg["model"]["dropout"]), float(reg_cfg.get("min_dropout", 0.2)))
    effective_weight_decay = max(float(cfg["train"]["weight_decay"]), float(reg_cfg.get("min_weight_decay", 1e-3)))

    model = GraphMatchRanker(
        q_features=len(q_cols),
        metric_features=len(m_cols),
        node_in_dim=3,  # [Total_L_Neighbors, x, y]
        d_model=cfg["model"]["d_model"],
        gnn_layers=cfg["model"]["gnn_layers"],
        cross_heads=cfg["model"]["cross_heads"],
        dropout=effective_dropout,
    ).to(device)

    optimizer_name = str(cfg["train"].get("optimizer", "adamw")).lower()
    if optimizer_name == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=effective_weight_decay)
    elif optimizer_name == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=effective_weight_decay)
    elif optimizer_name == "radam":
        optim = torch.optim.RAdam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=effective_weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Use one of: adamw, adam, radam")

    scheduler_cfg = cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="max",
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 2)),
        min_lr=float(scheduler_cfg.get("min_lr", 1e-6))
    )

    outputs_cfg = cfg.get("outputs", {})
    out_ckpt_dir = ensure_dir(outputs_cfg.get("checkpoint_dir", "./outputs/checkpoints"))
    log_dir = Path(outputs_cfg.get("train_log_dir", "./outputs/train_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "epoch_metrics.csv"

    # val eval configs (可不写进 config，默认也能跑)
    val_eval_cfg = cfg.get("val_eval", {})
    val_pair_batches = int(val_eval_cfg.get("max_val_pair_batches", 50))
    val_queries_per_bucket = int(val_eval_cfg.get("max_queries_per_bucket", 20))
    val_candidates_per_query = int(val_eval_cfg.get("max_candidates_per_query", 300))
    val_eval_seed = int(val_eval_cfg.get("seed", split_seed + 123))
    val_dynamic_seed = bool(val_eval_cfg.get("dynamic_seed", False))

    early_stop_cfg = cfg.get("early_stopping", {})
    early_stop_patience = int(early_stop_cfg.get("patience", 5))
    early_stop_min_delta = float(early_stop_cfg.get("min_delta", 1e-4))
    no_improve_epochs = 0

    print(f"[Regularization] dropout={effective_dropout:.3f}, weight_decay={effective_weight_decay:.6f}")
    print(f"[Optimizer] {optimizer_name} | lr={float(cfg['train']['lr']):.6g}")
    print(f"[SeedProtocol] run_seed={run_seed}, split_seed={split_seed}, pair_train_seed={pair_train_seed}, pair_val_seed={pair_val_seed}")
    print(f"[PairSampling] use_bucket_query={use_bucket_query}")
    print(f"[PairSampling] pair_bucket_mode={pair_bucket_mode}")
    print(f"[Scheduler] ReduceLROnPlateau(factor={float(scheduler_cfg.get('factor', 0.5))}, patience={int(scheduler_cfg.get('patience', 2))})")
    print(f"[EarlyStopping] patience={early_stop_patience}, min_delta={early_stop_min_delta}")
    print(f"[ValEval] dynamic_seed={val_dynamic_seed}, base_seed={val_eval_seed}")

    best_train_loss = 1e9
    best_val_ndcg10 = -1.0

    test_eval_cfg = cfg.get("test_eval", {})
    test_pair_batches = int(test_eval_cfg.get("max_test_pair_batches", 30))
    test_queries_per_bucket = int(test_eval_cfg.get("max_queries_per_bucket", 12))
    test_candidates_per_query = int(test_eval_cfg.get("max_candidates_per_query", 300))
    test_eval_seed = int(test_eval_cfg.get("seed", split_seed + 321))

    # write header
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,lr,train_loss,val_pairwise_acc,val_ndcg@10,val_ndcg@20,test_pairwise_acc,test_ndcg@10,test_ndcg@20\n")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        losses = []
        pbar = tqdm(loader_train, desc=f"Epoch {epoch}/{cfg['train']['epochs']}")

        for q, m, g, y, bucket_ids in pbar:
            q = q.to(device)
            m = m.to(device)
            y = y.to(device)
            g = g.to(device)

            s, _, _ = model(q, m, g.x, g.edge_index, g.batch)

            loss, num_pairs = inbatch_pairwise_logistic_loss(s, y, bucket_ids)
            if loss is None:
                continue

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optim.step()

            losses.append(loss.item())
            pbar.set_postfix(train_loss=float(np.mean(losses)))

        train_loss = float(np.mean(losses))

        # ---- validation metrics per epoch ----
        val_pair_acc = evaluate_inbatch_pairwise_accuracy(model, device, loader_val, max_batches=val_pair_batches)

        ndcg_seed = cfg["train"]["seed"] + epoch if val_dynamic_seed else val_eval_seed
        ndcg_dict = evaluate_val_ndcg(
            model=model,
            device=device,
            scaler=scaler,
            graph_cache_dir=cfg["cache"]["graph_cache_dir"],
            val_df=val_df,
            q_cols=q_cols,
            m_cols=m_cols,
            bucket_col="bucket_id",
            id_col=id_col,
            label_col=label_col,
            topk_list=(10, 20),
            max_queries_per_bucket=val_queries_per_bucket,
            max_candidates_per_query=val_candidates_per_query,
            seed=ndcg_seed
        )
        val_ndcg10 = float(ndcg_dict.get("ndcg@10", 0.0))
        val_ndcg20 = float(ndcg_dict.get("ndcg@20", 0.0))

        test_pair_acc = evaluate_inbatch_pairwise_accuracy(model, device, loader_test, max_batches=test_pair_batches)
        test_ndcg_dict = evaluate_val_ndcg(
            model=model,
            device=device,
            scaler=scaler,
            graph_cache_dir=cfg["cache"]["graph_cache_dir"],
            val_df=test_df,
            q_cols=q_cols,
            m_cols=m_cols,
            bucket_col="bucket_id",
            id_col=id_col,
            label_col=label_col,
            topk_list=(10, 20),
            max_queries_per_bucket=test_queries_per_bucket,
            max_candidates_per_query=test_candidates_per_query,
            seed=test_eval_seed
        )
        test_ndcg10 = float(test_ndcg_dict.get("ndcg@10", 0.0))
        test_ndcg20 = float(test_ndcg_dict.get("ndcg@20", 0.0))

        scheduler.step(val_ndcg10)

        current_lr = float(optim.param_groups[0]["lr"])

        print(
            f"Epoch {epoch}: lr={current_lr:.6g} | train_loss={train_loss:.4f} "
            f"| val_pair_acc={val_pair_acc:.4f} | val_ndcg@10={val_ndcg10:.4f} val_ndcg@20={val_ndcg20:.4f} "
            f"| test_pair_acc={test_pair_acc:.4f} | test_ndcg@10={test_ndcg10:.4f} test_ndcg@20={test_ndcg20:.4f}"
        )

        # write log
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{current_lr:.8f},{train_loss:.6f},{val_pair_acc:.6f},{val_ndcg10:.6f},{val_ndcg20:.6f},"
                f"{test_pair_acc:.6f},{test_ndcg10:.6f},{test_ndcg20:.6f}\n"
            )

        # checkpoint by train loss (兼容你原有逻辑)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save({"model": model.state_dict(), "cfg": cfg}, os.path.join(out_ckpt_dir, "best_by_trainloss.pt"))
            print(f"[Checkpoint] saved best_by_trainloss.pt (train_loss={best_train_loss:.4f})")

        # checkpoint by val ndcg@10（推荐论文用这个）
        if val_ndcg10 > best_val_ndcg10 + early_stop_min_delta:
            best_val_ndcg10 = val_ndcg10
            checkpoint_data = {
                "model": model.state_dict(),
                "cfg": cfg,
                "epoch": epoch,
                "val_ndcg@10": best_val_ndcg10,
                "test_ndcg@10": test_ndcg10,
                "test_ndcg@20": test_ndcg20,
                "test_pairwise_acc": test_pair_acc,
            }
            torch.save(checkpoint_data, os.path.join(out_ckpt_dir, "best_by_valndcg10.pt"))
            # Also save as latest.pt for easy inference
            torch.save(checkpoint_data, os.path.join(out_ckpt_dir, "latest.pt"))
            print(f"[Checkpoint] saved best_by_valndcg10.pt and latest.pt (val_ndcg@10={best_val_ndcg10:.4f})")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stop_patience:
            print(f"[EarlyStopping] no val_ndcg@10 improvement for {no_improve_epochs} epochs. Stop at epoch {epoch}.")
            break

    summary_path = log_dir / "test_metrics_last_epoch.csv"
    if log_path.exists():
        em = pd.read_csv(log_path)
        if len(em) > 0:
            em.tail(1).to_csv(summary_path, index=False)

    print(f"[Done] Logs saved to {log_path}")
    print("[Done] Run: python -m src.plot_paper_figures  (to generate curves)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MallTopoRanker model")
    parser.add_argument("--config", type=str, default="./config.yaml", help="Path to yaml config file")
    args = parser.parse_args()
    main(config_path=args.config)
