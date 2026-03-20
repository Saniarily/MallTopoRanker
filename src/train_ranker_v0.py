import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
# Optional: improve determinism/stability on Apple Silicon
torch.set_float32_matmul_precision("high")

from .paths import ensure_dir
from .utils_seed import set_seed
from .utils_bucket import make_bucket_id
from .utils_scaler import DualScaler
from .utils_graph_io import build_graph_cache
from .dataset_pairs import sample_pairs_within_bucket, PairwiseRankDataset
from .model_ranker import GraphMatchRanker

from torch_geometric.data import Batch

def _load_graph_pt(graph_cache_dir: str, sample_id: str):
    p = Path(graph_cache_dir) / f"{sample_id}.pt"
    return torch.load(p)

def collate_fn(batch, graph_cache_dir: str):
    # batch: list of dict
    q = torch.stack([b["q"] for b in batch], dim=0)
    m_pos = torch.stack([b["m_pos"] for b in batch], dim=0)
    m_neg = torch.stack([b["m_neg"] for b in batch], dim=0)

    sids_pos = [b["sid_pos"] for b in batch]
    sids_neg = [b["sid_neg"] for b in batch]

    gpos_list = [_load_graph_pt(graph_cache_dir, sid) for sid in sids_pos]
    gneg_list = [_load_graph_pt(graph_cache_dir, sid) for sid in sids_neg]

    gpos = Batch.from_data_list(gpos_list)
    gneg = Batch.from_data_list(gneg_list)

    return q, m_pos, m_neg, gpos, gneg

def pairwise_logistic_loss(s_pos, s_neg):
    # log(1 + exp(-(s_pos - s_neg)))
    return torch.nn.functional.softplus(-(s_pos - s_neg)).mean()

def main(config_path: str = "./config.yaml"):
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    set_seed(cfg["train"]["seed"])
    
    # Generate timestamp for this training session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.read_csv(cfg["data"]["main_table_csv"])
    id_col = cfg["features"]["id_col"]
    label_col = cfg["features"]["label_col"]

    # bucket
    t1, t2 = cfg["bucket"]["area_thresholds"]
    df["bucket_id"] = make_bucket_id(
        df,
        city_col=cfg["features"]["city_cluster_col"],
        area_col=cfg["features"]["total_area_col"],
        t1=t1, t2=t2
    )

    # basic clean
    df = df.dropna(subset=[id_col, label_col, "bucket_id"])
    df = df.reset_index(drop=True)

    q_cols = cfg["features"]["query_cols"]
    m_cols = cfg["features"]["metric_cols"]

    # Extract mall_id from floor_id (format: mall_id_floor_number)
    df["mall_id"] = df[id_col].astype(str).str.rsplit("_", n=1).str[0]

    # Split by mall_id (group-level split to prevent data leakage)
    unique_malls = df["mall_id"].unique()
    np.random.shuffle(unique_malls)
    
    n_malls = len(unique_malls)
    test_n_malls = int(n_malls * cfg["train"]["test_ratio"])
    val_n_malls = int(n_malls * cfg["train"]["val_ratio"])
    
    test_malls = set(unique_malls[:test_n_malls])
    val_malls = set(unique_malls[test_n_malls:test_n_malls+val_n_malls])
    train_malls = set(unique_malls[test_n_malls+val_n_malls:])
    
    test_idx = df[df["mall_id"].isin(test_malls)].index.values
    val_idx = df[df["mall_id"].isin(val_malls)].index.values
    train_idx = df[df["mall_id"].isin(train_malls)].index.values
    
    print(f"[Split] Malls - Train: {len(train_malls)}, Val: {len(val_malls)}, Test: {len(test_malls)}")
    print(f"[Split] Samples - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # build graph cache
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
    from pathlib import Path
    keep_mask = df[id_col].astype(str).apply(lambda sid: (Path(cfg["cache"]["graph_cache_dir"]) / f"{sid}.pt").exists())
    old_len = len(df)
    df = df[keep_mask].reset_index(drop=True)
    print(f"[Filter] Removed {old_len - len(df)} samples with missing graph cache")

    # recompute split after filtering (still by mall_id)
    unique_malls = df["mall_id"].unique()
    np.random.shuffle(unique_malls)
    
    n_malls = len(unique_malls)
    test_n_malls = int(n_malls * cfg["train"]["test_ratio"])
    val_n_malls = int(n_malls * cfg["train"]["val_ratio"])
    
    test_malls = set(unique_malls[:test_n_malls])
    val_malls = set(unique_malls[test_n_malls:test_n_malls+val_n_malls])
    train_malls = set(unique_malls[test_n_malls+val_n_malls:])
    
    test_idx = df[df["mall_id"].isin(test_malls)].index.values
    val_idx = df[df["mall_id"].isin(val_malls)].index.values
    train_idx = df[df["mall_id"].isin(train_malls)].index.values
    
    print(f"[Split after filter] Malls - Train: {len(train_malls)}, Val: {len(val_malls)}, Test: {len(test_malls)}")
    print(f"[Split after filter] Samples - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # matrices
    Q = df[q_cols].fillna(0).astype(float).values
    M = df[m_cols].fillna(0).astype(float).values
    y = df[label_col].fillna(0).astype(float).values

    scaler = DualScaler()
    scaler.fit(Q[train_idx], M[train_idx])
    ensure_dir(str(Path(cfg["cache"]["scaler_path"]).parent))
    scaler.save(cfg["cache"]["scaler_path"])

    Qs = scaler.transform_q(Q)
    Ms = scaler.transform_m(M)

    # sample pairs on train set
    df_train = df.loc[train_idx].copy()
    # 为了让索引对应Qs/Ms/y，需要用原始df的index位置，所以先做一个映射
    # 简化：直接在train子集内重建数组
    Q_train = Qs[train_idx]
    M_train = Ms[train_idx]
    y_train = y[train_idx]
    df_train = df_train.reset_index(drop=True)

    pairs = sample_pairs_within_bucket(
        df=df_train,
        bucket_col="bucket_id",
        id_col=id_col,
        q_mat=Q_train,
        m_mat=M_train,
        y=y_train,
        pairs_per_bucket=cfg["train"]["pairs_per_bucket"],
        seed=cfg["train"]["seed"]
    )
    dataset = PairwiseRankDataset(pairs, cfg["cache"]["graph_cache_dir"])

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size_pairs"],
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, cfg["cache"]["graph_cache_dir"])
    )

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
        node_in_dim=3,  # [Total_L_Neighbors, x, y]
        d_model=cfg["model"]["d_model"],
        gnn_layers=cfg["model"]["gnn_layers"],
        cross_heads=cfg["model"]["cross_heads"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    out_ckpt_dir = ensure_dir("./outputs/checkpoints")
    best_loss = 1e9
    train_loss_history = []  # Record loss for each epoch
    best_epoch = 0  # Track which epoch had the best loss
    
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        losses = []
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg['train']['epochs']}")
        for q, m_pos, m_neg, gpos, gneg in pbar:
            q = q.to(device)
            m_pos = m_pos.to(device)
            m_neg = m_neg.to(device)

            gpos = gpos.to(device)
            gneg = gneg.to(device)

            s_pos, _, _ = model(q, m_pos, gpos.x, gpos.edge_index, gpos.batch)
            s_neg, _, _ = model(q, m_neg, gneg.x, gneg.edge_index, gneg.batch)

            loss = pairwise_logistic_loss(s_pos, s_neg)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optim.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(losses))

        epoch_loss = float(np.mean(losses))
        train_loss_history.append(epoch_loss)
        print(f"Epoch {epoch}: train_loss={epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            
            # Save with timestamp and epoch info
            versioned_name = f"best_epoch{epoch}_{timestamp}.pt"
            versioned_path = os.path.join(out_ckpt_dir, versioned_name)
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "loss": best_loss}, versioned_path)
            
            # Also save as latest.pt for easy loading
            latest_path = os.path.join(out_ckpt_dir, "latest.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "loss": best_loss}, latest_path)
            
            print(f"[Checkpoint] saved {versioned_name} (loss={best_loss:.4f})")
            print(f"[Checkpoint] updated latest.pt")
    
    # Plot and save loss curve with Nature journal style
    plot_loss_curve(train_loss_history, out_ckpt_dir, timestamp)
    
    print(f"\n[Training Complete]")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Timestamp: {timestamp}")


def plot_loss_curve(loss_history, save_dir, timestamp):
    """Plot training loss curve with Nature journal style."""
    # Set Nature journal style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'lines.linewidth': 2.0,
    })
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    epochs = np.arange(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, color='#2E86AB', linewidth=2, marker='o', 
            markersize=4, markerfacecolor='white', markeredgewidth=1.5, 
            markeredgecolor='#2E86AB', label='Training Loss')
    
    ax.set_xlabel('Epoch', fontweight='normal')
    ax.set_ylabel('Loss', fontweight='normal')
    ax.set_title('Training Loss Curve', fontweight='bold', pad=15)
    
    # Grid with subtle style
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    ax.legend(frameon=False, loc='best')
    
    # Ensure integer ticks for epochs
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure with timestamp
    save_path = os.path.join(save_dir, f'loss_curve_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[Plot] Loss curve saved to {save_path}")
    
    # Also save as PDF for publication quality
    pdf_path = os.path.join(save_dir, f'loss_curve_{timestamp}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"[Plot] Loss curve (PDF) saved to {pdf_path}")
    
    # Save latest version for easy access
    latest_png = os.path.join(save_dir, 'loss_curve_latest.png')
    plt.savefig(latest_png, dpi=300, bbox_inches='tight', facecolor='white')
    latest_pdf = os.path.join(save_dir, 'loss_curve_latest.pdf')
    plt.savefig(latest_pdf, bbox_inches='tight', facecolor='white')
    
    plt.close()
    
    # Reset matplotlib parameters to default
    plt.rcdefaults()

if __name__ == "__main__":
    main()