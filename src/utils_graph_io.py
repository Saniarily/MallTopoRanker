import re
from pathlib import Path
import pandas as pd
import torch
from torch_geometric.data import Data

_CENTER_RE = re.compile(r"\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)")

def parse_centerpoint(s: str):
    if pd.isna(s):
        return None
    m = _CENTER_RE.search(str(s))
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))

def load_graph_from_files(edge_csv: str, node_csv: str) -> Data:
    # edges
    edf = pd.read_csv(edge_csv, sep=None, engine="python")
    if not {"Source", "Target"}.issubset(set(edf.columns)):
        raise ValueError(f"Edge file columns must include Source/Target, got {edf.columns.tolist()}")

    # nodes
    ndf = pd.read_csv(node_csv, sep=None, engine="python")
    if not {"Node_ID", "Total_L_Neighbors"}.issubset(set(ndf.columns)):
        raise ValueError(f"Node attr file columns must include Node_ID/Total_L_Neighbors, got {ndf.columns.tolist()}")

    # build node index
    node_ids = ndf["Node_ID"].astype(str).tolist()
    idx = {nid: i for i, nid in enumerate(node_ids)}

    # map edges to indices (drop edges with missing nodes)
    src = []
    tgt = []
    for s, t in zip(edf["Source"].astype(str), edf["Target"].astype(str)):
        if s in idx and t in idx:
            src.append(idx[s])
            tgt.append(idx[t])

    edge_index = torch.tensor([src + tgt, tgt + src], dtype=torch.long)  # 无向图：双向加边

    # node features: [Total_L_Neighbors, x, y]
    total_l = torch.tensor(ndf["Total_L_Neighbors"].fillna(0).astype(float).values, dtype=torch.float32).view(-1, 1)

    if "CenterPoint" in ndf.columns:
        xy = [parse_centerpoint(v) for v in ndf["CenterPoint"].tolist()]
        # 若缺失则置0
        xs = torch.tensor([p[0] if p else 0.0 for p in xy], dtype=torch.float32).view(-1, 1)
        ys = torch.tensor([p[1] if p else 0.0 for p in xy], dtype=torch.float32).view(-1, 1)
        coords = torch.cat([xs, ys], dim=1)
        # per-graph normalize to improve泛化：中心化 + 尺度归一
        coords = coords - coords.mean(dim=0, keepdim=True)
        denom = coords.abs().max(dim=0, keepdim=True).values.clamp(min=1.0)
        coords = coords / denom
    else:
        coords = torch.zeros((len(node_ids), 2), dtype=torch.float32)

    x = torch.cat([total_l, coords], dim=1)  # [N, 3]
    data = Data(x=x, edge_index=edge_index)
    data.node_ids = node_ids
    return data

def build_graph_cache(sample_ids, graph_dir: str, edge_suffix: str, node_suffix: str, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    graph_dir = Path(graph_dir)

    missing = 0
    for sid in sample_ids:
        out_path = Path(out_dir) / f"{sid}.pt"
        if out_path.exists():
            continue

        edge_path = graph_dir / f"{sid}{edge_suffix}"
        node_path = graph_dir / f"{sid}{node_suffix}"
        if (not edge_path.exists()) or (not node_path.exists()):
            missing += 1
            continue

        g = load_graph_from_files(str(edge_path), str(node_path))
        torch.save(g, out_path)

    return missing