import numpy as np
import pandas as pd


def _to_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").astype(float)

def area_bin(total_area: float, t1: float, t2: float) -> int:
    if pd.isna(total_area):
        return -1
    if total_area < t1:
        return 0
    if total_area < t2:
        return 1
    return 2

def make_bucket_id(df: pd.DataFrame, city_col: str, area_col: str, t1: float, t2: float) -> pd.Series:
    bins = df[area_col].apply(lambda x: area_bin(x, t1, t2))
    # city_cluster 可能不是 0/1/2，但 bucket 仍然可用原值
    return df[city_col].astype(str) + "_" + bins.astype(str)


def fit_query_bin_edges(train_df: pd.DataFrame, query_bin_specs: dict) -> dict:
    """
    Fit per-query-feature bin edges from train split only.

    query_bin_specs format:
      {
        "people": {"n_bins": 4},
        "Tx": {"n_bins": 3},
        "nearest_distance_km": {"edges": [1.0, 3.0, 5.0]}
      }
    """
    edge_map = {}
    for col, spec in query_bin_specs.items():
        if col not in train_df.columns:
            raise ValueError(f"query_bin_specs column '{col}' not found in dataframe")

        # Priority: explicit edges
        if isinstance(spec, dict) and "edges" in spec:
            edges = sorted(float(x) for x in spec.get("edges", []))
            edge_map[col] = edges
            continue

        # Default: quantile-based bins
        n_bins = int(spec.get("n_bins", 3)) if isinstance(spec, dict) else 3
        n_bins = max(2, n_bins)

        s = _to_float_series(train_df, col)
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) == 0:
            edge_map[col] = []
            continue

        qs = np.linspace(0.0, 1.0, n_bins + 1)
        q_vals = np.quantile(s.values, qs)
        q_vals = np.unique(q_vals)

        # internal cut points only
        edges = [float(v) for v in q_vals[1:-1]] if len(q_vals) > 2 else []
        edge_map[col] = edges

    return edge_map


def make_multidim_profile_bucket_id(df: pd.DataFrame, city_col: str, edge_map: dict) -> pd.Series:
    """
    Build profile bucket id by combining city cluster and multiple query-feature bins.
    """
    out = df[city_col].astype(str)
    for col in edge_map.keys():
        s = _to_float_series(df, col)
        edges = edge_map.get(col, [])

        # np.digitize returns bin index in [0, len(edges)]
        bins = np.digitize(s.fillna(-1e18).values, bins=np.array(edges, dtype=float), right=False)
        bins = pd.Series(bins, index=df.index)

        # Missing values get dedicated bin -1
        bins[s.isna()] = -1

        out = out + "|" + str(col) + "_b" + bins.astype(int).astype(str)

    return out