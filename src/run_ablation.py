import argparse
import copy
import csv
import datetime as dt
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

import yaml


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def set_by_dotted_key(data: dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    cur = data
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def slugify(value) -> str:
    s = str(value)
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "value"


def safe_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def summarize_metrics(metrics_csv: Path) -> dict:
    if not metrics_csv.exists():
        return {
            "best_epoch": -1,
            "best_ndcg10": 0.0,
            "best_ndcg20": 0.0,
            "best_pair_acc": 0.0,
            "last_epoch": -1,
            "last_train_loss": 0.0,
        }

    rows = []
    with open(metrics_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {
            "best_epoch": -1,
            "best_ndcg10": 0.0,
            "best_ndcg20": 0.0,
            "best_pair_acc": 0.0,
            "last_epoch": -1,
            "last_train_loss": 0.0,
        }

    best_row = max(rows, key=lambda r: safe_float(r, "val_ndcg@10", 0.0))
    last_row = rows[-1]
    return {
        "best_epoch": int(float(best_row.get("epoch", -1))),
        "best_ndcg10": safe_float(best_row, "val_ndcg@10", 0.0),
        "best_ndcg20": safe_float(best_row, "val_ndcg@20", 0.0),
        "best_pair_acc": safe_float(best_row, "val_pairwise_acc", 0.0),
        "last_epoch": int(float(last_row.get("epoch", -1))),
        "last_train_loss": safe_float(last_row, "train_loss", 0.0),
    }


def run_one_experiment(train_module: str, config_path: Path, run_dir: Path) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "train_stdout.log"

    cmd = [sys.executable, "-m", train_module, "--config", str(config_path)]
    with open(stdout_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return int(proc.returncode)


def write_csv(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_num(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _group_rows_for_paper(rows: list) -> list:
    groups = {}
    for r in rows:
        key = (str(r.get("factor", "")), str(r.get("key", "")), str(r.get("value", "")))
        groups.setdefault(key, []).append(r)

    out = []
    for (factor, key, value), gr in groups.items():
        ok = [x for x in gr if int(_safe_num(x.get("return_code", 1), 1)) == 0]
        use = ok if ok else gr

        ndcg10 = [_safe_num(x.get("best_ndcg10", 0.0)) for x in use]
        ndcg20 = [_safe_num(x.get("best_ndcg20", 0.0)) for x in use]
        pair_acc = [_safe_num(x.get("best_pair_acc", 0.0)) for x in use]
        last_loss = [_safe_num(x.get("last_train_loss", 0.0)) for x in use]
        last_epoch = [_safe_num(x.get("last_epoch", 0.0)) for x in use]

        out.append({
            "factor": factor,
            "key": key,
            "value": value,
            "n_seeds": len(gr),
            "n_success": len(ok),
            "success_rate": (len(ok) / len(gr)) if len(gr) > 0 else 0.0,
            "mean_best_ndcg10": mean(ndcg10) if ndcg10 else 0.0,
            "std_best_ndcg10": stdev(ndcg10) if len(ndcg10) > 1 else 0.0,
            "mean_best_ndcg20": mean(ndcg20) if ndcg20 else 0.0,
            "std_best_ndcg20": stdev(ndcg20) if len(ndcg20) > 1 else 0.0,
            "mean_best_pair_acc": mean(pair_acc) if pair_acc else 0.0,
            "std_best_pair_acc": stdev(pair_acc) if len(pair_acc) > 1 else 0.0,
            "mean_last_train_loss": mean(last_loss) if last_loss else 0.0,
            "std_last_train_loss": stdev(last_loss) if len(last_loss) > 1 else 0.0,
            "mean_last_epoch": mean(last_epoch) if last_epoch else 0.0,
        })

    out.sort(key=lambda x: (x["factor"], -x["mean_best_ndcg10"]))
    return out


def _best_per_factor(grouped_rows: list) -> list:
    best_map = {}
    for r in grouped_rows:
        f = r["factor"]
        if f not in best_map or r["mean_best_ndcg10"] > best_map[f]["mean_best_ndcg10"]:
            best_map[f] = r
    out = list(best_map.values())
    out.sort(key=lambda x: x["factor"])
    return out


def write_markdown_table(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("No rows.\n")
        return

    header = [
        "factor", "value", "n", "success",
        "NDCG@10 (mean±std)", "NDCG@20 (mean±std)",
        "PairAcc (mean±std)", "LastLoss (mean±std)", "LastEpoch(mean)"
    ]
    sep = ["---"] * len(header)

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")

    for r in rows:
        line = [
            str(r["factor"]),
            str(r["value"]),
            f"{int(r['n_seeds'])}",
            f"{int(r['n_success'])}/{int(r['n_seeds'])}",
            f"{r['mean_best_ndcg10']:.6f}+-{r['std_best_ndcg10']:.6f}",
            f"{r['mean_best_ndcg20']:.6f}+-{r['std_best_ndcg20']:.6f}",
            f"{r['mean_best_pair_acc']:.6f}+-{r['std_best_pair_acc']:.6f}",
            f"{r['mean_last_train_loss']:.6f}+-{r['std_last_train_loss']:.6f}",
            f"{r['mean_last_epoch']:.2f}",
        ]
        lines.append("| " + " | ".join(line) + " |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_paper_summaries(summary_root: Path, overall_rows: list):
    grouped = _group_rows_for_paper(overall_rows)
    best = _best_per_factor(grouped)

    write_csv(summary_root / "paper_table_grouped.csv", grouped)
    write_csv(summary_root / "paper_table_best_per_factor.csv", best)
    write_markdown_table(summary_root / "paper_table_grouped.md", grouped)
    write_markdown_table(summary_root / "paper_table_best_per_factor.md", best)


def build_factor_candidates(factor: dict) -> list:
    """
    Return candidates for one factor.

    Supported schemas:
    1) Single-key sweep (legacy):
       {name, key, values:[...]}
    2) Multi-key grid sweep:
       {name, grid:[{label:..., overrides:{k:v, ...}}, ...]}
    """
    if "grid" in factor:
        cands = []
        for item in factor.get("grid", []):
            label = str(item.get("label", "grid_item"))
            overrides = item.get("overrides", {})
            cands.append({
                "key": "<multi>",
                "value": label,
                "overrides": overrides,
            })
        return cands

    factor_key = str(factor["key"])
    return [{
        "key": factor_key,
        "value": v,
        "overrides": {factor_key: v},
    } for v in factor.get("values", [])]


def main():
    parser = argparse.ArgumentParser(description="Run single-factor ablation sweeps for train_ranker")
    parser.add_argument("--base-config", type=str, default="./config.yaml", help="Base train config path")
    parser.add_argument("--sweep-config", type=str, default="./sweep_single_factor.yaml", help="Sweep config path")
    parser.add_argument("--train-module", type=str, default="src.train_ranker", help="Python module for training")
    parser.add_argument("--output-root", type=str, default="./outputs/ablation", help="Root dir for ablation outputs")
    parser.add_argument("--dry-run", action="store_true", help="Only generate run configs without executing training")
    args = parser.parse_args()

    base_cfg_path = Path(args.base_config)
    sweep_cfg_path = Path(args.sweep_config)
    output_root = Path(args.output_root)

    base_cfg = load_yaml(base_cfg_path)
    sweep_cfg = load_yaml(sweep_cfg_path)

    factors = sweep_cfg.get("factors", [])
    if not factors:
        raise ValueError("No factors found in sweep config. Please define factors list.")

    seeds = sweep_cfg.get("seeds", [int(base_cfg.get("train", {}).get("seed", 42))])
    shared_overrides = sweep_cfg.get("shared_overrides", {})

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = output_root / f"ablation_{stamp}"
    runs_root = exp_root / "runs"
    cfgs_root = exp_root / "configs"
    summary_root = exp_root / "summaries"

    overall_rows = []

    for factor in factors:
        factor_name = str(factor.get("name", factor.get("key", "factor")))
        factor_candidates = build_factor_candidates(factor)

        factor_rows = []
        for cand in factor_candidates:
            factor_key = str(cand["key"])
            value = cand["value"]
            for seed in seeds:
                cfg = copy.deepcopy(base_cfg)

                for k, v in shared_overrides.items():
                    set_by_dotted_key(cfg, str(k), v)

                for kk, vv in cand["overrides"].items():
                    set_by_dotted_key(cfg, str(kk), vv)
                set_by_dotted_key(cfg, "train.seed", int(seed))

                value_slug = slugify(value)
                run_name = f"{factor_name}__{factor_key}={value_slug}__seed={seed}"
                run_dir = runs_root / factor_name / f"value_{value_slug}" / f"seed_{seed}"

                ckpt_dir = run_dir / "checkpoints"
                log_dir = run_dir / "train_logs"
                scaler_path = run_dir / "cache" / "scaler.pkl"

                cfg.setdefault("outputs", {})
                cfg["outputs"]["checkpoint_dir"] = str(ckpt_dir)
                cfg["outputs"]["train_log_dir"] = str(log_dir)

                cfg.setdefault("cache", {})
                cfg["cache"]["scaler_path"] = str(scaler_path)

                run_cfg_path = cfgs_root / f"{run_name}.yaml"
                dump_yaml(run_cfg_path, cfg)

                if args.dry_run:
                    return_code = 0
                else:
                    print(f"[Run] {run_name}")
                    return_code = run_one_experiment(args.train_module, run_cfg_path, run_dir)

                metrics = summarize_metrics(log_dir / "epoch_metrics.csv")

                row = {
                    "factor": factor_name,
                    "key": factor_key,
                    "value": value,
                    "seed": seed,
                    "return_code": return_code,
                    "best_epoch": metrics["best_epoch"],
                    "best_ndcg10": metrics["best_ndcg10"],
                    "best_ndcg20": metrics["best_ndcg20"],
                    "best_pair_acc": metrics["best_pair_acc"],
                    "last_epoch": metrics["last_epoch"],
                    "last_train_loss": metrics["last_train_loss"],
                    "run_dir": str(run_dir),
                    "config_path": str(run_cfg_path),
                }
                factor_rows.append(row)
                overall_rows.append(row)

        write_csv(summary_root / f"summary_{factor_name}.csv", factor_rows)

    write_csv(summary_root / "summary_all.csv", overall_rows)
    write_paper_summaries(summary_root, overall_rows)

    print(f"[Done] Sweep root: {exp_root}")
    print(f"[Done] Summary: {summary_root / 'summary_all.csv'}")
    print(f"[Done] Paper tables: {summary_root / 'paper_table_grouped.csv'}")


if __name__ == "__main__":
    main()
