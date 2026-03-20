# src/plot_paper_figures.py
from pathlib import Path
import glob

import pandas as pd
import numpy as np


def main():
    out_fig = Path("./outputs/figures")
    out_fig.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise RuntimeError("Please install matplotlib & seaborn: mamba install -c conda-forge matplotlib seaborn -y")

    # =========================
    # 0) Training/Validation curves (NEW)
    # =========================
    trainlog_path = Path("./outputs/train_logs/epoch_metrics.csv")
    if trainlog_path.exists():
        tl = pd.read_csv(trainlog_path)

        # train loss curve
        plt.figure(figsize=(7.5, 4))
        plt.plot(tl["epoch"], tl["train_loss"], color="#4a5568", linewidth=2, label="train_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_fig / "train_loss_curve.png", dpi=200)
        plt.close()

        # val pairwise acc curve
        if "val_pairwise_acc" in tl.columns:
            plt.figure(figsize=(7.5, 4))
            plt.plot(tl["epoch"], tl["val_pairwise_acc"], color="#2d3748", linewidth=2, label="val_pairwise_acc")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Validation Pairwise Accuracy Curve")
            plt.ylim(0.0, 1.0)
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(out_fig / "val_pairwise_acc_curve.png", dpi=200)
            plt.close()

        # val ndcg curve (10 & 20)
        if "val_ndcg@10" in tl.columns and "val_ndcg@20" in tl.columns:
            plt.figure(figsize=(7.5, 4))
            plt.plot(tl["epoch"], tl["val_ndcg@10"], color="#4a5568", linewidth=2, label="val_ndcg@10")
            plt.plot(tl["epoch"], tl["val_ndcg@20"], color="#718096", linewidth=2, linestyle="--", label="val_ndcg@20")
            plt.xlabel("Epoch")
            plt.ylabel("NDCG")
            plt.title("Validation NDCG@K Curve")
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_fig / "val_ndcg_curve.png", dpi=200)
            plt.close()

    # =========================
    # 1) Overall evaluation metrics bar (from outputs/eval) — keep existing
    # =========================
    per_q_path = Path("./outputs/eval/metrics_per_query.csv")
    if per_q_path.exists():
        df = pd.read_csv(per_q_path)

        cols = [c for c in df.columns if c.startswith("ndcg_cont@") or c.startswith("precision_bin@") or c.startswith("recall_bin@") or c == "spearman"]
        if cols:
            mean_vals = df[cols].mean().sort_values(ascending=False)

            plt.figure(figsize=(10, 4))
            sns.barplot(x=mean_vals.index, y=mean_vals.values, color="#4a5568")
            plt.xticks(rotation=45, ha="right")
            plt.title("Overall Evaluation Metrics (mean over queries)")
            plt.tight_layout()
            plt.savefig(out_fig / "overall_metrics_bar.png", dpi=200)
            plt.close()

    # =========================
    # 2) Bucket heatmap (NDCG@10)
    # =========================
    by_bucket_path = Path("./outputs/eval/metrics_by_bucket.csv")
    if by_bucket_path.exists():
        bb = pd.read_csv(by_bucket_path)
        if "ndcg_cont@10" in bb.columns and "bucket_id" in bb.columns:
            bb["city_cluster"] = bb["bucket_id"].astype(str).apply(lambda s: s.split("_")[0])
            bb["area_bin"] = bb["bucket_id"].astype(str).apply(lambda s: s.split("_")[1])

            pivot = bb.pivot_table(index="city_cluster", columns="area_bin", values="ndcg_cont@10", aggfunc="mean")
            plt.figure(figsize=(6.5, 4))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Greys", cbar=True)
            plt.title("NDCG@10 by Bucket (city_cluster × area_bin)")
            plt.xlabel("area_bin (0:<200k,1:200-450k,2:>=450k)")
            plt.ylabel("city_cluster")
            plt.tight_layout()
            plt.savefig(out_fig / "bucket_ndcg10_heatmap.png", dpi=200)
            plt.close()

    # =========================
    # 3) Attention heatmap (condition -> metric)
    # =========================
    attn_path = Path("./outputs/eval/attention_condition_metric_mean.csv")
    if attn_path.exists():
        attn = pd.read_csv(attn_path, index_col=0)
        plt.figure(figsize=(7.5, 4))
        sns.heatmap(attn, annot=True, fmt=".3f", cmap="Greys", cbar=True)
        plt.title("Attention Heatmap: Condition → Metric (mean)")
        plt.tight_layout()
        plt.savefig(out_fig / "attention_condition_metric_heatmap.png", dpi=200)
        plt.close()

    # =========================
    # 4) Cases (optional)
    # =========================
    case_files = glob.glob("./outputs/cases/*.json")
    if case_files:
        rows = []
        for fp in case_files:
            import json
            obj = json.load(open(fp, "r", encoding="utf-8"))
            name = Path(fp).stem
            if obj.get("topk"):
                top1 = obj["topk"][0]
                expl = top1.get("explanations", {})
                for tm in expl.get("top_metrics", []):
                    rows.append({"case": name, "metric": tm["name"], "contribution": tm["contribution"]})

        if rows:
            d = pd.DataFrame(rows)
            d = d.sort_values(["case", "contribution"], ascending=[True, False]).groupby("case").head(1)

            plt.figure(figsize=(10, 4))
            sns.barplot(data=d, x="case", y="contribution", hue="metric")
            plt.xticks(rotation=45, ha="right")
            plt.title("Case Study: Top-1 Metric Contribution (Top-1 item)")
            plt.tight_layout()
            plt.savefig(out_fig / "case_top1_metric_contribution.png", dpi=200)
            plt.close()

    print("Saved figures to outputs/figures/ (png)")


if __name__ == "__main__":
    main()
