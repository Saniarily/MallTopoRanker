import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot in-batch experiment metrics from epoch csv")
    parser.add_argument("--metrics-csv", type=str, required=True, help="Path to epoch_metrics.csv")
    parser.add_argument("--out-dir", type=str, default="./outputs/figures", help="Output figure directory")
    args = parser.parse_args()

    metrics_csv = Path(args.metrics_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics csv not found: {metrics_csv}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("matplotlib is required. Install with: pip install matplotlib") from e

    df = pd.read_csv(metrics_csv)
    if len(df) == 0:
        raise ValueError("metrics csv is empty")

    # Figure 1: train loss
    plt.figure(figsize=(8, 4.2))
    plt.plot(df["epoch"], df["train_loss"], color="#374151", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("In-Batch Pairwise Training Loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "inbatch_train_loss_curve.png", dpi=220)
    plt.close()

    # Figure 2: val/test NDCG
    has_val_ndcg = "val_ndcg@10" in df.columns and "val_ndcg@20" in df.columns
    has_test_ndcg = "test_ndcg@10" in df.columns and "test_ndcg@20" in df.columns
    if has_val_ndcg or has_test_ndcg:
        plt.figure(figsize=(8.4, 4.6))
        if has_val_ndcg:
            plt.plot(df["epoch"], df["val_ndcg@10"], label="val_ndcg@10", color="#111827", linewidth=2)
            plt.plot(df["epoch"], df["val_ndcg@20"], label="val_ndcg@20", color="#4b5563", linewidth=2, linestyle="--")
        if has_test_ndcg:
            plt.plot(df["epoch"], df["test_ndcg@10"], label="test_ndcg@10", color="#1d4ed8", linewidth=2)
            plt.plot(df["epoch"], df["test_ndcg@20"], label="test_ndcg@20", color="#60a5fa", linewidth=2, linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("NDCG")
        plt.title("Validation/Test NDCG Across Epochs")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "inbatch_val_test_ndcg_curve.png", dpi=220)
        plt.close()

    # Figure 3: val/test pairwise accuracy
    has_val_acc = "val_pairwise_acc" in df.columns
    has_test_acc = "test_pairwise_acc" in df.columns
    if has_val_acc or has_test_acc:
        plt.figure(figsize=(8.4, 4.4))
        if has_val_acc:
            plt.plot(df["epoch"], df["val_pairwise_acc"], label="val_pairwise_acc", color="#111827", linewidth=2)
        if has_test_acc:
            plt.plot(df["epoch"], df["test_pairwise_acc"], label="test_pairwise_acc", color="#1d4ed8", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Pairwise Accuracy")
        plt.title("Validation/Test Pairwise Accuracy")
        plt.ylim(0.0, 1.0)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "inbatch_val_test_pair_acc_curve.png", dpi=220)
        plt.close()

    print(f"[Done] Figures saved under: {out_dir}")


if __name__ == "__main__":
    main()
