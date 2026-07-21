from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = ["dice", "iou", "miou", "accuracy", "hd95", "sensitivity", "specificity", "precision", "fps", "params", "flops"]


def _fmt_mean_std(mean: float, std: float, scale: float = 1.0, digits: int = 2) -> str:
    if pd.isna(std):
        std = 0.0
    return f"{mean * scale:.{digits}f} +/- {std * scale:.{digits}f}"


def make_tables(df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    grouped = df.groupby(["model", "train_dataset", "eval_dataset"])[METRICS].agg(["mean", "std"])
    grouped.to_csv(out / "table_all_metrics_multiindex.csv")

    flat_rows = []
    for keys, sub in df.groupby(["model", "train_dataset", "eval_dataset"]):
        row = {"model": keys[0], "train_dataset": keys[1], "eval_dataset": keys[2]}
        for metric in METRICS:
            row[metric] = _fmt_mean_std(
                sub[metric].mean(),
                sub[metric].std(),
                scale=100.0 if metric in {"dice", "iou", "miou", "accuracy", "sensitivity", "specificity", "precision"} else 1.0,
            )
        row["params_m"] = _fmt_mean_std(sub["params"].mean() / 1e6, sub["params"].std() / 1e6 if len(sub) > 1 else 0.0)
        row["gflops"] = _fmt_mean_std(sub["flops"].mean() / 1e9, sub["flops"].std() / 1e9 if len(sub) > 1 else 0.0)
        flat_rows.append(row)
    flat = pd.DataFrame(flat_rows)
    flat.to_csv(out / "paper_table_formatted.csv", index=False)
    with open(out / "paper_table_formatted.tex", "w", encoding="utf-8") as f:
        f.write(flat.to_latex(index=False, escape=False))

    same_domain = df[df["train_dataset"] == df["eval_dataset"]]
    if not same_domain.empty:
        same_domain.to_csv(out / "same_domain_raw.csv", index=False)
    cross = df[df["train_dataset"] != df["eval_dataset"]]
    if not cross.empty:
        cross.to_csv(out / "cross_domain_raw.csv", index=False)


def make_pareto(df: pd.DataFrame, out: Path) -> None:
    summary = df.groupby(["model", "train_dataset", "eval_dataset"]).agg(
        dice=("dice", "mean"),
        hd95=("hd95", "mean"),
        params=("params", "mean"),
        flops=("flops", "mean"),
        fps=("fps", "mean"),
    ).reset_index()

    for train_dataset in sorted(summary["train_dataset"].unique()):
        sub = summary[summary["train_dataset"] == train_dataset]
        plt.figure(figsize=(7, 5))
        for _, row in sub.iterrows():
            x = row["params"] / 1e6
            y = row["dice"] * 100
            plt.scatter(x, y, s=max(30, min(250, row["fps"] * 2)), alpha=0.75)
            plt.text(x, y, f"{row['model']}->{row['eval_dataset']}", fontsize=8)
        plt.xscale("log")
        plt.xlabel("Parameters (M, log scale)")
        plt.ylabel("Dice (%)")
        plt.title(f"Accuracy-Efficiency Pareto: train on {train_dataset}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / f"pareto_params_dice_{train_dataset}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(7, 5))
        for _, row in sub.iterrows():
            x = row["flops"] / 1e9
            y = row["hd95"]
            plt.scatter(x, y, s=max(30, min(250, row["fps"] * 2)), alpha=0.75)
            plt.text(x, y, f"{row['model']}->{row['eval_dataset']}", fontsize=8)
        plt.xscale("log")
        plt.xlabel("FLOPs (G, log scale)")
        plt.ylabel("HD95 (lower is better)")
        plt.title(f"Boundary-Efficiency Pareto: train on {train_dataset}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / f"pareto_flops_hd95_{train_dataset}.png", dpi=300)
        plt.close()


def make_training_curves(root: Path, out: Path) -> None:
    history_files = sorted(root.glob("**/history.csv"))
    if not history_files:
        return
    plt.figure(figsize=(8, 5))
    for hist in history_files:
        df = pd.read_csv(hist)
        if "val_dice" not in df:
            continue
        label = hist.parent.name
        plt.plot(df["epoch"], df["val_dice"] * 100, label=label, linewidth=1.2)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Dice (%)")
    plt.title("Training Curves")
    plt.grid(True, alpha=0.3)
    if len(history_files) <= 12:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "training_curves_val_dice.png", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./outputs")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "paper_assets"
    out.mkdir(parents=True, exist_ok=True)

    all_results = root / "all_results.csv"
    if not all_results.exists():
        summary_files = sorted(root.glob("**/summary.csv"))
        if not summary_files:
            raise SystemExit(f"No all_results.csv or summary.csv found under {root}")
        df = pd.concat([pd.read_csv(f).assign(run=str(f.parent.relative_to(root))) for f in summary_files], ignore_index=True)
        df.to_csv(all_results, index=False)
    else:
        df = pd.read_csv(all_results)

    make_tables(df, out)
    make_pareto(df, out)
    make_training_curves(root, out)
    print(f"Wrote paper assets to {out}")


if __name__ == "__main__":
    main()
