from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./outputs")
    args = parser.parse_args()
    root = Path(args.root)
    all_results = root / "all_results.csv"
    if all_results.exists():
        df = pd.read_csv(all_results)
    else:
        files = sorted(root.glob("**/summary.csv"))
        if not files:
            raise SystemExit(f"No all_results.csv or summary.csv files found under {root}")
        df = pd.concat([pd.read_csv(f).assign(run=str(f.parent.relative_to(root))) for f in files], ignore_index=True)
        df.to_csv(all_results, index=False)
    metric_cols = ["dice", "iou", "miou", "accuracy", "hd95", "sensitivity", "specificity", "precision", "fps", "params", "flops"]
    group_cols = ["model", "train_dataset", "eval_dataset"]
    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else str(col)
        for col in agg.columns
    ]
    agg.to_csv(root / "summary_table.csv", index=False)
    print(f"Wrote {root / 'all_results.csv'}")
    print(f"Wrote {root / 'summary_table.csv'}")


if __name__ == "__main__":
    main()
