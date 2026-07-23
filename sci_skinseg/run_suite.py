from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-root", type=str, default="./outputs")
    parser.add_argument("--suite", choices=["quick", "full", "ablation", "cross", "extra"], default="quick")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seeds", nargs="*", type=int, default=[2026])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-preds", action="store_true")
    return parser.parse_args()


def experiments(suite: str) -> list[tuple[str, str, list[str], str]]:
    baselines = ["malunet", "lbunet", "unext", "egeunet"]
    extra_baselines = ["unet", "mobilevitv2", "mambahome", "litemamba_bound"]
    proposed = ["blmnet"]
    ablations = ["blmnet_no_boundary", "blmnet_no_scan"]
    if suite == "quick":
        return [("isic2018", "isic2018", ["isic2017", "PH2"], m) for m in ["malunet", "lbunet", "unext", "egeunet", "blmnet"]]
    if suite == "full":
        rows = []
        for train in ["isic2018", "isic2017"]:
            tests = ["isic2017", "PH2"] if train == "isic2018" else ["isic2018", "PH2"]
            for model in baselines + proposed:
                rows.append((train, train, tests, model))
        return rows
    if suite == "ablation":
        return [("isic2018", "isic2018", ["isic2017", "PH2"], m) for m in proposed + ablations]
    if suite == "cross":
        return [
            ("isic2018", "isic2018", ["isic2017", "PH2"], "blmnet"),
            ("isic2017", "isic2017", ["isic2018", "PH2"], "blmnet"),
        ]
    if suite == "extra":
        rows = []
        for train in ["isic2018", "isic2017"]:
            tests = ["isic2017", "PH2"] if train == "isic2018" else ["isic2018", "PH2"]
            for model in extra_baselines:
                rows.append((train, train, tests, model))
        return rows
    raise ValueError(suite)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for train, val, tests, model in experiments(args.suite):
            run_name = f"{args.suite}_{train}_{model}_seed{seed}"
            cmd = [
                sys.executable,
                str(script_dir / "run_experiment.py"),
                "--data-root",
                args.data_root,
                "--train-dataset",
                train,
                "--val-dataset",
                val,
                "--test-datasets",
                *tests,
                "--model",
                model,
                "--output-dir",
                str(root / run_name),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--img-size",
                str(args.img_size),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(seed),
                "--color-jitter",
            ]
            if args.amp:
                cmd.append("--amp")
            if args.save_preds:
                cmd.append("--save-preds")
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, str(script_dir / "summarize_results.py"), "--root", str(root)], check=True)


if __name__ == "__main__":
    main()
