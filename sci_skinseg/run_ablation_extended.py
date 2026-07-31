from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ABLATION_MODELS = [
    "blmnet",
    "blmnet_no_scan",
    "blmnet_scan2",
    "blmnet_no_boundary",
    "blmnet_no_boundary_loss",
    "blmnet_no_boundary_feedback",
    "blmnet_no_group_enhance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run extended BLMNet ablations for paper-grade component analysis."
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-root", type=str, default="./outputs_ablation_extended")
    parser.add_argument("--train-dataset", type=str, default="isic2018")
    parser.add_argument("--val-dataset", type=str, default="isic2018")
    parser.add_argument("--test-datasets", nargs="*", default=["isic2017", "PH2"])
    parser.add_argument("--models", nargs="*", default=ABLATION_MODELS)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seeds", nargs="*", type=int, default=[2026])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-preds", action="store_true")
    parser.add_argument("--no-color-jitter", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for model in args.models:
            run_name = f"ablation_ext_{args.train_dataset}_{model}_seed{seed}"
            cmd = [
                sys.executable,
                str(script_dir / "run_experiment.py"),
                "--data-root",
                args.data_root,
                "--train-dataset",
                args.train_dataset,
                "--val-dataset",
                args.val_dataset,
                "--test-datasets",
                *args.test_datasets,
                "--model",
                model,
                "--output-dir",
                str(output_root / run_name),
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
            ]
            if not args.no_color_jitter:
                cmd.append("--color-jitter")
            if args.amp:
                cmd.append("--amp")
            if args.save_preds:
                cmd.append("--save-preds")
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)

    subprocess.run(
        [
            sys.executable,
            str(script_dir / "summarize_results.py"),
            "--root",
            str(output_root),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
