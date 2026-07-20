from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from skinseg.data import SkinLesionDataset
from skinseg.engine import evaluate, save_checkpoint, train_one_epoch
from skinseg.models import get_model
from skinseg.utils import (
    count_params,
    ensure_dir,
    estimate_flops,
    measure_fps,
    model_size_mb,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--train-dataset", type=str, default="isic2018")
    parser.add_argument("--val-dataset", type=str, default="isic2018")
    parser.add_argument("--test-datasets", nargs="*", default=["isic2017", "PH2"])
    parser.add_argument("--model", type=str, default="blmnet")
    parser.add_argument("--output-dir", type=str, default="./outputs/debug")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-preds", action="store_true")
    parser.add_argument("--color-jitter", action="store_true")
    parser.add_argument("--patience", type=int, default=80)
    return parser.parse_args()


def make_loader(ds: SkinLesionDataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = vars(args).copy()
    config["device"] = str(device)
    save_json(config, out_dir / "config.json")

    train_ds = SkinLesionDataset(
        args.data_root,
        args.train_dataset,
        "train",
        args.img_size,
        augment=True,
        color_jitter=args.color_jitter,
    )
    val_ds = SkinLesionDataset(args.data_root, args.val_dataset, "val", args.img_size, augment=False)
    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, True)
    val_loader = make_loader(val_ds, args.batch_size, args.num_workers, False)

    model = get_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    params = count_params(model)
    flops = estimate_flops(model, args.img_size, device)
    size_mb = model_size_mb(model)
    best_dice = -1.0
    best_epoch = -1
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args.amp)
        scheduler.step()
        val_metrics, val_df = evaluate(model, val_loader, device, threshold=args.threshold)
        row = {"epoch": epoch, "loss": loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        val_df.to_csv(out_dir / "val_samples_latest.csv", index=False)
        dice = val_metrics["dice"]
        print(f"epoch={epoch:03d} loss={loss:.4f} val_dice={dice:.4f} val_iou={val_metrics['iou']:.4f} hd95={val_metrics['hd95']:.3f}")
        if dice > best_dice:
            best_dice = dice
            best_epoch = epoch
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, best_dice, config)
            val_df.to_csv(out_dir / "val_samples_best.csv", index=False)
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}.")
            break

    if (out_dir / "best.pt").exists():
        ckpt = torch.load(out_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])

    fps = measure_fps(model, args.img_size, device)
    final_rows = []
    eval_specs = [(args.val_dataset, "val")]
    for name in args.test_datasets:
        split = "test" if name.lower() in {"ph2", "ph2dataset"} else "val"
        eval_specs.append((name, split))

    for dataset_name, split in eval_specs:
        ds = SkinLesionDataset(args.data_root, dataset_name, split, args.img_size, augment=False)
        loader = make_loader(ds, args.batch_size, args.num_workers, False)
        pred_dir = out_dir / "predictions" / f"{dataset_name}_{split}" if args.save_preds else None
        metrics, sample_df = evaluate(model, loader, device, threshold=args.threshold, save_dir=pred_dir)
        sample_df.to_csv(out_dir / f"samples_{dataset_name}_{split}.csv", index=False)
        row = {
            "model": args.model,
            "train_dataset": args.train_dataset,
            "eval_dataset": dataset_name,
            "eval_split": split,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "params": params,
            "flops": flops,
            "size_mb": size_mb,
            "fps": fps,
            "runtime_min": (time.time() - start) / 60.0,
            **metrics,
        }
        final_rows.append(row)

    summary = pd.DataFrame(final_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    save_json({"best_epoch": best_epoch, "best_dice": best_dice, "rows": final_rows}, out_dir / "summary.json")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
