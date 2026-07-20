from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import multitask_loss
from .metrics import binary_metrics, logits_to_mask, mean_dict
from .models import unwrap_logits
from .utils import AverageMeter, autocast_context, ensure_dir


def train_one_epoch(model, loader: DataLoader, optimizer, scaler, device, amp: bool) -> float:
    model.train()
    meter = AverageMeter()
    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        boundary = batch["boundary"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            outputs = model(image)
            loss = multitask_loss(outputs, mask, boundary)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        meter.update(float(loss.detach().cpu()), image.size(0))
    return meter.avg


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device,
    threshold: float = 0.5,
    save_dir: str | Path | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows: list[dict[str, float | str]] = []
    pred_dir = ensure_dir(save_dir) if save_dir is not None else None
    for batch in tqdm(loader, desc="eval", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].cpu().numpy()
        ids = batch["id"]
        logits = unwrap_logits(model(image))
        pred = logits_to_mask(logits, threshold=threshold).cpu().numpy()
        for i, sample_id in enumerate(ids):
            p = pred[i, 0].astype(np.uint8)
            t = target[i, 0].astype(np.uint8)
            metrics = binary_metrics(p, t)
            row = {"id": sample_id, **metrics}
            rows.append(row)
            if pred_dir is not None:
                Image.fromarray((p * 255).astype(np.uint8)).save(pred_dir / f"{sample_id}.png")
    df = pd.DataFrame(rows)
    metric_rows = [{k: float(v) for k, v in r.items() if k != "id"} for r in rows]
    return mean_dict(metric_rows), df


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, best_metric: float, config: dict) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config,
        },
        path,
    )
