from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    split: str
    image_dir: Path
    mask_dir: Path


def resolve_split(data_root: str | Path, dataset: str, split: str) -> DatasetSplit:
    root = Path(data_root)
    key = dataset.lower()
    if key in {"isic2017", "2017"}:
        base = root / "isic2017"
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split for isic2017: {split}")
        real_split = "val" if split == "test" else split
        return DatasetSplit("isic2017", real_split, base / real_split / "images", base / real_split / "masks")
    if key in {"isic2018", "2018"}:
        base = root / "isic2018"
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split for isic2018: {split}")
        real_split = "val" if split == "test" else split
        return DatasetSplit("isic2018", real_split, base / real_split / "images", base / real_split / "masks")
    if key in {"ph2", "ph2dataset", "ph2dataset"}:
        base = root / "PH2Dataset" / "ph2" / "test"
        return DatasetSplit("PH2", "test", base / "images", base / "masks")
    raise ValueError(f"Unknown dataset: {dataset}")


def _stem_candidates(image_path: Path) -> list[str]:
    stem = image_path.stem
    return [
        stem,
        f"{stem}_segmentation",
        stem.replace("_Dermoscopic_Image", "_lesion"),
        stem.replace("_image", "_mask"),
    ]


def pair_images_masks(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")
    mask_map = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        mask = None
        for cand in _stem_candidates(img):
            if cand in mask_map:
                mask = mask_map[cand]
                break
        if mask is None:
            # Last resort: numeric datasets often use identical sorted names.
            same_name = mask_dir / img.name
            if same_name.exists():
                mask = same_name
        if mask is not None:
            pairs.append((img, mask))
    if not pairs:
        raise RuntimeError(f"No image-mask pairs found in {image_dir} and {mask_dir}")
    return pairs


class SkinLesionDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        dataset: str,
        split: str,
        img_size: int = 256,
        augment: bool = False,
        color_jitter: bool = False,
    ) -> None:
        self.info = resolve_split(data_root, dataset, split)
        self.pairs = pair_images_masks(self.info.image_dir, self.info.mask_dir)
        self.img_size = img_size
        self.augment = augment
        self.color_jitter = color_jitter

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        img_path, mask_path = self.pairs[index]
        image = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        if self.augment:
            image_np, mask_np = self._augment(image_np, mask_np)

        image_np = (image_np - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        image_t = torch.from_numpy(image_np.transpose(2, 0, 1).copy()).float()
        mask_t = torch.from_numpy(mask_np[None].copy()).float()
        boundary_t = mask_to_boundary(mask_t)
        return {"image": image_t, "mask": mask_t, "boundary": boundary_t, "id": img_path.stem}

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        if random.random() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
        if self.color_jitter:
            if random.random() < 0.8:
                gain = np.random.uniform(0.85, 1.15, size=(1, 1, 3)).astype(np.float32)
                bias = np.random.uniform(-0.06, 0.06, size=(1, 1, 3)).astype(np.float32)
                image = np.clip(image * gain + bias, 0.0, 1.0)
            if random.random() < 0.3:
                gamma = random.uniform(0.8, 1.25)
                image = np.clip(image**gamma, 0.0, 1.0)
        return image.copy(), mask.copy()


def mask_to_boundary(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    pad = k // 2
    dilated = torch.nn.functional.max_pool2d(mask, k, stride=1, padding=pad)
    eroded = 1.0 - torch.nn.functional.max_pool2d(1.0 - mask, k, stride=1, padding=pad)
    boundary = (dilated - eroded).clamp(0, 1)
    return boundary.squeeze(0)


def dataset_report(data_root: str | Path, names: Iterable[str] = ("isic2017", "isic2018", "PH2")) -> dict[str, int]:
    report: dict[str, int] = {}
    for name in names:
        splits = ["train", "val"] if name.lower() != "ph2" else ["test"]
        for split in splits:
            ds = SkinLesionDataset(data_root, name, split, img_size=64)
            report[f"{name}_{split}"] = len(ds)
    return report
