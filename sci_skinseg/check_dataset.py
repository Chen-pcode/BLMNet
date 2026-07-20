from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve_split(data_root: str | Path, dataset: str, split: str) -> tuple[Path, Path]:
    root = Path(data_root)
    key = dataset.lower()
    if key in {"isic2017", "2017"}:
        real_split = "val" if split == "test" else split
        base = root / "isic2017" / real_split
        return base / "images", base / "masks"
    if key in {"isic2018", "2018"}:
        real_split = "val" if split == "test" else split
        base = root / "isic2018" / real_split
        return base / "images", base / "masks"
    if key in {"ph2", "ph2dataset"}:
        candidates = [
            root / "PH2Dataset" / "ph2" / "test",
            root / "ph2dataset" / "ph2" / "test",
            root / "PH2" / "ph2" / "test",
            root / "ph2" / "test",
            root / "ph2dataset" / "test",
            root / "PH2Dataset" / "test",
        ]
        base = next((p for p in candidates if (p / "images").exists() and (p / "masks").exists()), candidates[0])
        return base / "images", base / "masks"
    raise ValueError(f"Unknown dataset: {dataset}")


def stem_candidates(image_path: Path) -> list[str]:
    stem = image_path.stem
    return [stem, f"{stem}_segmentation", stem.replace("_Dermoscopic_Image", "_lesion"), stem.replace("_image", "_mask")]


def pair_images_masks(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    mask_map = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    pairs = []
    for img in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        mask = None
        for cand in stem_candidates(img):
            if cand in mask_map:
                mask = mask_map[cand]
                break
        if mask is None:
            same_name = mask_dir / img.name
            if same_name.exists():
                mask = same_name
        if mask is not None:
            pairs.append((img, mask))
    if not pairs:
        raise RuntimeError(f"No image-mask pairs found in {image_dir} and {mask_dir}")
    return pairs


def inspect_split(data_root: str, dataset: str, split: str) -> dict[str, object]:
    image_dir, mask_dir = resolve_split(data_root, dataset, split)
    pairs = pair_images_masks(image_dir, mask_dir)
    widths = []
    heights = []
    mask_widths = []
    mask_heights = []
    for image_path, mask_path in pairs[: min(200, len(pairs))]:
        with Image.open(image_path) as img:
            widths.append(img.width)
            heights.append(img.height)
        with Image.open(mask_path) as mask:
            mask_widths.append(mask.width)
            mask_heights.append(mask.height)
    return {
        "dataset": dataset,
        "split": split,
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "pairs": len(pairs),
        "image_width_min": min(widths),
        "image_width_max": max(widths),
        "image_height_min": min(heights),
        "image_height_max": max(heights),
        "mask_width_min": min(mask_widths),
        "mask_width_max": max(mask_widths),
        "mask_height_min": min(mask_heights),
        "mask_height_max": max(mask_heights),
        "first_image": pairs[0][0].name,
        "first_mask": pairs[0][1].name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--out", type=str, default="./dataset_report.csv")
    args = parser.parse_args()

    rows = []
    for dataset, splits in {
        "isic2017": ["train", "val"],
        "isic2018": ["train", "val"],
        "PH2": ["test"],
    }.items():
        for split in splits:
            rows.append(inspect_split(args.data_root, dataset, split))
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
