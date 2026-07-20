# SCI Skin Lesion Segmentation Experiments

This repository contains a Kaggle-friendly experimental framework for a SCI-level
paper on boundary-aware lightweight skin lesion segmentation.

## Proposed Direction

Working title:

**Boundary-Aware Lightweight Mamba-CNN Network for Robust Skin Lesion Segmentation under Edge Constraints**

The code supports:

- same-domain training/evaluation on ISIC2017 and ISIC2018
- cross-domain evaluation on ISIC2017, ISIC2018, and PH2
- lightweight baselines (`malunet`, `lbunet`, `unext`, `egeunet`) and the proposed `blmnet`
- ablations for boundary branch and selective-scan block
- Dice, IoU, Accuracy, sensitivity, specificity, precision, F1, HD95
- parameters, approximate FLOPs, model size, and FPS
- CSV outputs ready for tables and paper writing

## Expected Data Layout

Place datasets under `data_root`:

```text
data/
  isic2017/
    train/images
    train/masks
    val/images
    val/masks
  isic2018/
    train/images
    train/masks
    val/images
    val/masks
  PH2Dataset/
    ph2/test/images
    ph2/test/masks
```

## Single Experiment

```bash
python run_experiment.py \
  --data-root ./data \
  --train-dataset isic2018 \
  --val-dataset isic2018 \
  --test-datasets isic2017 PH2 \
  --model blmnet \
  --epochs 200 \
  --batch-size 16 \
  --img-size 256 \
  --output-dir ./outputs/isic2018_blmnet
```

## Full Paper Suite

For a first Kaggle run:

```bash
python run_suite.py --data-root ./data --suite quick --epochs 80
```

For final paper experiments:

```bash
python run_suite.py --data-root ./data --suite full --epochs 300 --seeds 2026 2027 2028
```

After all runs:

```bash
python summarize_results.py --root ./outputs
```

## Kaggle Notes

For SOTA baselines, keep these folders next to `sci_skinseg` in the cloned
project root:

- `MALUNet/MALUNet-main`
- `LB-UNet/LB-UNet-main`
- `EGE-UNet/EGE-UNet`

`unext` is implemented internally as a dependency-light UNeXt-style baseline
because the original UNeXt code depends on `mmcv`.

On Kaggle, clone the project, upload the datasets, then run:

```bash
python run_suite.py \
  --data-root /kaggle/input/YOUR_DATASET_FOLDER/data \
  --output-root /kaggle/working/outputs \
  --suite full \
  --epochs 300 \
  --batch-size 16
```

All results are written to `outputs/summary.csv`, per-run metric CSV files, and
optional prediction masks.
