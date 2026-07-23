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
- extra comparison baselines (`unet`, `mobilevitv2`, `mambahome`, `litemamba_bound`)
- ablations for boundary branch and selective-scan block
- Dice, foreground IoU, mIoU, Accuracy, sensitivity, specificity, precision, F1, HD95
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

The extra baselines are also implemented inside `sci_skinseg/skinseg/models.py`
so they can run on Kaggle without copying large original repositories or custom
CUDA extensions:

- `unet`: standard U-Net reference baseline
- `mobilevitv2`: MobileViTv2-style lightweight CNN/attention segmentation model
- `mambahome`: 2D Mamba Goes HoME-style mixture-of-experts context baseline
- `litemamba_bound`: LiteMamba-Bound-style boundary-aware lightweight baseline

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

## Extra Comparison Suite

Run all added comparison models on both ISIC2018-trained and ISIC2017-trained
protocols:

```bash
python run_suite.py \
  --data-root /kaggle/input/datasets/zichengdoctor \
  --output-root /kaggle/working/outputs_extra \
  --suite extra \
  --epochs 300 \
  --batch-size 16 \
  --img-size 256 \
  --seeds 2026 \
  --amp \
  --save-preds
```

If Kaggle time is not enough, run one model at a time:

```bash
python run_experiment.py \
  --data-root /kaggle/input/datasets/zichengdoctor \
  --train-dataset isic2018 \
  --val-dataset isic2018 \
  --test-datasets isic2017 PH2 \
  --model mobilevitv2 \
  --output-dir /kaggle/working/outputs_extra/mobilevitv2_isic2018 \
  --epochs 300 \
  --batch-size 16 \
  --img-size 256 \
  --seed 2026 \
  --amp \
  --color-jitter \
  --save-preds
```

Replace `mobilevitv2` with `unet`, `mambahome`, or `litemamba_bound` for the
other extra comparisons. Zip the results before downloading:

```bash
python summarize_results.py --root /kaggle/working/outputs_extra
zip -r /kaggle/working/outputs_extra.zip /kaggle/working/outputs_extra
```
