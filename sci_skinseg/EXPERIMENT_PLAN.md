# Paper Framework and Experiment Plan

## Proposed Paper Direction

Title candidate:

**Boundary-Aware Lightweight Mamba-CNN Network for Robust Skin Lesion Segmentation under Edge Constraints**

## Main Hypothesis

Lightweight skin lesion segmentation models usually lose boundary fidelity and
cross-domain robustness after compression. A compact CNN encoder, a selective
scan global context block, and a boundary-guided decoder can improve the
accuracy-efficiency Pareto trade-off.

## Contributions

1. A lightweight Mamba-style CNN hybrid model named `BLMNet`.
2. A boundary-guided multi-scale decoder with auxiliary boundary supervision.
3. A unified evaluation protocol covering Dice, IoU, HD95, FPS, FLOPs, and Params.
4. Cross-domain validation on ISIC2017, ISIC2018, and PH2.

## Datasets

- ISIC2018: primary training benchmark.
- ISIC2017: secondary benchmark and cross-domain target.
- PH2: external cross-domain test set.

## Metrics

- Region overlap: Dice, foreground IoU, mIoU.
- Pixel classification: Accuracy, sensitivity, specificity, precision.
- Boundary quality: HD95.
- Efficiency: Params, FLOPs, model size, FPS.

## Experiment Tables

Table 1: Same-domain comparison on ISIC2018.

Models:

- MALUNet
- LB-UNet
- UNeXt
- EGE-UNet
- BLMNet

Table columns:

- Dice, IoU, mIoU, Accuracy, HD95, sensitivity, specificity, Params, FLOPs, FPS

Table 2: Same-domain comparison on ISIC2017.

Same models and metrics as Table 1.

Table 3: Cross-domain generalization.

Training/testing:

- ISIC2017 -> ISIC2017
- ISIC2018 -> ISIC2018
- ISIC2018 -> ISIC2017
- ISIC2018 -> PH2
- ISIC2017 -> ISIC2018
- ISIC2017 -> PH2

Table 4: Ablation study.

Models:

- BLMNet
- BLMNet without boundary-guided decoder
- BLMNet without selective scan block

Table 5: Efficiency-accuracy Pareto analysis.

Plot:

- x-axis: Params or FLOPs
- y-axis: Dice or HD95
- point size/color: FPS

## Recommended Running Order

1. Smoke test:

```bash
python run_suite.py --data-root ./data --suite quick --epochs 2 --batch-size 2 --img-size 128
```

2. Quick comparison:

```bash
python run_suite.py --data-root ./data --suite quick --epochs 80 --batch-size 16 --img-size 256 --amp
```

3. Full comparison:

```bash
python run_suite.py --data-root ./data --suite full --epochs 300 --batch-size 16 --img-size 256 --seeds 2026 2027 2028 --amp
```

4. Ablation:

```bash
python run_suite.py --data-root ./data --suite ablation --epochs 300 --batch-size 16 --img-size 256 --seeds 2026 2027 2028 --amp
```

## Files to Send Back for Paper Writing

After experiments finish, send:

- `outputs/all_results.csv`
- `outputs/summary_table.csv`
- representative prediction images from `outputs/*/predictions`
- `outputs/*/history.csv` for training curves
