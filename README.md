# BLMNet

Boundary-aware lightweight Mamba-CNN experiments for skin lesion segmentation.

The main runnable framework is in `sci_skinseg/`. The repository also keeps
baseline model folders beside it so the unified experiment scripts can import
the comparison models:

- `MALUNet/`
- `LB-UNet/`
- `EGE-UNet/`
- `UNeXt/`

Datasets are intentionally not tracked. See `sci_skinseg/KAGGLE_GUIDE_CN.md`
for the Kaggle workflow and `sci_skinseg/EXPERIMENT_PLAN.md` for the experiment
matrix.
