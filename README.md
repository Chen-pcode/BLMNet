# BLMNet

Boundary-aware lightweight Mamba-CNN experiments for skin lesion segmentation.

The main runnable framework is in `sci_skinseg/`. The repository also keeps
baseline model folders beside it so the unified experiment scripts can import
the comparison models:

- `MALUNet/`
- `LB-UNet/`
- `EGE-UNet/`
- `UNeXt/`

Additional Kaggle-safe comparison entries are implemented directly in
`sci_skinseg/skinseg/models.py`, so the original heavy repositories do not need
to be committed:

- `unet`
- `mobilevitv2`
- `mambahome`
- `litemamba_bound`

Datasets are intentionally not tracked. See `sci_skinseg/KAGGLE_GUIDE_CN.md`
for the Kaggle workflow and `sci_skinseg/EXPERIMENT_PLAN.md` for the experiment
matrix.

Run the extra comparison suite on Kaggle:

```bash
cd /kaggle/working/BLMNet/sci_skinseg
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
