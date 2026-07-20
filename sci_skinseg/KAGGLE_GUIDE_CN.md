# Kaggle 运行说明

## 1. 上传代码

建议把当前项目根目录作为 GitHub 仓库上传，而不是只上传 `sci_skinseg`。因为主实验会调用这些对比模型源码：

- `MALUNet/MALUNet-main`
- `LB-UNet/LB-UNet-main`
- `EGE-UNet/EGE-UNet`

`UNeXt` 原始代码依赖 `mmcv`，为了 Kaggle 稳定运行，当前工程内置了一个 UNeXt-style 轻量 Shift-MLP 版本，运行名仍为 `unext`。

Kaggle Notebook 中执行：

```bash
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名
pip install -r requirements.txt
```

如果你直接上传 zip，也可以在 Kaggle 中解压后进入 `sci_skinseg` 目录。

## 2. 上传数据

数据在 Kaggle Input 中建议保持以下结构：

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

假设 Kaggle 路径是：

```text
/kaggle/input/skin-lesion-data/data
```

先检查数据：

```bash
python check_dataset.py \
  --data-root /kaggle/input/skin-lesion-data/data \
  --out /kaggle/working/dataset_report.csv
```

## 3. 冒烟测试

先用 1-2 个 epoch 确认代码能跑通：

```bash
python run_suite.py \
  --data-root /kaggle/input/skin-lesion-data/data \
  --output-root /kaggle/working/outputs_smoke \
  --suite quick \
  --epochs 2 \
  --batch-size 4 \
  --img-size 128 \
  --num-workers 2 \
  --amp
```

## 4. 正式实验

### 快速对比

```bash
python run_suite.py \
  --data-root /kaggle/input/skin-lesion-data/data \
  --output-root /kaggle/working/outputs \
  --suite quick \
  --epochs 80 \
  --batch-size 16 \
  --img-size 256 \
  --num-workers 2 \
  --amp
```

### 完整主实验

```bash
python run_suite.py \
  --data-root /kaggle/input/skin-lesion-data/data \
  --output-root /kaggle/working/outputs \
  --suite full \
  --epochs 300 \
  --batch-size 16 \
  --img-size 256 \
  --num-workers 2 \
  --seeds 2026 2027 2028 \
  --amp
```

### 消融实验

```bash
python run_suite.py \
  --data-root /kaggle/input/skin-lesion-data/data \
  --output-root /kaggle/working/outputs_ablation \
  --suite ablation \
  --epochs 300 \
  --batch-size 16 \
  --img-size 256 \
  --num-workers 2 \
  --seeds 2026 2027 2028 \
  --amp
```

## 5. 生成论文表格和图片

```bash
python make_paper_assets.py --root /kaggle/working/outputs
python make_paper_assets.py --root /kaggle/working/outputs_ablation
```

输出重点文件：

- `/kaggle/working/outputs/all_results.csv`
- `/kaggle/working/outputs/summary_table.csv`
- `/kaggle/working/outputs/paper_assets/paper_table_formatted.csv`
- `/kaggle/working/outputs/paper_assets/paper_table_formatted.tex`
- `/kaggle/working/outputs/paper_assets/*.png`

## 6. 发给我用于写论文的文件

请打包以下内容：

- `dataset_report.csv`
- `outputs/all_results.csv`
- `outputs/summary_table.csv`
- `outputs/paper_assets/`
- `outputs_ablation/all_results.csv`
- `outputs_ablation/summary_table.csv`
- 若开启 `--save-preds`，再发一部分预测图。
