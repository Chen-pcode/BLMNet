# BLMNet 项目使用与结果复现说明

本文档说明当前仓库的目标、目录结构、代码运行方式、Kaggle 复现实验流程，以及如何生成论文表格和图。

## 1. 项目目标

本项目目标是完成一篇面向皮肤病灶分割的 SCI 论文实验。核心方法为 `BLMNet`，即 Boundary-aware Lightweight Mamba-CNN Network。

BLMNet 主要解决三个问题：

1. 轻量皮肤病灶分割模型容易损失边界细节；
2. 现有方法在跨数据集泛化时性能不稳定；
3. 许多工作只报告 Dice，而缺少 mIoU、HD95、FPS、FLOPs、Params 等完整评价。

当前代码可以完成：

- ISIC2017、ISIC2018、PH2 数据集实验；
- 同域测试和跨域测试；
- BLMNet 与 MALUNet、LB-UNet、EGE-UNet、UNeXt 的统一对比；
- BLMNet 消融实验；
- Dice、IoU、mIoU、Accuracy、Sensitivity、Specificity、Precision、HD95、Params、FLOPs、FPS 指标计算；
- 论文表格和图自动生成。

## 2. 仓库目录结构

仓库根目录建议保持如下结构：

```text
BLMNet/
  README.md
  MALUNet/
  LB-UNet/
  EGE-UNet/
  UNeXt/
  sci_skinseg/
    run_experiment.py
    run_suite.py
    summarize_results.py
    make_paper_assets.py
    check_dataset.py
    requirements.txt
    skinseg/
      data.py
      models.py
      losses.py
      metrics.py
      engine.py
      utils.py
    result_exper/
    paper_outputs/
```

说明：

- `sci_skinseg/` 是主实验工程；
- `MALUNet/`、`LB-UNet/`、`EGE-UNet/` 是对比模型源码；
- `UNeXt/` 保留原始论文代码，但实际实验中使用 `sci_skinseg/skinseg/models.py` 内置的 UNeXt-style 版本，避免 Kaggle 安装 `mmcv` 的问题；
- `result_exper/` 是本地保存的 Kaggle 返回结果；
- `paper_outputs/` 是根据实验结果生成的论文表格和图片。

## 3. 数据集目录要求

Kaggle 中的数据路径为：

```text
/kaggle/input/datasets/zichengdoctor/
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
  ph2dataset/
    ph2/test/images
    ph2/test/masks
```

代码已兼容以下 PH2 命名：

- `PH2Dataset/ph2/test`
- `ph2dataset/ph2/test`
- `PH2/test`
- `ph2/test`
- `ph2dataset/test`

## 4. Kaggle 环境初始化

进入 Kaggle Notebook 后，先克隆仓库：

```bash
%cd /kaggle/working
!git clone git@github.com:Chen-pcode/BLMNet.git
%cd /kaggle/working/BLMNet/sci_skinseg
```

如果 Kaggle 无法使用 SSH，也可以用 HTTPS：

```bash
!git clone https://github.com/Chen-pcode/BLMNet.git
```

安装依赖：

```bash
!pip install -r requirements.txt
```

检查数据集：

```bash
!python check_dataset.py \
  --data-root /kaggle/input/datasets/zichengdoctor \
  --out /kaggle/working/dataset_report.csv
```

正常情况下应看到：

- ISIC2017 train: 1500
- ISIC2017 val: 650
- ISIC2018 train: 1886
- ISIC2018 val: 808
- PH2 test: 200

## 5. 单模型运行方式

只训练并测试 BLMNet：

```bash
!python run_experiment.py \
  --data-root /kaggle/input/datasets/zichengdoctor \
  --train-dataset isic2018 \
  --val-dataset isic2018 \
  --test-datasets isic2017 PH2 \
  --model blmnet \
  --output-dir ./outputs/blmnet_isic2018 \
  --epochs 300 \
  --batch-size 16 \
  --img-size 256 \
  --seed 2026 \
  --amp \
  --color-jitter \
  --save-preds
```

参数说明：

- `--train-dataset`: 训练集，可选 `isic2017` 或 `isic2018`；
- `--val-dataset`: 验证集，通常与训练集相同；
- `--test-datasets`: 额外测试集；
- `--model`: 模型名，可选 `blmnet`、`malunet`、`lbunet`、`egeunet`、`unext`；
- `--output-dir`: 输出目录；
- `--epochs`: 最大训练轮数，实际可能因 early stopping 提前结束；
- `--amp`: 混合精度训练；
- `--color-jitter`: 使用颜色增强，主实验中必须打开；
- `--save-preds`: 保存预测 mask，用于论文可视化。

## 6. 主实验复现

主实验包含两套训练协议：

1. ISIC2018 train -> ISIC2018 / ISIC2017 / PH2
2. ISIC2017 train -> ISIC2017 / ISIC2018 / PH2

运行完整主实验：

```bash
!python run_suite.py \
  --data-root /kaggle/input/datasets/zichengdoctor \
  --output-root ./outputs_main \
  --suite full \
  --epochs 300 \
  --batch-size 16 \
  --img-size 256 \
  --seeds 2026 \
  --amp \
  --save-preds
```

如果 Kaggle 时间不够，可以拆开单独跑每个模型。例如：

```bash
!python run_experiment.py \
  --data-root /kaggle/input/datasets/zichengdoctor \
  --train-dataset isic2017 \
  --val-dataset isic2017 \
  --test-datasets isic2018 PH2 \
  --model blmnet \
  --output-dir ./outputs_main/full_isic2017_blmnet_seed2026 \
  --epochs 300 \
  --batch-size 16 \
  --img-size 256 \
  --seed 2026 \
  --amp \
  --color-jitter \
  --save-preds
```

## 7. 消融实验复现

消融实验包含：

- `blmnet`: 完整模型；
- `blmnet_no_boundary`: 去掉边界引导解码；
- `blmnet_no_scan`: 去掉选择性扫描上下文模块。

运行命令：

```bash
!python run_suite.py \
  --data-root /kaggle/input/datasets/zichengdoctor \
  --output-root ./outputs_ablation_v2 \
  --suite ablation \
  --epochs 300 \
  --batch-size 16 \
  --img-size 256 \
  --seeds 2026 \
  --amp \
  --save-preds
```

## 8. 输出文件说明

每个实验目录中包含：

```text
best.pt
config.json
history.csv
summary.csv
summary.json
samples_isic2017_val.csv
samples_isic2018_val.csv
samples_PH2_test.csv
val_samples_best.csv
val_samples_latest.csv
predictions/
```

含义：

- `best.pt`: 验证集 Dice 最佳模型；
- `config.json`: 本次运行参数；
- `history.csv`: 每个 epoch 的训练 loss 和验证指标；
- `summary.csv`: 最终测试集汇总指标；
- `samples_*.csv`: 每张图的逐样本指标；
- `predictions/`: 预测 mask 图像。

## 9. 生成论文表格和图

当前本地已将 Kaggle 结果放入：

```text
sci_skinseg/result_exper/
```

最终论文表格和图已经生成到：

```text
sci_skinseg/paper_outputs/
```

主要文件：

```text
main_all_results.csv
table_overall_mean.csv
table_overall_mean_formatted.csv
table_overall_mean.md
table_overall_mean.tex
table_main_protocols.csv
table_main_protocols.md
table_main_protocols.tex
ablation_all_results.csv
table_ablation_mean.csv
table_ablation_mean.md
fig_overall_dice.png
fig_overall_hd95.png
fig_pareto_params_dice.png
fig_protocol_dice_comparison.png
fig_ablation_dice.png
```

如需重新生成，可以根据 `result_exper` 中的 summary 文件运行对应统计脚本，或使用当前生成好的 `paper_outputs`。

## 10. 当前最终结果概览

跨 6 个训练-测试协议的平均结果：

| Model | Dice (%) | IoU (%) | mIoU (%) | Acc (%) | HD95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BLMNet | 89.85 | 83.30 | 87.76 | 95.43 | 14.08 |
| UNeXt | 88.90 | 81.83 | 86.71 | 95.07 | 16.33 |
| MALUNet | 88.73 | 81.56 | 86.51 | 94.91 | 15.35 |
| EGE-UNet | 88.60 | 81.55 | 86.69 | 95.04 | 15.47 |
| LB-UNet | 88.53 | 81.34 | 86.28 | 94.81 | 16.32 |

BLMNet 在 6 个训练-测试组合中均取得最高 Dice 和最低 HD95。

消融实验平均结果：

| Variant | Dice (%) | mIoU (%) | HD95 |
| --- | ---: | ---: | ---: |
| BLMNet | 90.18 | 87.82 | 14.02 |
| no scan | 90.13 | 87.75 | 14.20 |
| no boundary | 89.52 | 87.10 | 14.91 |

## 11. 复现注意事项

1. 主实验必须使用 `--color-jitter`，否则与当前论文结果设置不一致。
2. Kaggle 可能因最长运行时间中断，建议按模型拆分运行。
3. 如果中断后只有 `history.csv` 没有 `summary.csv`，说明最终评估没有完成，需要单独重跑该模型。
4. 论文主结果来自 seed `2026`。若后续要进一步增强可信度，可以补跑 3 个 seed。
5. `FLOPs` 当前主要统计卷积和线性层，不完整统计 `cumsum` 等非卷积操作，因此论文中应同时报告 `FPS`。

