# 论文完整框架

## 拟题

**Boundary-Aware Lightweight Mamba-CNN Network for Robust Skin Lesion Segmentation under Edge Constraints**

中文理解：  
面向边缘部署约束的边界感知轻量 Mamba-CNN 皮肤病灶分割网络。

## 摘要逻辑

1. 皮肤病灶分割是皮肤癌辅助诊断的重要前处理步骤。
2. 现有高精度模型通常计算量大，不适合边缘设备部署。
3. 轻量化模型虽然高效，但容易损失边界细节和跨域泛化能力。
4. 本文提出 BLMNet：轻量 CNN 局部编码 + Mamba-style 选择性扫描全局建模 + 边界引导解码。
5. 在 ISIC2017、ISIC2018 和 PH2 上进行同域、跨域、消融和效率实验。
6. 结果证明 BLMNet 在 Dice、HD95 和效率之间取得更优 Pareto 权衡。

## 1. Introduction

写作要点：

- 皮肤癌和黑色素瘤早筛的重要性。
- 分割能够提供病灶边界、面积、形态等可解释特征。
- CNN/U-Net 方法成熟，但全局建模不足。
- Transformer 全局建模强，但计算复杂度高。
- Mamba/SSM 具有线性复杂度，适合轻量全局建模，但在皮肤病灶边界恢复方面仍未充分研究。
- 当前轻量化皮肤病灶分割存在三个问题：
  - 多数方法只追求 Dice，忽略 HD95 等边界指标；
  - 只在单一数据集测试，跨域泛化不足；
  - 缺少 FLOPs、FPS、Params 的统一效率评价。

贡献写法：

1. 提出一种边界感知轻量 Mamba-CNN 网络 BLMNet。
2. 设计无自定义 CUDA 依赖的选择性扫描上下文模块，用于低成本全局建模。
3. 设计边界引导多尺度解码器，通过辅助边界监督改善病灶边缘恢复。
4. 在 ISIC2017、ISIC2018、PH2 上建立统一评价协议，报告 Dice、IoU、HD95、Params、FLOPs 和 FPS。

## 2. Related Work

### 2.1 CNN-based Medical Image Segmentation

讨论 U-Net、UNet++、Attention U-Net、nnU-Net、轻量 CNN。

### 2.2 Transformer and Mamba for Medical Segmentation

讨论 Transformer 的全局建模优势和高复杂度问题；引出 Mamba 的线性复杂度和选择性状态空间机制。

### 2.3 Lightweight Skin Lesion Segmentation

讨论 EGE-UNet、MALUNet、LB-UNet、UNeXt 等轻量模型。

### 2.4 Boundary-Aware Segmentation

强调皮肤病灶边界模糊、低对比度、毛发遮挡等问题；说明 HD95 的必要性。

## 3. Method

### 3.1 Overall Architecture

BLMNet 由三部分组成：

- Lightweight CNN encoder
- Selective-scan global context block
- Boundary-guided decoder

### 3.2 Lightweight CNN Encoder

使用深度可分离卷积和残差连接降低参数和计算量。

### 3.3 Selective-Scan Global Context Block

沿水平和垂直方向执行累计扫描，模拟 Mamba 式长程依赖建模，同时避免复杂 CUDA 扩展。

### 3.4 Boundary-Guided Decoder

每层解码时预测边界图，并将边界概率作为显式引导参与高低层特征融合。

### 3.5 Loss Function

复合损失：

- BCE + Dice segmentation loss
- boundary-weighted segmentation loss
- auxiliary boundary BCE loss

## 4. Experiments

### 4.1 Datasets

- ISIC2017
- ISIC2018
- PH2

### 4.2 Implementation Details

建议写：

- image size: 256 x 256
- optimizer: AdamW
- learning rate: 1e-3
- scheduler: cosine annealing
- epochs: 300
- batch size: 16
- seeds: 2026, 2027, 2028
- hardware: Kaggle GPU，最终根据实际输出填写

### 4.3 Evaluation Metrics

区域指标：Dice、前景 IoU、mIoU。  
分类指标：Accuracy、Sensitivity、Specificity、Precision。  
边界指标：HD95。  
效率指标：Params、FLOPs、FPS、model size。

### 4.4 Comparison with State-of-the-Art Lightweight Models

对比：

- MALUNet
- LB-UNet
- UNeXt
- EGE-UNet
- BLMNet

### 4.5 Cross-Domain Generalization

训练-测试组合：

- ISIC2017 训练集 -> ISIC2017 验证/测试集
- ISIC2018 训练集 -> ISIC2018 验证/测试集
- ISIC2018 -> ISIC2017
- ISIC2018 -> PH2
- ISIC2017 -> ISIC2018
- ISIC2017 -> PH2

说明：当前数据目录中 ISIC2017 和 ISIC2018 使用 `val` 目录作为测试/验证集；论文写作时可以称为 held-out test set 或 validation/test split，具体根据数据来源说明统一表述。

### 4.6 Ablation Study

对比：

- BLMNet full
- without boundary-guided decoder
- without selective-scan block

### 4.7 Efficiency Analysis

绘制：

- Params-Dice Pareto
- FLOPs-HD95 Pareto
- FPS-Dice comparison

### 4.8 Qualitative Analysis

选取典型图像：

- 边界清晰样本
- 边界模糊样本
- 小病灶样本
- 低对比度样本
- 跨域失败样本

## 5. Discussion

讨论重点：

- 为什么边界引导能改善 HD95。
- 为什么选择性扫描有利于全局上下文建模。
- 为什么轻量模型在跨域上容易退化。
- 当前方法的局限：还没有真实移动端部署、没有前瞻性临床验证、没有肤色公平性元数据。

## 6. Conclusion

总结 BLMNet 在轻量化、边界质量和跨域泛化上的贡献，并指出未来将扩展到真实边缘设备和临床场景验证。
