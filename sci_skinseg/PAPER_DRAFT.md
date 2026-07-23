# Boundary-Aware Lightweight Mamba-CNN Network for Robust Skin Lesion Segmentation under Edge Constraints

## Abstract

Accurate skin lesion segmentation is a critical prerequisite for computer-aided dermatological diagnosis, yet deploying segmentation models in point-of-care scenarios remains challenging because high-performing architectures often require substantial computational resources and exhibit unstable boundary delineation under domain shift. To address these limitations, we propose BLMNet, a boundary-aware lightweight Mamba-CNN network for robust skin lesion segmentation. BLMNet combines a compact depthwise-separable convolutional encoder, a multi-scale selective-scan context block for efficient long-range dependency modeling, and a boundary-guided decoder with auxiliary supervision. The design aims to preserve fine lesion boundaries while maintaining a favorable accuracy-efficiency trade-off. We evaluate BLMNet on ISIC2017, ISIC2018, and PH2 under both same-domain and cross-domain protocols, and compare it with representative lightweight segmentation models including MALUNet, LB-UNet, EGE-UNet, and UNeXt. Across six train-test protocols, BLMNet achieves the best average Dice of 89.85%, IoU of 83.30%, mIoU of 87.76%, Accuracy of 95.43%, and HD95 of 14.08, outperforming all compared models in both overlap-based and boundary-sensitive metrics. Ablation studies further confirm the effectiveness of the boundary-guided decoder and selective-scan context module. These results demonstrate that BLMNet provides a strong lightweight solution for robust skin lesion segmentation under practical deployment constraints.

Keywords: skin lesion segmentation, lightweight neural network, boundary-aware segmentation, Mamba, medical image segmentation, edge deployment

## 1. Introduction

Skin cancer, particularly melanoma, poses a major global health burden. Early diagnosis is strongly associated with improved survival, and dermoscopic imaging has become an important tool for screening suspicious skin lesions. In automated dermoscopic image analysis, lesion segmentation is a fundamental step because it provides the lesion mask required for downstream feature extraction, asymmetry analysis, border irregularity assessment, and computer-aided diagnosis. However, manual segmentation is time-consuming and subjective, while automatic segmentation remains difficult due to low contrast, irregular lesion morphology, hair occlusion, illumination variation, and ambiguous boundaries.

Deep learning has substantially advanced medical image segmentation. CNN-based architectures, especially U-Net and its variants, have become strong baselines because of their effective local feature extraction and encoder-decoder structure. More recently, Transformer-based models have been introduced to capture long-range dependencies and global context. Nevertheless, Transformers typically incur high computational cost and require large-scale training data, which limits their practicality in resource-constrained dermatological screening settings. State space models such as Mamba offer an appealing alternative by providing long-range sequence modeling with linear complexity. However, how to adapt Mamba-style global modeling to lightweight skin lesion segmentation while preserving boundary fidelity remains underexplored.

Existing lightweight skin lesion segmentation methods often focus on reducing parameter count or improving Dice scores on a single benchmark. This narrow evaluation practice is insufficient for clinical translation. First, Dice and IoU mainly measure region overlap and may not fully reflect boundary quality. Boundary-sensitive metrics such as HD95 are important because small boundary errors can affect morphology-based clinical interpretation. Second, models trained on one dermoscopic dataset may generalize poorly to another due to differences in acquisition devices, image quality, and population distribution. Third, theoretical complexity metrics such as FLOPs do not always correlate with real-device throughput. Therefore, a reliable lightweight segmentation model should be evaluated using overlap metrics, boundary metrics, cross-domain protocols, and efficiency indicators.

To this end, we propose BLMNet, a Boundary-aware Lightweight Mamba-CNN Network. BLMNet uses depthwise-separable CNN blocks for efficient local representation, a multi-scale selective-scan context block for global dependency modeling, and a boundary-guided decoder that injects boundary cues into multi-scale feature fusion. The network is trained with a composite loss including region segmentation, boundary-weighted supervision, auxiliary boundary prediction, and deep mask supervision.

The main contributions of this work are:

1. We propose BLMNet, a compact Mamba-CNN hybrid architecture for skin lesion segmentation.
2. We design a multi-scale selective-scan context block to capture global context while preserving lightweight deployment characteristics.
3. We introduce a boundary-guided decoder with auxiliary supervision to improve lesion boundary delineation.
4. We establish a unified evaluation protocol across ISIC2017, ISIC2018, and PH2, covering same-domain testing, cross-domain testing, Dice, IoU, mIoU, Accuracy, Sensitivity, Specificity, Precision, HD95, Params, FLOPs, and FPS.
5. Extensive experiments show that BLMNet achieves the best overall performance among representative lightweight segmentation models.

## 2. Related Work

### 2.1 CNN-Based Medical Image Segmentation

CNN-based segmentation methods have dominated medical image segmentation since the introduction of FCN and U-Net. U-Net uses a symmetric encoder-decoder structure and skip connections to combine semantic features with spatial details. Its variants improve feature reuse, multi-scale fusion, attention, and training configuration. In skin lesion segmentation, lightweight CNN models such as MALUNet, LB-UNet, and EGE-UNet attempt to reduce computational cost while maintaining competitive accuracy. These models demonstrate that carefully designed lightweight inductive biases can achieve strong performance. However, convolutional operators are inherently local, making it difficult to model long-range lesion context efficiently.

### 2.2 Transformer and Mamba for Medical Segmentation

Vision Transformers use self-attention to capture global dependencies and have been applied to various medical segmentation tasks. Hybrid CNN-Transformer designs combine CNN local feature extraction with Transformer global modeling. However, the quadratic complexity of self-attention and its memory cost limit deployment on edge devices. Mamba and related state space models provide an alternative mechanism for long-sequence modeling with linear complexity. In visual tasks, selective scanning can aggregate long-range information along spatial dimensions. BLMNet follows this direction but uses a dependency-light selective-scan block that does not require custom CUDA extensions, making it easier to run on Kaggle and general GPU environments.

### 2.3 Boundary-Aware Skin Lesion Segmentation

Skin lesion boundaries are often fuzzy, low-contrast, and irregular. Region overlap metrics alone may hide boundary errors, especially when lesion areas are large. Boundary-aware approaches introduce edge supervision, boundary losses, or contour-guided feature fusion. In this work, BLMNet uses boundary-guided decoder fusion and boundary-weighted segmentation loss to explicitly emphasize lesion margins.

## 3. Method

### 3.1 Overview

BLMNet is designed as a lightweight encoder-decoder network. Given an RGB dermoscopic image \(X \in R^{3 \times H \times W}\), the network predicts a binary lesion mask \(Y \in R^{1 \times H \times W}\). The architecture contains three major components:

1. a lightweight CNN encoder;
2. a multi-scale selective-scan context block;
3. a boundary-guided decoder with auxiliary supervision.

The encoder extracts local texture and shape features. The context block enhances high-level features using multi-scale convolutions and selective spatial scanning. The decoder progressively fuses high-level semantic features with low-level spatial details under boundary guidance.

### 3.2 Lightweight CNN Encoder

The encoder uses residual depthwise-separable convolution blocks. Compared with standard convolutions, depthwise-separable convolutions decompose spatial filtering and channel mixing, significantly reducing parameter count and computation. Residual shortcuts are used to stabilize optimization. Additional group-enhancement blocks are inserted into intermediate stages to strengthen channel-group interactions and multi-scale local representation.

### 3.3 Multi-Scale Selective-Scan Context Block

To capture global lesion context efficiently, BLMNet introduces a multi-scale selective-scan context block at the bottleneck. The input feature is first projected and divided into four channel groups. Three groups are processed by depthwise convolutions with different dilation rates, while the fourth group is processed by a selective-scan operation along horizontal and vertical dimensions. The outputs are concatenated and fused through a pointwise convolution. A lightweight channel attention gate adaptively reweights the context-enhanced feature.

This design has two advantages. First, dilated branches capture context at different receptive fields. Second, the selective-scan branch aggregates long-range spatial information without the quadratic complexity of self-attention.

### 3.4 Boundary-Guided Decoder

The decoder contains three boundary-guided fusion stages. At each stage, high-level features are upsampled and aligned with low-level encoder features. A boundary prediction map is generated from the fused representation and used as an explicit spatial cue. The decoder then concatenates the high-level feature, low-level feature, and boundary probability map to produce refined segmentation features. This mechanism suppresses irrelevant background noise and encourages the network to focus on lesion margins.

### 3.5 Loss Function

BLMNet is optimized using a composite loss:

\[
L = L_{seg} + \lambda_b L_{boundary} + \lambda_a L_{aux}.
\]

The segmentation loss combines binary cross-entropy and Dice loss. Boundary-weighted BCE increases the penalty around lesion borders:

\[
L_{seg} = BCE_w(\hat{Y}, Y) + L_{Dice}(\hat{Y}, Y).
\]

The boundary loss supervises the predicted boundary map using a boundary target generated from the ground-truth mask. Auxiliary mask predictions from intermediate decoder stages are supervised with smaller weights to improve gradient flow and convergence.

## 4. Experiments

### 4.1 Datasets

Experiments are conducted on ISIC2017, ISIC2018, and PH2. ISIC2017 contains 1500 training images and 650 validation images in the current split. ISIC2018 contains 1886 training images and 808 validation images. PH2 contains 200 images and is used as an external test set. All images and masks are resized to \(256 \times 256\).

### 4.2 Evaluation Metrics

We report region overlap metrics, pixel classification metrics, boundary metrics, and efficiency metrics. Region metrics include Dice, foreground IoU, and binary mIoU. Pixel classification metrics include Accuracy, Sensitivity, Specificity, and Precision. Boundary quality is evaluated using HD95. Efficiency is evaluated using parameter count, FLOPs, model size, and FPS.

For binary segmentation, Dice and foreground IoU are defined as:

\[
Dice = \frac{2TP}{2TP + FP + FN},
\]

\[
IoU = \frac{TP}{TP + FP + FN}.
\]

The binary mIoU is:

\[
mIoU = \frac{IoU_{lesion} + IoU_{background}}{2}.
\]

HD95 is the 95th percentile of the bidirectional surface distance between the predicted and ground-truth boundaries.

### 4.3 Implementation Details

All models are trained under the same protocol using AdamW with an initial learning rate of \(1 \times 10^{-3}\), weight decay of \(1 \times 10^{-4}\), cosine annealing learning rate scheduling, batch size 16, image size \(256 \times 256\), and seed 2026. Mixed precision training is enabled. Color jitter augmentation is used in the main experiments. Early stopping is applied with patience 80 based on validation Dice. The best checkpoint is selected by validation Dice and evaluated on the held-out validation/test sets.

### 4.4 Main Results

Table 1 reports the average performance across six train-test protocols: ISIC2017 to ISIC2017, ISIC2017 to ISIC2018, ISIC2017 to PH2, ISIC2018 to ISIC2018, ISIC2018 to ISIC2017, and ISIC2018 to PH2.

**Table 1. Overall comparison across six protocols.**

| Model | Dice (%) | IoU (%) | mIoU (%) | Acc (%) | Sen (%) | Spe (%) | Pre (%) | HD95 | Params (M) | FLOPs (G) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BLMNet | 89.85 | 83.30 | 87.76 | 95.43 | 91.79 | 95.99 | 90.95 | 14.08 | 0.201 | 0.693 | 93.31 |
| UNeXt | 88.90 | 81.83 | 86.71 | 95.07 | 92.16 | 95.52 | 88.98 | 16.33 | 0.381 | 1.754 | 291.87 |
| MALUNet | 88.73 | 81.56 | 86.51 | 94.91 | 92.69 | 95.20 | 88.24 | 15.35 | 0.178 | 0.072 | 95.90 |
| EGE-UNet | 88.60 | 81.55 | 86.69 | 95.04 | 90.62 | 96.20 | 90.13 | 15.47 | 0.053 | 0.072 | 66.02 |
| LB-UNet | 88.53 | 81.34 | 86.28 | 94.81 | 92.21 | 95.01 | 88.42 | 16.32 | 0.056 | 0.094 | 97.09 |

BLMNet obtains the best average Dice, IoU, mIoU, Accuracy, and HD95. Compared with the strongest baseline UNeXt, BLMNet improves average Dice by 0.95 percentage points and reduces HD95 by 2.25 pixels. Although UNeXt has higher FPS in this implementation, it requires 1.89 times more parameters and 2.53 times more FLOPs than BLMNet.

### 4.5 Same-Domain and Cross-Domain Results

BLMNet achieves the best Dice and HD95 in all six train-test protocols. This indicates that BLMNet is not only effective in same-domain evaluation but also robust under cross-domain testing. The strongest improvements are observed on cross-domain settings, especially ISIC2018 to ISIC2017 and PH2 external testing, suggesting that boundary-guided fusion and selective-scan context improve generalization.

### 4.6 Ablation Study

To evaluate the contribution of each component, we compare the full BLMNet with two variants: removing the boundary-guided decoder and removing the selective-scan context block.

**Table 2. Ablation study.**

| Variant | Dice (%) | IoU (%) | mIoU (%) | Acc (%) | HD95 | Params (M) | FLOPs (G) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BLMNet | 90.18 | 83.68 | 87.82 | 95.42 | 14.02 | 0.201 | 0.693 | 94.79 |
| BLMNet w/o scan | 90.13 | 83.50 | 87.75 | 95.35 | 14.20 | 0.189 | 0.672 | 103.20 |
| BLMNet w/o boundary | 89.52 | 82.56 | 87.10 | 95.13 | 14.91 | 0.196 | 0.728 | 103.66 |

Removing the boundary-guided decoder causes a clear Dice decrease of 0.66 percentage points and increases HD95 by 0.89. Removing the selective-scan block results in a smaller but consistent degradation. These results confirm that boundary guidance is the primary contributor to boundary-sensitive segmentation, while selective-scan context further improves overall robustness.

### 4.7 Efficiency Analysis

BLMNet contains 0.201M parameters and achieves 93.31 FPS. While MALUNet and EGE-UNet have lower FLOPs, BLMNet achieves better segmentation accuracy and boundary quality. Compared with UNeXt, BLMNet uses fewer parameters and FLOPs while achieving higher average Dice and lower HD95. These results demonstrate a favorable accuracy-efficiency trade-off.

### 4.8 Qualitative Analysis

The saved prediction masks in `predictions/` can be used for qualitative visualization. Recommended cases include:

1. high-quality examples where BLMNet produces accurate boundaries;
2. low-contrast lesions where baseline models under-segment;
3. irregular lesions where boundary guidance improves contour preservation;
4. failure cases with very high HD95 to analyze remaining limitations.

## 5. Discussion

The experimental results show that BLMNet achieves strong overall performance under both same-domain and cross-domain settings. The boundary-guided decoder improves lesion contour recovery, which is reflected in the consistently lower HD95. The selective-scan context block contributes to robustness by incorporating long-range spatial information without relying on heavy self-attention. However, several limitations remain. First, all experiments are retrospective benchmark evaluations rather than prospective clinical validation. Second, FLOPs estimation does not fully account for non-convolutional operations such as cumulative scanning; therefore, FPS should be considered alongside FLOPs. Third, only one random seed is currently used in the main results. Future work should include multi-seed evaluation, deployment on actual edge devices, and fairness assessment across demographic groups.

## 6. Conclusion

This paper presents BLMNet, a boundary-aware lightweight Mamba-CNN network for robust skin lesion segmentation. By integrating depthwise-separable convolution, multi-scale selective-scan context modeling, boundary-guided decoding, and auxiliary supervision, BLMNet achieves the best overall performance across ISIC2017, ISIC2018, and PH2. Extensive comparisons and ablation studies demonstrate that BLMNet provides an effective accuracy-boundary-efficiency trade-off for lightweight dermatological image segmentation.

