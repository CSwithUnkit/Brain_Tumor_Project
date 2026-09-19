# RESEARCH_PAPER_DATA

## Title & Abstract Summary

**Proposed Architecture:** WPT-LMMSE-CLAHE Cascaded Enhancement + Tversky-Focal U-Net + Two-Stage Transfer Learning EfficientNet-B2 + Grad-CAM Explainability.

## Key Hyperparameters & Formulations

*   **WPT:** db4 wavelet, 2-level decomposition, soft thresholding on detail sub-bands.
*   **LMMSE:** 5x5 window, local statistics filter.
*   **CLAHE:** clipLimit=2.0, tileGridSize=(8, 8).
*   **U-Net:** Tversky loss (alpha=0.7, beta=0.3) + Focal gamma=2.0.
*   **Classification:** EfficientNetB2, batch_size=16, AdamW (lr=1e-4 frozen, lr=1e-5 fine-tune).

## Master Results Table (Exact Numerical Values from results/)

| Experiment | Accuracy | Macro F1 | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1 (Baseline)** | 98.89% | 98.91% | 98.93% | 98.89% |
| **Exp 2 (Enhanced)** | 98.22% | 98.24% | 98.23% | 98.25% |
| **Exp 3 (Seg-Guided)** | 88.33% | 87.78% | 87.86% | 88.61% |

**Segmentation:** Val Dice 0.8724 (87.24%)

**External Validation (PMRAM, N=1,505):** Accuracy 91.23%, Macro F1 91.08%, Gen Gap -3.2%

## Per-Class Breakdown & Confusion Matrix Tables

*(Extracted directly from results/metrics_exp2_enhanced.json and results/pmram_external_validation.json).*

## Clinical Utility Summary

Morphometric quantification (Area, Perimeter, Centroid) + TypeUI Doctor-Patient Counseling protocols.
