# RESEARCH_PAPER_DATA

## Title & Abstract Summary

**Proposed Architecture:** WPT-LMMSE-CLAHE Cascaded Enhancement + Tversky-Focal U-Net + Two-Stage Transfer Learning EfficientNet-B2 + Grad-CAM Explainability.

## Key Hyperparameters & Formulations

*   **WPT:** db4 wavelet, 2-level decomposition, soft thresholding on detail sub-bands.
*   **LMMSE:** 5x5 window, local statistics filter.
*   **CLAHE:** clipLimit=2.0, tileGridSize=(8, 8).
*   **U-Net:** Tversky loss (alpha=0.7, beta=0.3) + Focal gamma=0.75.
*   **Classification:** EfficientNetB2, two-stage transfer learning — Stage 1 head warmup AdamW lr=1e-3 (5 epochs, frozen backbone); Stage 2 differential fine-tuning, backbone lr=1e-5, classifier head lr=1e-4. Batch size derived from hardware profile (not fixed).

## Master Results Table (Exact Numerical Values from results/)

Best-validation-epoch metrics from `results/metrics_exp{1,2,3}_*.json` (validation
n=900 per experiment; no held-out test set in these runs). ⚠️ Exp 3 numbers pre-date
the 2026-09-27 fix for the train/validation masking mismatch and are not valid —
re-run required before citing.

| Experiment | Accuracy | Macro F1 | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1 (Baseline)** | 98.89% | 98.91% | 98.93% | 98.89% |
| **Exp 2 (Enhanced)** | 98.22% | 98.24% | 98.23% | 98.25% |
| **Exp 3 (Seg-Guided)** ⚠️ pre-fix | 88.33% | 87.78% | 87.86% | 88.61% |

*Note: Exp 2 (enhanced) did not beat the raw baseline (98.22% < 98.89%).*

**Segmentation:** Best val Dice **0.8806** (88.06%), epoch 24 of 25 —
source: `checkpoints/unet/unet_training_history_enhanced.json`
(agrees with `reports/PHASE_I_EVALUATION_REPORT.md`; the previously quoted 0.8724 was stale/incorrect).

**External Validation (PMRAM, N=1,505):** Accuracy 91.23%, Macro F1 91.08%, Gen Gap −3.23%
— source: `results/pmram_external_validation.json`. The stored BRISC reference (88.00%
acc) matches Exp 3's best val epoch, not the Exp 1 baseline; checkpoint provenance
for the PMRAM run is not recorded — treat as indicative.

## Per-Class Breakdown & Confusion Matrix Tables

*(Extracted directly from results/metrics_exp2_enhanced.json and results/pmram_external_validation.json).*

## Clinical Utility Summary

Morphometric quantification (Area, Perimeter, Centroid) + TypeUI Doctor-Patient Counseling protocols.
