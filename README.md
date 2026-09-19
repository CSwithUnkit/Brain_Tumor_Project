---
title: NeuroScan Enterprise
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.38.0"
app_file: dashboard/app.py
pinned: false
---

# MRI Image Enhancement and Tumor Detection
### An Explainable Deep Learning Framework for Brain Tumor Segmentation and Classification

**Authors:** Ankit Yadav, Bhaskar Rawat, Harsh Singh, Mayank Chandra Das  
**Supervisor:** Mr. Manish Kumar Sharma  
**Affiliation:** ITS Engineering College, AKTU  

---

## Architecture Overview

1. **Image Enhancement:** Fixed sequence pipeline utilizing Wavelet Packet Transform (WPT) -> Linear Minimum Mean Square Error (LMMSE) -> Contrast Limited Adaptive Histogram Equalization (CLAHE) for maximal noise reduction and contrast recovery.
2. **Segmentation:** U-Net architecture powered by a compound Tversky Focal Loss and cKDTree Hausdorff Distance (HD95) for robust small-tumor binary segmentation.
3. **Classification:** EfficientNetB2 with Two-Stage Transfer Learning. Assessed via 3 experiments: 
   - Exp 1: Baseline 
   - Exp 2: Enhanced 
   - Exp 3: Cascaded Segmentation-Guided crop
4. **Explainability:** Grad-CAM feature attribution on `features[-1]` combined with quantitative IoU spatial localization analysis.
5. **External Generalization:** Zero-retraining PMRAM validation on 1,600 original scans to measure clinical generalization.

---

## Google Colab Execution Guide

To replicate the study or train the models from scratch on Google Colab, execute the 3 independent, modular notebooks strictly in order:

### Step 1: Data Preparation & Caching
Open `notebooks/01_data_prep_and_enhancement.ipynb`
- Ingests datasets and verifies file integrity using MD5 hashing.
- Runs the offline Enhancement Caching strategy (WPT+LMMSE+CLAHE) to completely eliminate CPU bottlenecks during model training.

### Step 2: Tumor Segmentation Training
Open `notebooks/02_train_unet_segmentation.ipynb`
- Trains the U-Net model on enhanced cached images.
- Reaches peak validation Dice using Tversky Focal Loss and strictly saves the best checkpoint as `best_unet_enhanced.pth`.

### Step 3: Classification, XAI, & Dashboard Launch
Open `notebooks/03_train_classifier_and_xai.ipynb`
- Executes two-stage transfer learning to train EfficientNetB2.
- Evaluates XAI, generates Grad-CAM visualizations, and executes external generalization tests.
- Launches the live **NeuroScan-Enterprise** Dashboard directly from Colab via a secure Cloudflare Tunnel.

---

## Clinical Notes & Disclaimer
*This repository represents a research prototype and is strictly intended for academic and computational study. The software provided is not FDA or CE approved for live diagnostic inference or clinical decision-making. Always consult a licensed radiologist or neurosurgeon.*
