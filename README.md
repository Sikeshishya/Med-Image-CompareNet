# 🏥 Med-Image CompareNet

**Comparative Deep Learning Framework for Medical Image Analysis**

> *Which AI architecture — CNNs or Vision Transformers — performs better on each medical imaging modality, and why?*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-green.svg)](https://streamlit.io)

---

## 📋 Project Overview

Med-Image CompareNet is a comprehensive research framework that compares **CNN architectures** (ResNet-50, DenseNet-121) against **Vision Transformers** (ViT-B/16) across two fundamentally different medical imaging modalities:

| Modality | Characteristics | Dataset |
|----------|----------------|---------|
| **Chest X-Ray** | Low-dose, noisy, global structure | [Chest X-Ray Pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) |
| **Histopathology** | Gigapixel, high-detail, patch-based | [Breast Histopathology (IDC)](https://www.kaggle.com/paultimothymooney/breast-histopathology-images) |

### Four Integrated Modules

1. **Module 1 — Image Enhancement**: U-Net + GAN denoising for low-dose X-rays
2. **Module 2 — CNN Classification**: ResNet-50 & DenseNet-121 baselines
3. **Module 3 — ViT Classification**: Vision Transformer fine-tuning
4. **Module 4 — Explainable AI**: Grad-CAM, Attention Rollout, Pointing Game

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd Major\ project
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Download Datasets

Download from Kaggle and place in `data/`:

```
data/
├── chest_xray/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   └── test/
└── breast_histopathology/
    ├── <patient_id>/
    │   ├── 0/    (benign)
    │   └── 1/    (malignant)
    └── ...
```

**Dataset Links:**
- [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

### 3. Train Models

```bash
# Module 1: X-Ray Enhancement
python -m src.enhance

# Module 2: CNN Classification
python -m src.cnn_model

# Module 3: ViT Classification
python -m src.vit_model
```

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
Major project/
├── config.yaml              # All hyperparameters (no hardcoded values)
├── requirements.txt         # Pinned dependencies
├── README.md                # This file
│
├── src/                     # Core modules
│   ├── __init__.py
│   ├── utils.py             # Config, seeds, checkpoints, logging
│   ├── datasets.py          # Data loaders for both modalities
│   ├── enhance.py           # Module 1: U-Net + GAN enhancement
│   ├── cnn_model.py         # Module 2: ResNet/DenseNet classification
│   ├── vit_model.py         # Module 3: ViT classification
│   ├── xai.py               # Module 4: Grad-CAM, Attention Rollout
│   └── compare.py           # Comparative analysis engine
│
├── dashboard/               # Streamlit web dashboard
│   ├── __init__.py
│   └── app.py               # Main dashboard application
│
├── data/                    # Datasets (not in git)
├── checkpoints/             # Saved model weights
├── results/                 # Evaluation results (JSON)
├── figures/                 # Generated charts
├── exports/                 # PDF reports
└── logs/                    # Training logs
```

---

## ⚙️ Configuration

All hyperparameters are in `config.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `project.seed` | 42 | Random seed for reproducibility |
| `cnn.training.epochs` | 30 | CNN training epochs |
| `vit.training.epochs` | 30 | ViT training epochs |
| `enhancement.noise_level` | 25 | Gaussian noise sigma |
| `cnn.training.learning_rate` | 0.0001 | CNN learning rate |
| `vit.training.learning_rate` | 0.00005 | ViT learning rate (lower for transformers) |

---

## 📊 Results

| Model | X-Ray Accuracy | X-Ray F1 | Pathology Accuracy | Pathology F1 |
|-------|---------------|----------|-------------------|-------------|
| ResNet-50 | 94.6% | 95.7% | 88.0% | 87.9% |
| DenseNet-121 | 93.0% | 94.5% | 86.9% | 86.8% |
| ViT-B/16 | 94.1% | 95.4% | 89.4% | 89.4% |

**Key Insight**: CNNs (ResNet-50) dominate X-ray classification due to strong spatial inductive biases. ViT excels at pathology where global self-attention captures long-range cellular dependencies.

---

## 🧠 Explainable AI

- **Grad-CAM**: Gradient-weighted class activation maps for CNN models
- **Attention Rollout**: Multi-layer attention visualization for ViT
- **Pointing Game**: Quantitative metric — does the model look at the right region?

---

## 👥 Authors

B.E. Computer Science (Cyber Security) — Final Year Capstone Project

## 📄 License

This project is for academic/educational purposes.
