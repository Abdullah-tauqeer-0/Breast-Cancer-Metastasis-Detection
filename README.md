# Detection, Localization, and Staging of Breast Cancer Lymph Node Metastasis

[![Nature Scientific Reports](https://img.shields.io/badge/Nature-Scientific%20Reports-green)](https://www.nature.com/articles/s41598-025-21787-9)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the official PyTorch implementation of the paper:
**"Detection, localization, and staging of breast cancer lymph node metastasis in digital pathology whole slide images using selective neighborhood attention-based deep learning"**
*Abdullah Tauqeer et al., Nature Scientific Reports (2025)*

## 📌 Abstract
We propose a novel deep learning framework for the precise detection and staging of breast cancer metastases in Whole Slide Images (WSIs). Our approach integrates **Selective Neighborhood Attention (SNA)** with a Multiple Instance Learning (MIL) backbone, allowing the model to effectively attend to local tissue context while maintaining global slide-level awareness.

Key contributions:
1.  **Selective Neighborhood Attention (SNA):** A gated attention mechanism that dynamically weighs local features based on tissue relevance.
2.  **Automated pN-Staging:** A robust pipeline to calculate tumor burden and assign pN stages (pN0, pN0(i+), pN1mi, pN1) with near-expert concordance.
3.  **State-of-the-Art Performance:** Achieves superior AUC and Kappa scores on the CAMELYON16 dataset compared to standard MIL approaches.

## 🚀 Features
*   **WSI Preprocessing:** Efficient tiling and tissue segmentation using HSV thresholding.
*   **NATTEN Integration:** Seamless integration with [Neighborhood Attention Transformer](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer).
*   **Staging Utility:** Automated calculation of tumor diameter and TNM staging.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Abdullah-tauqeer-0/Breast-Cancer-Metastasis-Detection.git
cd Breast-Cancer-Metastasis-Detection

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset
This project uses the **CAMELYON16** dataset.
1. Download WSIs from the [official challenge website](https://camelyon17.grand-challenge.org/).
2. Organize data as follows:
   ```
   data/
   ├── train/
   │   ├── normal/
   │   └── tumor/
   └── test/
   ```

## 💻 Usage

### 1. Preprocessing (Tiling)
Convert raw WSIs into patches:
```bash
python preprocessing/wsi_processing.py --slide_dir /path/to/slides --output_dir /path/to/patches
```

### 2. Training
Train the SNA-MIL model:
```bash
python train.py --epochs 100 --batch_size 32 --lr 1e-4
```

## 📈 Results
| Metric | Score |
| :--- | :--- |
| **AUC** | 0.985 |
| **F1-Score** | 0.957 |
| **Kappa (Staging)** | 0.940 |

## 📜 Citation
If you find this code useful, please cite our paper:
```bibtex
@article{tauqeer2025detection,
  title={Detection, localization, and staging of breast cancer lymph node metastasis...},
  author={Tauqeer, Abdullah and others},
  journal={Scientific Reports},
  year={2025},
  publisher={Nature Publishing Group}
}
```
