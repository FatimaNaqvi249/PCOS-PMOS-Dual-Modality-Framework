# Endocrine Phenotype First: A Dual-Modality Machine Learning Framework for PCOS/PMOS Diagnosis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20GPU-blue)](https://kaggle.com)

---

## 📄 Paper

**Title:** Endocrine Phenotype First: A Dual-Modality Machine Learning Framework for PCOS Diagnosis with SHAP-Based Interpretation and Age-Stratified Validation

**Authors:** Syeda Fatima Naqvi · Sabien Sibtain · Dr. Adil Khan

**Affiliation:** Department of Computer Science, Bahria University, Islamabad, Pakistan

**Contact:** fatimanaqvi847@gmail.com

---

## 🧠 Overview

Polycystic Ovary Syndrome (PCOS) — now officially renamed to **Polyendocrine Metabolic Ovarian Syndrome (PMOS)** — affects 8–13% of women of reproductive age worldwide. Current diagnostic criteria (Rotterdam consensus) assign equal diagnostic weight to hormonal dysfunction and ultrasound-derived follicle counts, despite growing evidence that ovarian morphology is a downstream consequence of endocrine dysregulation, not its cause.

This repository presents a **dual-modality machine learning framework** that:

- Classifies PCOS from **clinical tabular data** (hormonal + metabolic + lifestyle features) using XGBoost, Random Forest, and L1-Logistic Regression
- Classifies PCOS from **ovarian ultrasound images** using three state-of-the-art deep learning architectures: EfficientNetV2-S, ConvNeXt-Small, and Swin Transformer-Tiny
- Applies **SHAP (SHapley Additive exPlanations)** to identify which biomarkers truly drive diagnosis
- Provides **age-stratified evaluation** across five age groups to reveal diagnostic sensitivity changes with age
- Applies **Grad-CAM++** visualization to confirm CNN attention localizes to follicular structures
- Performs **MD5-based duplicate detection** — a critical preprocessing step missing from all prior ultrasound PCOS studies

---

## 🔑 Key Contributions

### 1. First Systematic MD5 Duplicate Detection in Ovarian Ultrasound
- Raw corpus: **11,784 images**
- Duplicate groups found: **1,956** (all intra-class — zero cross-class)
- Duplicate files removed: **7,788 (66.1% of raw data)**
- Clean corpus used for evaluation: **3,996 images**
- This eliminates train/test data leakage that inflates accuracy in prior published work

### 2. State-of-the-Art CNN Benchmark on Clean Data
- **ConvNeXt-Small** achieves **99.83% accuracy and AUC-ROC = 1.0000** on the deduplicated test set
- First application of ConvNeXt architecture to PCOS ultrasound classification
- Progressive two-phase fine-tuning with differential learning rates

### 3. SHAP-Driven Feature Attribution Challenging Clinical Dogma
- Top predictors: **Follicle No. (R) = 2.490**, hair growth = 1.158, weight gain = 0.781, Follicle No. (L) = 0.741, AMH = 0.700
- **LH/FSH ratio does not appear in the top 15 features** — contradicting conventional clinical belief
- SHAP dependence plot confirms a non-linear threshold at LH/FSH = 2.0, consistent with clinical cutoffs

### 4. Age-Stratified Evaluation
- Peak recall = **1.000 in the 20–25 age group** (42/42 PCOS cases correctly identified)
- Recall declines to **0.964 in the >35 group** — consistent with declining AMH after age 30
- First ML PCOS study to report age-stratified diagnostic performance

### 5. Grad-CAM++ Interpretability for Ultrasound CNN
- Confirms model attention localizes to follicular ring structures
- Validates that high accuracy reflects genuine ovarian morphology learning, not image artefacts

---

## 📊 Results Summary

### Ultrasound CNN — Test Set (N=600, after deduplication)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| **ConvNeXt-Small** | **0.9983** | **0.9979** | **1.0000** | **0.9990** | **1.0000** |
| EfficientNetV2-S | 0.9917 | 0.9958 | 0.9937 | 0.9948 | 0.9986 |
| Swin-Transformer-Tiny | 0.9767 | 0.9936 | 0.9770 | 0.9852 | 0.9985 |
| Ensemble (soft voting) | 0.9900 | 0.9917 | 0.9958 | 0.9937 | 0.9993 |

### Tabular Classifiers — 5-Fold CV on SMOTE-Balanced Training Set

| Model | Accuracy | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| **XGBoost** | **0.9969 ± 0.0030** | **0.9973 ± 0.0036** | **0.9969 ± 0.0030** | **0.9999 ± 0.0001** |
| Random Forest | 0.9946 ± 0.0023 | 0.9919 ± 0.0044 | 0.9946 ± 0.0023 | 0.9999 ± 0.0001 |
| LR (L1) | 0.9044 ± 0.0104 | 0.9165 ± 0.0238 | 0.9055 ± 0.0107 | 0.9641 ± 0.0077 |

### Tabular Classifiers — Held-Out Test Set (N=400: 278 non-PCOS, 122 PCOS)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| XGBoost | 0.9901 | 0.988 | 0.991 | 0.989 | 0.9999 |
| Random Forest | 0.9875 | 0.985 | 0.988 | 0.986 | 0.9999 |
| LR (L1) | 0.9700 | 0.966 | 0.972 | 0.969 | 0.9940 |

### Age-Stratified XGBoost Performance

| Age Group | N (test) | PCOS Count | Accuracy | Recall |
|---|---|---|---|---|
| < 20 | 58 | 17 | 0.9655 | 0.966 |
| 20–25 | 142 | 42 | 0.9929 | 1.000 |
| 26–30 | 108 | 32 | 0.9907 | 0.989 |
| 31–35 | 62 | 19 | 0.9839 | 0.984 |
| > 35 | 30 | 9 | 0.9667 | 0.964 |

### Top 10 SHAP Features (XGBoost, Test Set)

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | Follicle No. (R) | 2.490 |
| 2 | Hair growth (Y/N) | 1.158 |
| 3 | Weight gain (Y/N) | 0.781 |
| 4 | Follicle No. (L) | 0.741 |
| 5 | AMH (ng/mL) | 0.700 |
| 6 | Skin darkening (Y/N) | 0.624 |
| 7 | Cycle (R/I) | 0.443 |
| 8 | Fast food (Y/N) | 0.385 |
| 9 | Pimples (Y/N) | 0.281 |
| 10 | Avg. F size (R) (mm) | 0.189 |

---

## 📁 Repository Structure
PCOS-PMOS-Dual-Modality-Framework/

│

├── notebooks/

│   ├── PCOS_Code_Part1.ipynb        # Part 1: Tabular clinical data

│   │                                 # (preprocessing, EDA, XGBoost,

│   │                                 #  Random Forest, LR, SHAP,

│   │                                 #  age-stratified analysis)

│   │

│   └── PCOS_Code_Part2.ipynb        # Part 2: Ultrasound image data

│                                     # (MD5 dedup, CLAHE, EfficientNetV2-S,

│                                     #  ConvNeXt-Small, Swin-Transformer-T,

│                                     #  TTA, ensemble, Grad-CAM++)

│

├── images/                           # Figures used in the paper

│   ├── fig1.png                  # Framework overview diagram

│   ├── fig2.png                      # Class distribution before/after dedup

│   ├── fig3.png                      # CLAHE sample images

│   ├── fig4.jpg                      # Training curves

│   ├── fig5.png                      # Confusion matrices (tabular)

│   ├── fig6.png                      # ROC curves (tabular)

│   ├── fig7.png                      # SHAP global feature importance

│   └── fig9.png                      # Grad-CAM++ localization maps


├── paper/

│   └── paper.tex                     # Full IEEE LaTeX source


├── requirements.txt                  # Python dependencies

├── README.md                         # This file

└── .gitignore                        # Files excluded from repository

---

## Datasets

> ⚠️ Datasets are **not included** in this repository due to Kaggle licensing. Download them directly from Kaggle using the links below.

### Dataset 1 — Clinical Tabular Data
- **Source:** https://www.kaggle.com/datasets/cm037divya/pcos-dataset
- **Credit:** C. M. Divya (Kaggle, 2024)
- **Size:** 2,000 patients × 44 features
- **Classes:** 608 PCOS-positive (30.4%), 1,392 healthy controls (69.6%)

### Dataset 2 — Ultrasound Images
- **Source:** https://www.kaggle.com/datasets/ibadeus/pcos-xai-ultrasound-dataset
- **Credit:** ibadeus (Kaggle, 2023)
- **Raw size:** 11,784 images (6,784 infected + 5,000 noninfected)
- **After MD5 deduplication:** 3,996 images (3,184 infected + 812 noninfected)

---

## ⚙️ Installation and Setup

### Option A — Kaggle (Recommended, GPU Required for Part 2)

Part 2 (CNN training) requires a GPU. Kaggle provides free GPU access (P100/T4).

1. Go to https://www.kaggle.com → create an account
2. Click **+ Create** → **New Notebook**
3. Upload `PCOS_Code_Part1.ipynb` or `PCOS_Code_Part2.ipynb`
4. Attach the relevant dataset under **Add Data** (right panel)
5. Under **Session options** → enable **GPU T4 x2** for Part 2
6. Click **Run All**

### Option B — Local Setup

```bash
# Step 1: Clone this repository
git clone https://github.com/FatimaNaqvi249/PCOS-PMOS-Dual-Modality-Framework.git
cd PCOS-PMOS-Dual-Modality-Framework

# Step 2: Create a virtual environment (recommended)
python -m venv pcos_env

# Step 3: Activate the environment
# On Windows:
pcos_env\Scripts\activate
# On Mac/Linux:
source pcos_env/bin/activate

# Step 4: Install all dependencies
pip install -r requirements.txt

# Step 5: Download datasets from Kaggle links above
#         and place them in the data/ folder as described

# Step 6: Launch Jupyter
jupyter notebook

# Step 7: Open and run Part 1
# notebooks/PCOS_Code_Part1.ipynb

# Step 8: Open and run Part 2 (GPU strongly recommended)
# notebooks/PCOS_Code_Part2.ipynb
```

---

## 🔧 Dependencies

All dependencies are listed in `requirements.txt`. Key libraries:

| Library | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Base language |
| PyTorch | 2.10+ | CNN training |
| timm | 1.0.25 | Pretrained models (EfficientNetV2, ConvNeXt, Swin) |
| scikit-learn | 1.3+ | Tabular models, metrics, SMOTE splitting |
| imbalanced-learn | 0.11+ | SMOTE oversampling |
| xgboost | 2.0+ | XGBoost classifier |
| shap | 0.44+ | SHAP feature attribution |
| opencv-python | 4.8+ | CLAHE image enhancement |
| pandas | 2.0+ | Data manipulation |
| numpy | 1.24+ | Numerical operations |
| matplotlib | 3.7+ | Visualization |
| seaborn | 0.12+ | Statistical plots |
| Pillow | 10.0+ | Image loading |

---

## 🚀 Reproducing Key Results

### Reproduce SHAP Analysis (Part 1, Section 8)

Run all cells in `PCOS_Code_Part1.ipynb` sequentially.
The SHAP bar chart and dependence plot will be saved as:
- `shap_01_bar.png`
- `shap_02_beeswarm.png`
- `shap_03_lh_fsh_dependence.png`

### Reproduce Age-Stratified Results (Part 1, Section 7.5)

Run all cells including Section 7.5.
Output table will print to console and save as `age_01_prevalence_by_group.png`

### Reproduce ConvNeXt-Small Training (Part 2, Section 8)

> Requires Kaggle GPU or equivalent CUDA device

Run all cells in `PCOS_Code_Part2.ipynb`.
Trained model saves to: `convnext_small_best.pth`

Expected output:
ConvNeXt-Small Test Results:

Accuracy:  0.9983

Precision: 0.9979

Recall:    1.0000

F1-Score:  0.9990

AUC-ROC:   1.0000

### Reproduce MD5 Duplicate Detection (Part 2, Section 2)
Total files scanned: 11784

infected    : 6784

noninfected : 5000
Total duplicate groups     : 1956

Cross-class (dangerous!) : 0

Intra-class              : 1956

Total duplicate files      : 7788
After deduplication:

Total clean files : 3996

infected          : 3184

noninfected       : 812

Files removed     : 7788

---

## 🖥️ Environment

All experiments were conducted on:

| Component | Specification |
|---|---|
| Platform | Kaggle Notebooks |
| GPU | NVIDIA P100 / T4 |
| CUDA | 12.8 |
| PyTorch | 2.10.0+cu128 |
| timm | 1.0.25 |
| OS | Linux (Ubuntu) |

---

## 📖 Citation

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{naqvi2026pcos,
  title     = {Endocrine Phenotype First: A Dual-Modality Machine Learning
               Framework for PCOS Diagnosis with SHAP-Based Interpretation
               and Age-Stratified Validation},
  author    = {Naqvi, Syeda Fatima and Sibtain, Sabien and Khan, Adil},
  booktitle = {IEEE Conference Proceedings},
  year      = {2026},
  address   = {Islamabad, Pakistan}
}
```

---

## 📚 Key References

1. WHO, "Polycystic ovary syndrome (PCOS) – Key facts," 2025. https://www.who.int/news-room/fact-sheets/detail/polycystic-ovary-syndrome

2. Rotterdam ESHRE/ASRM Consensus Group, "Revised 2003 consensus on diagnostic criteria," *Fertility and Sterility*, vol. 81, 2004.

3. H. J. Teede et al., "Recommendations from the 2023 international evidence-based guideline for PCOS," *Human Reproduction*, vol. 38, 2023.

4. H. J. Teede et al., "Polyendocrine metabolic ovarian syndrome, the new name for PCOS," *The Lancet*, 2026. doi:10.1016/S0140-6736(26)00717-8

5. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," *NeurIPS*, 2017.

6. A. Chattopadhyay et al., "Grad-CAM++: Generalized gradient-based visual explanations," *IEEE WACV*, 2018.

7. Z. Liu et al., "A ConvNet for the 2020s," *IEEE CVPR*, 2022.

8. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *ACM KDD*, 2016.

---

## 🤝 Acknowledgments

- **Dr. Adil Khan** — Supervisor, Department of Computer Science, Bahria University Islamabad
- **C. M. Divya** — PCOS Clinical Dataset (Kaggle, 2024)
- **ibadeus** — PCOS-XAI Ultrasound Dataset (Kaggle, 2023)
- **Kaggle** — Free GPU compute enabling this research

---

## 📜 License

This project is licensed under the **MIT License**.

You are free to use, copy, modify, and distribute this code for any purpose with attribution.

See the [LICENSE](LICENSE) file for full details.

---

## ⭐ If this work helped you, please star the repository!