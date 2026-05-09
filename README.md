# PANCAN-XAI Clinical Prediction 🔬🧬

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Optimization: Optuna](https://img.shields.io/badge/Optimization-Optuna-blue.svg)](https://optuna.org/)

**SOTA Explainable Machine Learning Pipeline for Multi-Omic Clinical Risk Prediction** with high-fidelity interpretations and automated scientific reporting.

## 🎯 Project Objectives

- **High-Dimensional Benchmarking**: Evaluate **18 state-of-the-art** machine learning models (Ensembles, Boostings, Neural Networks) for clinical risk stratification.
- **Precision Prediction**: Achieve >0.99 AUC-ROC on TCGA-BRCA cohorts through multi-modal fusion.
- **Total Transparency**: Provide dual-layer explainability using **SHAP (Game Theory)** and **LIME (Local Surrogates)**.
- **Scientific Validation**: Quantify model reliability through the **XAI Stability Index (Jaccard Index)** and robustness stress tests.
- **Clinical Actionability**: Translate complex genomic signals into interactive, interpretable decision-support dashboards.

## 🌟 Key Features (v2.5 SOTA)

- ✅ **18 ML Classifiers**: XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, MLP, SVM, KNN, AdaBoost, GBC, etc.
- ✅ **SOTA Voting Ensemble**: Hybrid soft-voting architecture combining the top-performing gradient boosting machines.
- ✅ **Multi-Omic Fusion Layer**: Integrated processing of Gene Expression (5000+ features), Clinical data, Mutations, and CNA.
- ✅ **Bayesian Optimization**: Deep hyperparameter tuning via **Optuna** (100+ trials per model).
- ✅ **Interactive Clinical Hub**: Premium Streamlit dashboard with:
    - **🛰️ Terminal Overview**: 3D Bio-Clustering (PCA) and real-time cohort monitoring.
    - **📉 Diagnostic Lab**: Live ROC & Precision-Recall curves.
    - **🧬 Bio-Insights**: Violin distributions and Interaction Matrices for dominant biomarkers.
    - **🔍 XAI Deep Dive**: Waterfall plots, Summary plots, and SHAP vs LIME consistency audits.
    - **🛡️ Robustness Audit**: Stability Index tracking and noise sensitivity stress tests.
    - **🏗️ Architecture Blueprint**: Visual structural overview of the multi-layer pipeline.
- ✅ **Automated Scientific Reporting**: End-to-end generation of LaTeX-formatted research reports and thesis sections.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/NidhalZitouni/xai-clinical-prediction.git
cd xai-clinical-prediction

# Setup environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Execution

**Run the Full Pipeline (Training + Optimization + XAI + Reporting):**
```bash
python -m xai_clinical.pipeline
```

**Launch the Interactive Intelligence Dashboard:**
```bash
streamlit run app/app.py
```

## 🏗️ Architecture Overview

The pipeline follows a hierarchical design:
1. **Input**: TCGA-BRCA Multimodal Data (Genomics + Clinical).
2. **Preprocessing**: 12-char alignment, Robust Scaling, SMOTE Balancing.
3. **Core Engine**: 18 Optimized Classifiers + SOTA Soft-Voting Ensemble.
4. **Interpretation**: Global/Local SHAP + LIME consistency checks.
5. **Output**: Risk Prediction + Clinical Biomarkers + Stability Metrics.

## 📊 Performance Benchmark

| Model | AUC-ROC | F1-Score | Stability (Jaccard) |
| :--- | :--- | :--- | :--- |
| **SOTA Ensemble** | **0.998** | **0.985** | **0.89** |
| XGBoost | 0.991 | 0.972 | 0.86 |
| CatBoost | 0.988 | 0.965 | 0.84 |
| MLP | 0.965 | 0.941 | 0.78 |

---
*Developed by Nidhal Zitouni as part of a research thesis in Clinical AI & Bioinformatics.*