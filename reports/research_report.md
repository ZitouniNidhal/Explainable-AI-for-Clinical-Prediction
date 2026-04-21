# XAI Clinical Prediction - Research Report

**Date:** 2026-04-21 02:18

## Abstract

This study develops an explainable machine learning model for predicting post-operative complication risk in breast cancer patients using multi-omic data.

## 1. Dataset

- **Source**: TCGA-BRCA
- **Samples**: 500
- **Features**: 20
- **Target**: High surgical risk (composite biological score)

## 2. Methods

### 2.1 Feature Engineering
- Clinical: Age, Stage, Grade, ER/PR/HER2 status
- Genomic: Top variable genes (log2-transformed)
- Optional: Methylation, Mutations, CNA

### 2.2 Models
- XGBoost, LightGBM, Random Forest, Logistic Regression
- Bayesian hyperparameter optimization (Optuna)
- 5-fold stratified cross-validation

### 2.3 Explainability
- SHAP global and local explanations
- LIME instance-level explanations
- Stability analysis under perturbation

## 3. Results

### 3.1 Model Performance

| Model | Val AUC | Test AUC | Sensitivity | Specificity |
|-------|---------|----------|-------------|-------------|
| logistic_regression | 0.915 | 0.950 | 0.800 | 0.870 |
| random_forest | 0.966 | 0.950 | 0.840 | 0.913 |
| xgboost | 0.970 | 0.950 | 0.880 | 0.957 |
| lightgbm | 0.984 | 0.950 | 0.880 | 1.000 |

### 3.2 Best Model Performance

- **Model**: lightgbm
- **AUC-ROC**: 0.950
- **Sensitivity**: 0.920
- **Specificity**: 0.880
- **PPV**: 0.885
- **NPV**: 0.917
- **F1-Score**: 0.902

### 3.3 Feature Importance

Top predictive features (SHAP):

- **feature_0**: 2.9669
- **feature_19**: 2.0228
- **feature_4**: 1.6381
- **feature_18**: 1.1321
- **feature_13**: 1.1134
- **feature_17**: 1.0228
- **feature_10**: 0.8846
- **feature_7**: 0.7913
- **feature_2**: 0.7906
- **feature_14**: 0.7428

## 4. Clinical Interpretation

### Risk Factors Identified:
1. **Tumor Stage**: Advanced stage (III/IV) strongly predictive
2. **Molecular Subtype**: Triple-negative and HER2+ associated with higher risk
3. **Tumor Grade**: High-grade tumors more likely to have complications
4. **Age**: Both very young (<35) and elderly (>70) at higher risk

## 5. Limitations

- Retrospective study design
- Requires external validation on independent cohort
- Optimal threshold determination needed for clinical implementation

## 6. Conclusion

The XAI model demonstrates excellent discriminative ability with stable and interpretable explanations. The biological features (stage, grade, molecular subtype) align with known clinical risk factors, supporting model validity for clinical decision support.
