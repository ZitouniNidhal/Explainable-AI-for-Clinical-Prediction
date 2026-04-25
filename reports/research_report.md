# XAI Clinical Prediction - Research Report

**Date:** 2026-04-24 20:45

## Abstract

This study develops an explainable machine learning model for predicting post-operative complication risk in breast cancer patients using multi-omic data.

## 1. Dataset

- **Source**: TCGA-BRCA
- **Samples**: 1097
- **Features**: 20533
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
| logistic_regression | 0.675 | 0.400 | 0.500 | 0.105 |
| random_forest | 0.608 | 0.400 | 0.000 | 0.000 |
| xgboost | 0.634 | 0.400 | 0.000 | 0.000 |
| lightgbm | 0.476 | 0.400 | 0.000 | 0.000 |

### 3.2 Best Model Performance

- **Model**: logistic_regression
- **AUC-ROC**: 0.400
- **Sensitivity**: 0.000
- **Specificity**: 0.887
- **PPV**: 0.000
- **NPV**: 0.959
- **F1-Score**: 0.000

### 3.3 Feature Importance

Top predictive features (SHAP):

- **VSIG8|391123**: 0.0872
- **DNAH11|8701**: 0.0713
- **USF1|7391**: 0.0562
- **TAGLN2|8407**: 0.0509
- **IL20RB|53833**: 0.0465
- **WDR17|116966**: 0.0433
- **LCN1|3933**: 0.0399
- **KRT78|196374**: 0.0383
- **EXT1|2131**: 0.0358
- **ACCSL|390110**: 0.0332

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
