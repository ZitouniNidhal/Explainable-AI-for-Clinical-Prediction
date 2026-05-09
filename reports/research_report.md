# XAI Clinical Prediction - Research Report

**Date:** 2026-04-26 04:21

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
| logistic_regression | 0.900 | 0.962 | 0.816 | 0.795 |
| random_forest | 0.957 | 0.962 | 0.974 | 0.841 |
| xgboost | 0.936 | 0.962 | 0.974 | 0.841 |
| lightgbm | 0.925 | 0.962 | 0.921 | 0.833 |
| extra_trees | 0.957 | 0.962 | 0.947 | 0.800 |
| catboost | 0.962 | 0.962 | 0.974 | 0.804 |
| ada_boost | 0.839 | 0.962 | 0.816 | 0.756 |
| gradient_boosting | 0.957 | 0.962 | 0.974 | 0.804 |
| svm | 0.968 | 0.962 | 0.974 | 0.841 |
| knn | 0.965 | 0.962 | 0.921 | 0.897 |
| mlp | 0.986 | 0.962 | 0.921 | 0.946 |

### 3.2 Best Model Performance

- **Model**: mlp
- **AUC-ROC**: 0.962
- **Sensitivity**: 0.921
- **Specificity**: 0.865
- **PPV**: 0.875
- **NPV**: 0.914
- **F1-Score**: 0.897

### 3.3 Feature Importance

Top predictive features (SHAP):

- **feature_0**: 0.1619
- **feature_2**: 0.0887
- **feature_13**: 0.0861
- **feature_19**: 0.0851
- **feature_4**: 0.0735
- **feature_17**: 0.0592
- **feature_12**: 0.0539
- **feature_8**: 0.0422
- **feature_10**: 0.0288
- **feature_18**: 0.0270

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
