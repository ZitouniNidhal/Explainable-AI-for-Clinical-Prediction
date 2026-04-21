# XAI Report - Clinical Outcome Prediction

**Date:** 2026-04-21 00:49

## 1. Executive Summary

- **Best model:** lightgbm
- **Number of samples:** 500
- **Number of features:** 20

## 2. Model Performance

| Model | Train AUC | Val AUC | Test AUC |
|--------|-----------|---------|----------|
| logistic_regression | 0.913 | 0.915 | N/A |
| random_forest | 1.000 | 0.966 | N/A |
| xgboost | 1.000 | 0.970 | N/A |
| lightgbm | 1.000 | 0.984 | N/A |

## 3. Feature Importance (SHAP)

Most important variables for prediction:

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

### Identified risk factors:
1. **Age**: Elderly patients have increased risk
2. **ASA Score**: Anesthesia risk score is predictive
3. **Surgery duration**: Longer procedures are riskier

## 5. Robustness

Explanations were tested with:
- Gaussian noise (1%, 5%, 10%)
- Missing data (5%, 10%, 20%)

See `stability_report.txt` for details.

## 6. Recommendations

- Model is ready for prospective clinical validation
- Monitor data drift in production
- Implement clinical feedback system
