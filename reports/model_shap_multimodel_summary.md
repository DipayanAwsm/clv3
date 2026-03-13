# Multi-Model Explainability Summary

## Target Variable
- `premium_customer` (binary)
- Definition: 1 if customer CLV is in the top configured quantile (default top 20%), else 0.

## Features Used
- Total model input features used: **39**
- File: `artifacts/all_model_features_used.csv`

## Models Fitted
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost: not available in current environment
- LightGBM: not available in current environment

## Model Performance (Test Set)
See full table: `artifacts/model_comparison_all_models.csv`

- Best model by ROC AUC: **GradientBoosting**
- ROC AUC: **0.9997**
- Accuracy: **0.9957**
- Precision: **0.9891**
- Recall: **0.9891**
- F1: **0.9891**

## Explainability Outputs
Per-model importance exports:
- `artifacts/shap_by_model/logisticregression_shap_importance.csv`
- `artifacts/shap_by_model/randomforest_shap_importance.csv`
- `artifacts/shap_by_model/gradientboosting_shap_importance.csv`

Combined explainability table:
- `artifacts/model_shap_comparison_all_models.csv`

Environment note:
- SHAP package is not installed in this runtime, so explainability methods used were:
  - `feature_importance` (tree models)
  - `coefficient_abs` (logistic regression)
- Availability file: `artifacts/model_availability.json`
