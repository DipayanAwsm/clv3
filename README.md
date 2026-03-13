# Premium Customer Director Showcase

A complete insurance analytics project to identify, explain, and operationalize **Premium Customers** using policy-year data transformed into customer-level CLV intelligence.

## 1. Project Overview
This project answers a leadership-critical question:

**Which customers are most valuable and how can the business identify, retain, and grow them?**

The solution combines:
- portfolio and segment EDA,
- customer-level CLV engineering,
- premium-customer classification,
- explainable ML outputs,
- and an executive Streamlit dashboard.

## 2. Business Objective
Enable profitable growth by moving from broad retention decisions to **value-driven customer prioritization** using reproducible analytics and explainable scoring.

## 3. Dataset Description
Input dataset: `data/clv_realistic_50000_5yr.csv`

- Granularity: policy-year level
- Volume: 50,000 rows
- Coverage: customer, policy, property, premium, claims, expense, delinquency, retention, and satisfaction fields
- Time span: 2021-2025

## 4. Target Definition
Customer-level value proxy:

`CustomerProfit = EarnedPremium - NetLoss - CommissionExpense - AdminExpense - Tax`

`CustomerCLV = sum(CustomerProfit across years by customer)`

Binary target:

`PremiumCustomer = 1 if CustomerCLV >= quantile threshold`

Default threshold is top 20% (`premium_quantile=0.80`), configurable in `src/data_preparation.py`.

## 5. Project Structure
```text
premium_customer_director_showcase/
│
├── data/
│   ├── clv_realistic_50000_5yr.csv
│   ├── customer_level_dataset.csv
│   └── policy_year_enriched.csv
│
├── notebooks/
│   └── premium_customer_eda_model_shap.ipynb
│
├── src/
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── explain_model.py
│   └── utils.py
│
├── artifacts/
│   ├── premium_customer_model.joblib
│   ├── model_metrics.json
│   ├── model_comparison.csv
│   ├── customer_scores.csv
│   ├── feature_importance.csv
│   ├── shap_feature_importance.csv
│   └── shap_summary.png
│
├── dashboard/
│   └── app.py
│
├── presentation/
│   └── director_storyline.md
│
├── reports/
│   ├── eda_summary.md
│   └── business_recommendations.md
│
├── requirements.txt
└── README.md
```

## 6. Installation
From project root:

```bash
cd premium_customer_director_showcase
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 7. Run Data Preparation
```bash
python src/data_preparation.py --premium-quantile 0.80
```

Output:
- `data/customer_level_dataset.csv`
- `data/policy_year_enriched.csv`

## 8. Train Model and Generate Scores
```bash
python src/train_model.py
```

If your dashboard/runtime environment does not have optional packages like `xgboost` or `lightgbm`, generate a portable sklearn-only model:

```bash
python src/train_model.py --disable-optional-models --model-output-path artifacts/premium_customer_model_portable.joblib --metrics-path artifacts/model_metrics_portable.json --comparison-path artifacts/model_comparison_portable.csv
```

Outputs:
- `artifacts/premium_customer_model.joblib`
- `artifacts/model_metrics.json`
- `artifacts/model_comparison.csv`
- `artifacts/customer_scores.csv`
- `artifacts/feature_importance.csv`

## 9. Run Explainability
```bash
python src/explain_model.py
```

Outputs:
- `artifacts/shap_feature_importance.csv`
- `artifacts/shap_summary.png`

Note: If SHAP is not installed, script gracefully falls back to model feature importance and still generates the expected files.

## 9b. Run Multi-Model Benchmark + SHAP/Fallback by Model
```bash
python src/benchmark_models_shap.py
```

Outputs:
- `artifacts/model_comparison_all_models.csv`
- `artifacts/model_shap_comparison_all_models.csv`
- `artifacts/shap_by_model/*.csv`
- `artifacts/all_model_features_used.csv`
- `artifacts/model_availability.json`

## 10. Run the Notebook
```bash
jupyter notebook notebooks/premium_customer_eda_model_shap.ipynb
```

## 11. Launch the Dashboard
```bash
streamlit run dashboard/app.py
```

Dashboard tabs:
- Executive Overview
- Portfolio Trends
- Customer Value Segments
- Driver Analysis
- Customer Explorer
- Scenario / What-if

## 12. Artifact Descriptions
- `premium_customer_model.joblib`: best model bundle + preprocessing metadata
- `model_metrics.json`: best model metrics + full benchmark summary
- `model_comparison.csv`: side-by-side model performance
- `customer_scores.csv`: customer-level scores + action band
- `feature_importance.csv`: model feature importance
- `shap_feature_importance.csv`: SHAP/fallback driver ranking
- `shap_summary.png`: explainability summary visual
- `premium_customer_model_portable.joblib`: sklearn-only fallback model for environments without optional boosters

## 13. Presenting to Leadership
Recommended demo flow:
1. Open `presentation/director_storyline.md` to align narrative.
2. Show `reports/eda_summary.md` for data and portfolio context.
3. Present `artifacts/model_comparison.csv` and `artifacts/shap_summary.png` for model confidence and explainability.
4. Live-demo `dashboard/app.py` (focus on executive overview, driver analysis, and customer explorer/action bands).

## 14. Current Run Snapshot
Latest run in this workspace selected **GradientBoosting** as best model by ROC AUC.
