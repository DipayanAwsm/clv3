# Premium Customer Intelligence
## Identifying and Growing the Most Valuable Insurance Customers

### Slide 1: Title
**Premium Customer Intelligence**  
Identifying and Growing the Most Valuable Insurance Customers

Presenter: Data Science & Analytics  
Portfolio: Personal Property Insurance  
Data Horizon: 2021-2025 Policy-Year Portfolio

### Slide 2: Executive Summary
- We converted 50,000 policy-year records into 20,248 customer-level value profiles.
- We built a premium-customer scoring engine that predicts high-value customers with strong performance (ROC AUC: **0.9997**).
- We operationalized this into action bands for business teams:
  - **Retain & Grow**
  - **Protect**
  - **Monitor / Re-price**
- Immediate impact: prioritize retention and growth spend on the customers with highest long-term value potential.

### Slide 3: Business Problem
- Portfolio profitability is uneven; customer value is highly concentrated.
- Broad, non-segmented retention strategies underinvest in top-value relationships and overinvest in low-return cohorts.
- Leadership needs a repeatable way to:
  - identify high-value customers early,
  - protect profitable segments,
  - and intervene on loss-making risk patterns.

### Slide 4: Data Foundation
- Source dataset: **50,000 policy-year rows**, **53 fields**, **2021-2025**.
- Coverage includes premium, claims, expenses, retention, delinquency, risk, and satisfaction signals.
- We transformed policy-year records to customer-level features and CLV proxy:
  - `CustomerProfit = EarnedPremium - NetLoss - CommissionExpense - AdminExpense - Tax`
  - `CustomerCLV = sum(CustomerProfit across years)`
- Premium target definition: top **20% CLV** customers (configurable threshold).

### Slide 5: EDA Highlights
- Renewal rate is healthy at **75.0%**, but claim incidence is material at **22.2%**.
- Net losses exceed earned premium across all years in this synthetic portfolio, signaling margin pressure.
- Stable business relationships observed:
  - Property value and coverage are strongly aligned.
  - Coverage level scales with premium.
  - Higher complaints align with lower satisfaction.
  - Payment delays track delinquency risk.

### Slide 6: Customer Value Segmentation
- Segment mix:
  - **Premium:** 4,050 customers (20.0%)
  - **Core:** 11,136 customers (55.0%)
  - **Low / Loss-Making:** 5,062 customers (25.0%)
- Value concentration:
  - Premium segment contributes positive CLV pool (~**$31.8M**)
  - Low/Loss segment drives substantial negative CLV burden (~**-$288.1M**)
- Strategic implication: growth and retention resources must be selectively allocated.

### Slide 7: Modeling Approach
- Built a modular supervised classification pipeline with stratified train/test split.
- Preprocessing:
  - numeric imputation + scaling,
  - categorical imputation + one-hot encoding.
- Model benchmarked:
  - Logistic Regression,
  - Random Forest,
  - Gradient Boosting.
- Optional XGBoost/LightGBM included in code when environment supports them.

### Slide 8: Model Performance
- Best model: **Gradient Boosting**.
- Test performance:
  - ROC AUC: **0.9997**
  - Accuracy: **99.57%**
  - Precision: **98.91%**
  - Recall: **98.91%**
  - F1: **98.91%**
- Confusion matrix summary:
  - TN: 4,039
  - FP: 11
  - FN: 11
  - TP: 1,001
- Business meaning: reliable prioritization engine for high-value retention targeting.

### Slide 9: Top Drivers of Premium Customers
- Strongest model drivers:
  - `loss_ratio`
  - `total_earned_premium`
  - `average_loss_ratio`
  - `total_commission`
  - `average_earned_premium`
- Behavioral and risk pattern interpretation:
  - High retained premium with controlled losses increases premium likelihood.
  - Elevated claim frequency and high loss burden reduce value probability.
  - Better renewal and satisfaction profiles are associated with premium segments.
- SHAP module is implemented; environment fallback used feature importance where SHAP package is unavailable.

### Slide 10: Strategic Actions
- **Retain & Grow** (high score):
  - proactive retention outreach,
  - bundled product offers,
  - loyalty incentives for long-tenure/high-satisfaction households.
- **Protect** (mid score):
  - targeted service interventions,
  - payment and communication nudges,
  - selective coverage optimization.
- **Monitor / Re-price** (low score):
  - tighter underwriting reviews,
  - pricing correction and deductible strategy,
  - risk mitigation and claims prevention programs.

### Slide 11: Operationalization
- Deploy customer scores into CRM and renewal workflows.
- Integrate score and action band into:
  - retention case queues,
  - underwriting review triggers,
  - portfolio steering dashboards.
- Governance:
  - quarterly model monitoring,
  - drift checks by state/channel,
  - KPI tracking on retention lift and CLV improvement.

### Slide 12: Recommendation and Next Steps
- Launch a **90-day pilot** in 2-3 states/channels.
- Run champion/challenger retention strategy using action bands.
- Measure uplift on:
  - renewal rate,
  - retained premium,
  - net loss ratio,
  - CLV improvement.
- Expand to pricing and claims intervention once lift is validated.

---

## PPT Export Notes (Optional)
- Use one slide per `Slide` section above.
- Visuals to embed from project artifacts:
  - `artifacts/model_comparison.csv` (performance table)
  - `artifacts/feature_importance.csv` (driver bar chart)
  - `artifacts/shap_summary.png` (explainability visual)
  - dashboard screenshots from Streamlit tabs.
- Keep verbal emphasis on profitable growth and targeted retention execution.

## Executive Summary Text (Optional)
We built a production-ready premium customer intelligence solution that converts policy-year insurance data into customer-level CLV analytics and predictive scoring. The model identifies high-value customers with strong discriminative performance and provides explainable value drivers for business teams. The output is operationalized into action bands to improve retention focus, cross-sell targeting, and risk-aware pricing decisions.
