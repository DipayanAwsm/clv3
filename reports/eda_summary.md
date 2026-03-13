# EDA Summary: Premium Customer Intelligence

## Dataset and Coverage
- Source: `clv_realistic_50000_5yr.csv`
- Rows: **50,000** policy-year records
- Customers: **20,248** unique customers
- Years: **2021-2025**
- Scope includes premium, earned premium, losses, claims, retention, delinquency, customer satisfaction, and customer/property risk attributes.

## Portfolio Health Snapshot
- Average earned premium per policy-year: **$2,561.88**
- Claim incidence rate: **22.16%**
- Average claim count: **0.298**
- Renewal rate: **75.03%**
- Delinquency rate: **17.68%**

## Relationship Validation (Business Logic Checks)
- Property value and coverage are strongly aligned.
- Coverage amount is positively associated with premium.
- Hazard score is positively associated with claim activity.
- Complaint burden is negatively associated with satisfaction.
- Payment delays align strongly with delinquency.
- Higher satisfaction aligns with higher renewal likelihood.

## Time Trends (2021-2025)
- Earned premium rises from **$23.3M** to **$28.1M**.
- Net losses remain materially high each year.
- Portfolio renewal improves from **71.99%** (2021) to **77.62%** (2025).
- Underwriting margin (`earned premium - net loss`) is negative across all years in this synthetic portfolio, reinforcing profitability segmentation needs.

## Customer-Level Aggregation and CLV
- CLV proxy at customer level:
  - `CustomerProfit = EarnedPremium - NetLoss - CommissionExpense - AdminExpense - Tax`
  - `CustomerCLV = sum(CustomerProfit over years)`
- Premium customer threshold: top **20%** of CLV.
- Segment distribution:
  - Premium: **4,050**
  - Core: **11,136**
  - Low / Loss-Making: **5,062**

## Key Strategic Insight
Portfolio economics are highly uneven. A minority of customers represents the positive value pool, while low/loss-making segments create disproportionate downside pressure. Value-based retention, growth, and repricing actions are required to improve portfolio quality.
