"""Data preparation script for premium customer analytics.

Transforms policy-year records to a customer-level modeling dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parent))
from utils import (
    ARTIFACTS_DIR,
    DATA_DIR,
    assign_value_segment,
    ensure_directories,
    mode_or_unknown,
    safe_divide,
)


NUMERIC_COLUMNS: list[str] = [
    "Year",
    "DIRECTWRITTENPREMIUM_AM",
    "EARNEDPREMIUM_AM",
    "NETLOSS_PAID_AM",
    "CLAIMCOUNT_CT",
    "COMMISSION_EXPENSE_AM",
    "ADMIN_EXPENSE_AM",
    "TAX_AM",
    "PaymentDelayDays",
    "CustomerSatisfaction",
    "ComplaintCount",
    "DelequencyFlag",
    "POLICY_RENEWED_FLAG",
    "MULTIPRODUCTDISCOUNT_FLAG",
    "HAZARD_SCORE",
    "PropertyValue",
    "CoverageAmount",
    "CreditScore",
    "Deductible",
    "AgentExperienceYears",
    "CustomerTenure",
]


CATEGORICAL_MODE_MAPPING: dict[str, str] = {
    "POLICYRATEDSTATE_TP": "dominant_state",
    "IncomeBracket": "dominant_income_bracket",
    "PaymentMethod": "dominant_payment_method",
    "MarketingChannel": "dominant_marketing_channel",
    "AGENT_CHANNEL": "dominant_agent_channel",
    "INSUREDITEM_TP": "dominant_item_type",
    "PROPERTYCOVERAGESUBTYPE_TP": "dominant_subtype",
    "CREDITMODEL_CD": "dominant_credit_band",
}


def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_policy_data(input_path: Path) -> pd.DataFrame:
    """Load and sanitize policy-year dataset."""
    df = pd.read_csv(input_path)

    if "POLICYEFFECTIVE_DT" in df.columns:
        df["POLICYEFFECTIVE_DT"] = pd.to_datetime(df["POLICYEFFECTIVE_DT"], errors="coerce")
    if "ACCOUNTING_MONTH" in df.columns:
        df["ACCOUNTING_MONTH"] = pd.to_datetime(df["ACCOUNTING_MONTH"], errors="coerce")

    df = _coerce_numeric_columns(df, NUMERIC_COLUMNS)

    # Standardize missing values in core categorical fields used for rollups.
    for column in CATEGORICAL_MODE_MAPPING:
        if column in df.columns:
            df[column] = df[column].fillna("Unknown")

    # Policy-year profit proxy used to derive customer CLV.
    df["customer_profit"] = (
        df["EARNEDPREMIUM_AM"].fillna(0)
        - df["NETLOSS_PAID_AM"].fillna(0)
        - df["COMMISSION_EXPENSE_AM"].fillna(0)
        - df["ADMIN_EXPENSE_AM"].fillna(0)
        - df["TAX_AM"].fillna(0)
    )

    # Convenience metric used in EDA/dashboard trend plots.
    df["policy_margin"] = df["EARNEDPREMIUM_AM"].fillna(0) - df["NETLOSS_PAID_AM"].fillna(0)
    return df


def aggregate_customer_level(df: pd.DataFrame, premium_quantile: float = 0.80) -> pd.DataFrame:
    """Aggregate policy-year data into customer-level features and target."""
    if "CustomerID" not in df.columns:
        raise ValueError("Input dataset must include CustomerID column.")

    grouped = df.groupby("CustomerID", dropna=False)

    customer_df = grouped.agg(
        policy_year_records=("CustomerID", "size"),
        total_earned_premium=("EARNEDPREMIUM_AM", "sum"),
        total_written_premium=("DIRECTWRITTENPREMIUM_AM", "sum"),
        total_net_loss=("NETLOSS_PAID_AM", "sum"),
        total_claim_count=("CLAIMCOUNT_CT", "sum"),
        total_commission=("COMMISSION_EXPENSE_AM", "sum"),
        total_admin_expense=("ADMIN_EXPENSE_AM", "sum"),
        total_tax=("TAX_AM", "sum"),
        average_premium=("DIRECTWRITTENPREMIUM_AM", "mean"),
        average_earned_premium=("EARNEDPREMIUM_AM", "mean"),
        average_payment_delay=("PaymentDelayDays", "mean"),
        average_customer_satisfaction=("CustomerSatisfaction", "mean"),
        total_complaints=("ComplaintCount", "sum"),
        delinquency_rate=("DelequencyFlag", "mean"),
        renewal_rate=("POLICY_RENEWED_FLAG", "mean"),
        multi_product_rate=("MULTIPRODUCTDISCOUNT_FLAG", "mean"),
        average_hazard_score=("HAZARD_SCORE", "mean"),
        average_property_value=("PropertyValue", "mean"),
        average_coverage_amount=("CoverageAmount", "mean"),
        average_credit_score=("CreditScore", "mean"),
        average_deductible=("Deductible", "mean"),
        average_agent_experience=("AgentExperienceYears", "mean"),
        years_active=("Year", pd.Series.nunique),
        max_tenure=("CustomerTenure", "max"),
        customer_clv=("customer_profit", "sum"),
    )

    # Categorical mode rollups.
    for raw_col, new_col in CATEGORICAL_MODE_MAPPING.items():
        customer_df[new_col] = grouped[raw_col].agg(mode_or_unknown)

    customer_df = customer_df.reset_index()

    # Ratio and derived features.
    customer_df["average_loss_ratio"] = customer_df.apply(
        lambda row: safe_divide(row["total_net_loss"], row["total_earned_premium"]), axis=1
    )
    customer_df["average_claim_severity"] = customer_df.apply(
        lambda row: safe_divide(row["total_net_loss"], row["total_claim_count"]), axis=1
    )
    customer_df["claim_frequency"] = customer_df.apply(
        lambda row: safe_divide(row["total_claim_count"], row["years_active"]), axis=1
    )
    customer_df["premium_to_property_ratio"] = customer_df.apply(
        lambda row: safe_divide(row["average_premium"], row["average_property_value"]), axis=1
    )
    customer_df["coverage_to_property_ratio"] = customer_df.apply(
        lambda row: safe_divide(row["average_coverage_amount"], row["average_property_value"]), axis=1
    )
    customer_df["loss_ratio"] = customer_df.apply(
        lambda row: safe_divide(row["total_net_loss"], row["total_earned_premium"]), axis=1
    )
    customer_df["claim_frequency_per_year"] = customer_df.apply(
        lambda row: safe_divide(row["total_claim_count"], row["years_active"]), axis=1
    )

    # Target definition (configurable premium threshold).
    premium_threshold = customer_df["customer_clv"].quantile(premium_quantile)
    low_threshold = customer_df["customer_clv"].quantile(0.25)

    customer_df["premium_threshold"] = premium_threshold
    customer_df["premium_customer"] = (customer_df["customer_clv"] >= premium_threshold).astype(int)
    customer_df["value_segment"] = customer_df["customer_clv"].apply(
        lambda value: assign_value_segment(value, premium_threshold, low_threshold)
    )

    # Sort for convenient exploration.
    customer_df = customer_df.sort_values("customer_clv", ascending=False).reset_index(drop=True)
    return customer_df


def save_prepared_data(
    policy_df: pd.DataFrame,
    customer_df: pd.DataFrame,
    output_customer_path: Path,
    output_policy_enriched_path: Path | None = None,
) -> None:
    output_customer_path.parent.mkdir(parents=True, exist_ok=True)
    customer_df.to_csv(output_customer_path, index=False)

    if output_policy_enriched_path is not None:
        output_policy_enriched_path.parent.mkdir(parents=True, exist_ok=True)
        policy_df.to_csv(output_policy_enriched_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare customer-level CLV dataset.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DATA_DIR / "clv_realistic_50000_5yr.csv",
        help="Input policy-year CSV path.",
    )
    parser.add_argument(
        "--output-customer-path",
        type=Path,
        default=DATA_DIR / "customer_level_dataset.csv",
        help="Output customer-level dataset path.",
    )
    parser.add_argument(
        "--output-policy-enriched-path",
        type=Path,
        default=DATA_DIR / "policy_year_enriched.csv",
        help="Output path for policy-year data with derived profit/margin fields.",
    )
    parser.add_argument(
        "--premium-quantile",
        type=float,
        default=0.80,
        help="Quantile threshold for premium customer label (0.80 = top 20%).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    if not 0.5 <= args.premium_quantile < 1.0:
        raise ValueError("premium_quantile must be between 0.5 (inclusive) and 1.0 (exclusive).")

    policy_df = load_policy_data(args.input_path)
    customer_df = aggregate_customer_level(policy_df, premium_quantile=args.premium_quantile)
    save_prepared_data(
        policy_df=policy_df,
        customer_df=customer_df,
        output_customer_path=args.output_customer_path,
        output_policy_enriched_path=args.output_policy_enriched_path,
    )

    premium_share = customer_df["premium_customer"].mean()
    print(f"Prepared customer dataset: {args.output_customer_path}")
    print(f"Customers: {len(customer_df):,}")
    print(f"Premium customer share: {premium_share:.2%}")
    print(f"Artifacts directory: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
