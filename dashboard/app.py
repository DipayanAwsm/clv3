"""Director-ready Streamlit dashboard for premium customer intelligence."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

st.set_page_config(
    page_title="Premium Customer Intelligence",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {background: linear-gradient(180deg, #f8fbff 0%, #ffffff 35%);} 
    h1, h2, h3 {color: #0a3d62;}
    [data-testid="stMetricValue"] {color: #0a3d62;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy_path = DATA_DIR / "policy_year_enriched.csv"
    if not policy_path.exists():
        policy_path = DATA_DIR / "clv_realistic_50000_5yr.csv"

    customer_path = DATA_DIR / "customer_level_dataset.csv"
    scores_path = ARTIFACTS_DIR / "customer_scores.csv"

    policy_df = pd.read_csv(policy_path)
    customer_df = pd.read_csv(customer_path)
    scores_df = pd.read_csv(scores_path)
    return policy_df, customer_df, scores_df


@st.cache_data
def load_driver_table() -> pd.DataFrame:
    shap_path = ARTIFACTS_DIR / "shap_feature_importance.csv"
    fallback_path = ARTIFACTS_DIR / "feature_importance.csv"

    if shap_path.exists():
        df = pd.read_csv(shap_path)
        value_col = "mean_abs_shap" if "mean_abs_shap" in df.columns else df.columns[1]
        df = df.rename(columns={value_col: "importance"})
    else:
        df = pd.read_csv(fallback_path)
        value_col = "importance" if "importance" in df.columns else df.columns[1]
        df = df.rename(columns={value_col: "importance"})

    return df.sort_values("importance", ascending=False)


@st.cache_resource
def load_model_bundle():
    model_path = ARTIFACTS_DIR / "premium_customer_model.joblib"
    portable_path = ARTIFACTS_DIR / "premium_customer_model_portable.joblib"
    try:
        model_bundle = joblib.load(model_path)
        return model_bundle, None
    except ModuleNotFoundError as exc:
        missing_pkg = exc.name or "optional package"
        if portable_path.exists():
            model_bundle = joblib.load(portable_path)
            message = (
                f"Primary model depends on '{missing_pkg}'. "
                f"Loaded portable fallback model '{model_bundle.get('model_name', 'Unknown')}'."
            )
            return model_bundle, message
        raise RuntimeError(
            "Unable to load model artifact because an optional dependency is missing: "
            f"'{missing_pkg}'. Re-run training with "
            "'python src/train_model.py --disable-optional-models --model-output-path "
            "artifacts/premium_customer_model_portable.joblib'."
        ) from exc


def metric_card_columns(policy_df: pd.DataFrame, customer_df: pd.DataFrame) -> None:
    total_policies = len(policy_df)
    total_customers = customer_df["CustomerID"].nunique()
    premium_share = customer_df["premium_customer"].mean()
    avg_clv = customer_df["customer_clv"].mean()
    avg_renewal = policy_df["POLICY_RENEWED_FLAG"].mean()
    avg_claim_rate = (policy_df["CLAIMCOUNT_CT"] > 0).mean()

    cols = st.columns(6)
    cols[0].metric("Policy-Year Rows", f"{total_policies:,}")
    cols[1].metric("Customers", f"{total_customers:,}")
    cols[2].metric("Premium Customer Share", f"{premium_share:.1%}")
    cols[3].metric("Average CLV", f"${avg_clv:,.0f}")
    cols[4].metric("Average Renewal Rate", f"{avg_renewal:.1%}")
    cols[5].metric("Claim Incidence", f"{avg_claim_rate:.1%}")


def portfolio_trend_frame(policy_df: pd.DataFrame) -> pd.DataFrame:
    trend_df = (
        policy_df.groupby("Year", as_index=False)
        .agg(
            earned_premium=("EARNEDPREMIUM_AM", "sum"),
            net_loss=("NETLOSS_PAID_AM", "sum"),
            claims=("CLAIMCOUNT_CT", "sum"),
            renewal_rate=("POLICY_RENEWED_FLAG", "mean"),
        )
        .sort_values("Year")
    )
    trend_df["margin"] = trend_df["earned_premium"] - trend_df["net_loss"]
    return trend_df


def render_executive_overview(policy_df: pd.DataFrame, customer_df: pd.DataFrame, scores_df: pd.DataFrame) -> None:
    st.subheader("Executive Overview")
    metric_card_columns(policy_df, customer_df)

    left, right = st.columns([1.4, 1])
    with left:
        action_mix = scores_df["action_band"].value_counts().reset_index()
        action_mix.columns = ["action_band", "customers"]
        fig = px.bar(
            action_mix,
            x="action_band",
            y="customers",
            color="action_band",
            color_discrete_sequence=["#0a3d62", "#2e86de", "#e58e26"],
            title="Recommended Action Mix",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        seg_share = customer_df["value_segment"].value_counts(normalize=True).reset_index()
        seg_share.columns = ["segment", "share"]
        fig = px.pie(
            seg_share,
            names="segment",
            values="share",
            hole=0.45,
            color="segment",
            color_discrete_map={
                "Premium": "#0a3d62",
                "Core": "#74b9ff",
                "Low / Loss-Making": "#e17055",
            },
            title="Customer Value Segmentation",
        )
        st.plotly_chart(fig, use_container_width=True)



def render_portfolio_trends(policy_df: pd.DataFrame) -> None:
    st.subheader("Portfolio Trends")
    trend_df = portfolio_trend_frame(policy_df)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(
            trend_df,
            x="Year",
            y=["earned_premium", "net_loss"],
            markers=True,
            title="Yearly Earned Premium vs Net Loss",
            color_discrete_sequence=["#0a3d62", "#e17055"],
        )
        fig.update_layout(yaxis_title="Amount ($)", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            trend_df,
            x="Year",
            y="claims",
            title="Yearly Claim Count",
            color_discrete_sequence=["#2e86de"],
        )
        fig.update_layout(yaxis_title="Claim Count")
        st.plotly_chart(fig, use_container_width=True)

    fig_margin = px.area(
        trend_df,
        x="Year",
        y="margin",
        title="Yearly Underwriting Margin (Earned Premium - Net Loss)",
        color_discrete_sequence=["#16a085"],
    )
    fig_margin.update_layout(yaxis_title="Margin ($)")
    st.plotly_chart(fig_margin, use_container_width=True)



def render_segments(customer_df: pd.DataFrame) -> None:
    st.subheader("Customer Value Segments")

    segment_kpi = (
        customer_df.groupby("value_segment", as_index=False)
        .agg(customers=("CustomerID", "nunique"), avg_clv=("customer_clv", "mean"), total_clv=("customer_clv", "sum"))
        .sort_values("total_clv", ascending=False)
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            segment_kpi,
            x="value_segment",
            y="total_clv",
            color="value_segment",
            title="Total CLV Contribution by Segment",
            color_discrete_map={
                "Premium": "#0a3d62",
                "Core": "#74b9ff",
                "Low / Loss-Making": "#e17055",
            },
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Total CLV ($)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            customer_df,
            x="customer_clv",
            nbins=60,
            color="value_segment",
            opacity=0.7,
            title="CLV Distribution by Value Segment",
            color_discrete_map={
                "Premium": "#0a3d62",
                "Core": "#74b9ff",
                "Low / Loss-Making": "#e17055",
            },
        )
        fig.update_layout(xaxis_title="Customer CLV ($)", yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)

    state_rates = (
        customer_df.groupby("dominant_state", as_index=False)
        .agg(premium_rate=("premium_customer", "mean"), customers=("CustomerID", "nunique"))
        .query("customers >= 100")
        .sort_values("premium_rate", ascending=False)
        .head(15)
    )

    fig_state = px.bar(
        state_rates,
        x="dominant_state",
        y="premium_rate",
        color="premium_rate",
        color_continuous_scale="Blues",
        title="Top States by Premium Customer Rate (Min 100 Customers)",
    )
    fig_state.update_layout(yaxis_tickformat=".0%", xaxis_title="State", yaxis_title="Premium Rate")
    st.plotly_chart(fig_state, use_container_width=True)



def render_drivers(customer_df: pd.DataFrame) -> None:
    st.subheader("Driver Analysis")

    driver_df = load_driver_table().head(20)
    fig = px.bar(
        driver_df.iloc[::-1],
        x="importance",
        y="feature",
        orientation="h",
        title="Top Predictive Drivers of Premium Customers",
        color_discrete_sequence=["#0a3d62"],
    )
    fig.update_layout(yaxis_title="", xaxis_title="Driver Importance")
    st.plotly_chart(fig, use_container_width=True)

    shap_img = ARTIFACTS_DIR / "shap_summary.png"
    if shap_img.exists():
        st.image(str(shap_img), caption="SHAP Summary: Feature Impact on Premium-Customer Probability")

    top_drivers = ", ".join(driver_df["feature"].head(5).tolist())
    numeric_corr = (
        customer_df.select_dtypes(include=["number"])
        .corr(numeric_only=True)
        .get("premium_customer", pd.Series(dtype=float))
        .drop(labels=["premium_customer"], errors="ignore")
        .sort_values(ascending=False)
    )
    pos_signal = numeric_corr.head(3).index.tolist()
    neg_signal = numeric_corr.tail(3).index.tolist()

    st.markdown(
        f"""
        **Business Interpretation**

        - Highest-impact predictors in the model: `{top_drivers}`.
        - Positive signals associated with premium customers: `{', '.join(pos_signal)}`.
        - Negative signals associated with premium customers: `{', '.join(neg_signal)}`.
        - Practical takeaway: prioritize high-value, high-retention customers for proactive retention and cross-sell while actively monitoring late-payment and high-claims profiles.
        """
    )



def render_customer_explorer(policy_df: pd.DataFrame, customer_df: pd.DataFrame, scores_df: pd.DataFrame) -> None:
    st.subheader("Customer Explorer")

    explorer_df = customer_df.merge(
        scores_df[["CustomerID", "premium_score", "action_band"]],
        on="CustomerID",
        how="left",
    )

    year_options = sorted(policy_df["Year"].dropna().unique().tolist())
    selected_years = st.multiselect("Policy Year", options=year_options, default=year_options)

    state_options = sorted(explorer_df["dominant_state"].dropna().unique().tolist())
    selected_states = st.multiselect("State", options=state_options, default=state_options[:8])

    segment_options = ["Premium", "Core", "Low / Loss-Making"]
    selected_segments = st.multiselect("Value Segment", options=segment_options, default=segment_options)

    channel_options = sorted(explorer_df["dominant_marketing_channel"].dropna().unique().tolist())
    selected_channels = st.multiselect(
        "Marketing Channel",
        options=channel_options,
        default=channel_options,
    )

    eligible_customers = set(
        policy_df.loc[policy_df["Year"].isin(selected_years), "CustomerID"].astype(str).unique().tolist()
    )

    filtered = explorer_df[
        explorer_df["CustomerID"].astype(str).isin(eligible_customers)
        & explorer_df["dominant_state"].isin(selected_states)
        & explorer_df["value_segment"].isin(selected_segments)
        & explorer_df["dominant_marketing_channel"].isin(selected_channels)
    ].copy()

    filtered = filtered.sort_values("premium_score", ascending=False)

    st.caption(f"Filtered customers: {len(filtered):,}")
    st.dataframe(
        filtered[
            [
                "CustomerID",
                "dominant_state",
                "dominant_marketing_channel",
                "value_segment",
                "customer_clv",
                "premium_score",
                "action_band",
                "renewal_rate",
                "delinquency_rate",
                "average_customer_satisfaction",
                "claim_frequency",
            ]
        ].head(500),
        use_container_width=True,
        height=460,
    )

    st.download_button(
        "Download Filtered Customers",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_customer_scores.csv",
        mime="text/csv",
    )



def render_what_if(customer_df: pd.DataFrame) -> None:
    st.subheader("Scenario / What-if")

    model_bundle, load_message = load_model_bundle()
    if load_message:
        st.info(load_message)
    pipeline = model_bundle["pipeline"]
    feature_columns = model_bundle["feature_columns"]

    base_row = {}
    for col in feature_columns:
        if col not in customer_df.columns:
            base_row[col] = 0
            continue
        if pd.api.types.is_numeric_dtype(customer_df[col]):
            base_row[col] = float(customer_df[col].median())
        else:
            mode_vals = customer_df[col].mode()
            base_row[col] = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"

    col1, col2, col3 = st.columns(3)
    avg_property_value = col1.slider("Avg Property Value", 50000, 2000000, int(base_row.get("average_property_value", 400000)), 5000)
    avg_premium = col2.slider("Avg Written Premium", 500, 12000, int(base_row.get("average_premium", 3000)), 50)
    avg_satisfaction = col3.slider("Avg Satisfaction", 1.0, 5.0, float(base_row.get("average_customer_satisfaction", 3.5)), 0.1)

    col4, col5, col6 = st.columns(3)
    claim_frequency = col4.slider("Claim Frequency", 0.0, 3.0, float(base_row.get("claim_frequency", 0.3)), 0.01)
    avg_payment_delay = col5.slider("Avg Payment Delay (Days)", 0.0, 60.0, float(base_row.get("average_payment_delay", 5.0)), 0.5)
    renewal_rate = col6.slider("Renewal Rate", 0.0, 1.0, float(base_row.get("renewal_rate", 0.7)), 0.01)

    scenario = base_row.copy()
    scenario["average_property_value"] = avg_property_value
    scenario["average_premium"] = avg_premium
    scenario["average_customer_satisfaction"] = avg_satisfaction
    scenario["claim_frequency"] = claim_frequency
    scenario["claim_frequency_per_year"] = claim_frequency
    scenario["average_payment_delay"] = avg_payment_delay
    scenario["renewal_rate"] = renewal_rate

    if "years_active" in scenario:
        years_active = max(float(scenario["years_active"]), 1.0)
        scenario["total_claim_count"] = claim_frequency * years_active
    if "average_coverage_amount" in scenario:
        scenario["average_coverage_amount"] = avg_property_value * 0.8
    if "premium_to_property_ratio" in scenario:
        scenario["premium_to_property_ratio"] = avg_premium / max(avg_property_value, 1)
    if "coverage_to_property_ratio" in scenario:
        scenario["coverage_to_property_ratio"] = scenario.get("average_coverage_amount", avg_property_value * 0.8) / max(
            avg_property_value, 1
        )

    scenario_df = pd.DataFrame([scenario])[feature_columns]

    probability = pipeline.predict_proba(scenario_df)[0, 1]

    band = "Retain & Grow" if probability >= 0.75 else "Protect" if probability >= 0.45 else "Monitor / Re-price"

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Predicted Premium-Customer Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0a3d62"},
                "steps": [
                    {"range": [0, 45], "color": "#f5b7b1"},
                    {"range": [45, 75], "color": "#fdebd0"},
                    {"range": [75, 100], "color": "#d5f5e3"},
                ],
            },
        )
    )
    gauge.update_layout(height=300)

    st.plotly_chart(gauge, use_container_width=True)
    st.markdown(f"**Recommended Action Band:** `{band}`")



def main() -> None:
    st.title("Premium Customer Intelligence")
    st.caption("Identify, retain, and grow the most valuable insurance customers.")

    try:
        policy_df, customer_df, scores_df = load_data()
    except Exception as exc:
        st.error(
            "Required artifacts are missing. Run data preparation and model training first. "
            f"Details: {exc}"
        )
        st.stop()

    tabs = st.tabs(
        [
            "Executive Overview",
            "Portfolio Trends",
            "Customer Value Segments",
            "Driver Analysis",
            "Customer Explorer",
            "Scenario / What-if",
        ]
    )

    with tabs[0]:
        render_executive_overview(policy_df, customer_df, scores_df)
    with tabs[1]:
        render_portfolio_trends(policy_df)
    with tabs[2]:
        render_segments(customer_df)
    with tabs[3]:
        render_drivers(customer_df)
    with tabs[4]:
        render_customer_explorer(policy_df, customer_df, scores_df)
    with tabs[5]:
        render_what_if(customer_df)


if __name__ == "__main__":
    main()
