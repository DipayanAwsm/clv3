"""Generate SHAP explainability outputs for the premium customer model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parent))
from feature_engineering import load_customer_dataset
from utils import ARTIFACTS_DIR, DATA_DIR, ensure_directories


def _to_dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _fallback_importance(feature_importance_path: Path, shap_importance_path: Path) -> pd.DataFrame:
    fallback_df = pd.read_csv(feature_importance_path)
    fallback_df = fallback_df.rename(columns={"importance": "mean_abs_shap"})
    fallback_df.to_csv(shap_importance_path, index=False)
    return fallback_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain premium customer model with SHAP.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ARTIFACTS_DIR / "premium_customer_model.joblib",
        help="Trained model bundle path.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DATA_DIR / "customer_level_dataset.csv",
        help="Customer-level dataset path.",
    )
    parser.add_argument(
        "--feature-importance-path",
        type=Path,
        default=ARTIFACTS_DIR / "feature_importance.csv",
        help="Fallback feature importance file.",
    )
    parser.add_argument(
        "--shap-importance-path",
        type=Path,
        default=ARTIFACTS_DIR / "shap_feature_importance.csv",
        help="Output SHAP feature importance CSV path.",
    )
    parser.add_argument(
        "--shap-summary-path",
        type=Path,
        default=ARTIFACTS_DIR / "shap_summary.png",
        help="Output SHAP summary plot path.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Maximum number of rows to sample for SHAP plotting.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    model_bundle = joblib.load(args.model_path)
    pipeline = model_bundle["pipeline"]
    feature_columns = model_bundle["feature_columns"]

    customer_df = load_customer_dataset(args.input_path)
    X = customer_df[feature_columns].copy()

    rng = np.random.default_rng(args.random_state)
    n_sample = min(args.sample_size, len(X))
    sample_indices = rng.choice(len(X), size=n_sample, replace=False)
    X_sample = X.iloc[sample_indices]

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    transformed = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()
    transformed_dense = _to_dense(transformed)
    transformed_df = pd.DataFrame(transformed_dense, columns=feature_names)

    try:
        import shap

        explainer = shap.Explainer(model, transformed_df)
        shap_values = explainer(transformed_df)

        # For some model types SHAP returns (n, f, class) tensor; use positive-class contributions.
        values = shap_values.values
        if values.ndim == 3:
            values = values[:, :, 1]

        mean_abs_shap = np.abs(values).mean(axis=0)
        shap_importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap,
            }
        ).sort_values("mean_abs_shap", ascending=False)

        # Aggregate encoded categorical levels to their parent feature for business readability.
        def base_feature(name: str) -> str:
            name = name.replace("num__", "").replace("cat__", "")
            if "_" not in name:
                return name
            for raw_feature in model_bundle.get("categorical_columns", []):
                if name.startswith(f"{raw_feature}_"):
                    return raw_feature
            return name

        shap_importance_grouped = (
            shap_importance_df.assign(base_feature=shap_importance_df["feature"].map(base_feature))
            .groupby("base_feature", as_index=False)["mean_abs_shap"]
            .sum()
            .rename(columns={"base_feature": "feature"})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        shap_importance_grouped.to_csv(args.shap_importance_path, index=False)

        # Save SHAP summary (beeswarm) plot.
        plt.figure(figsize=(12, 8))
        shap.summary_plot(values, transformed_df, show=False, max_display=20)
        plt.tight_layout()
        args.shap_summary_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.shap_summary_path, dpi=220, bbox_inches="tight")
        plt.close()

        print(f"SHAP outputs saved: {args.shap_importance_path}, {args.shap_summary_path}")
    except Exception as exc:
        print(f"SHAP generation failed ({exc}). Using fallback feature importance.")
        fallback_df = _fallback_importance(args.feature_importance_path, args.shap_importance_path)

        plt.figure(figsize=(11, 6))
        top_df = fallback_df.head(20).iloc[::-1]
        plt.barh(top_df["feature"], top_df["mean_abs_shap"], color="#0068c9")
        plt.title("Feature Importance (Fallback: SHAP unavailable)")
        plt.xlabel("Importance")
        plt.tight_layout()
        args.shap_summary_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.shap_summary_path, dpi=220, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
