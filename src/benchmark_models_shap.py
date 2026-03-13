"""Benchmark multiple classifiers and generate per-model SHAP/fallback importance.

This script trains supported models on all engineered customer-level features
and exports comparable explainability outputs per model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import sys

sys.path.append(str(Path(__file__).resolve().parent))
from feature_engineering import build_feature_set, build_preprocessor, get_preprocessed_feature_names, load_customer_dataset
from utils import ARTIFACTS_DIR, DATA_DIR, ensure_directories, save_json


def build_models(random_state: int = 42) -> tuple[dict[str, Any], dict[str, str]]:
    """Return available model objects and availability status for optional libraries."""
    models: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
    }

    availability = {
        "XGBoost": "not_available",
        "LightGBM": "not_available",
        "SHAP": "not_available",
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=random_state,
            n_jobs=-1,
        )
        availability["XGBoost"] = "available"
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            objective="binary",
            random_state=random_state,
            n_jobs=-1,
        )
        availability["LightGBM"] = "available"
    except Exception:
        pass

    try:
        import shap  # noqa: F401

        availability["SHAP"] = "available"
    except Exception:
        pass

    return models, availability


def probability_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)
        if probabilities.ndim == 2:
            return probabilities[:, 1]
        return probabilities

    if hasattr(pipeline, "decision_function"):
        decision_values = pipeline.decision_function(X)
        min_val = np.min(decision_values)
        max_val = np.max(decision_values)
        if max_val == min_val:
            return np.zeros_like(decision_values)
        return (decision_values - min_val) / (max_val - min_val)

    return pipeline.predict(X).astype(float)


def evaluate(y_true: pd.Series, y_score: np.ndarray) -> dict[str, Any]:
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def aggregate_encoded_importance(
    feature_scores: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Roll one-hot level scores up to base feature names."""
    categorical_prefixes = tuple(f"{col}_" for col in categorical_columns)

    def to_base(feature: str) -> str:
        cleaned = feature.replace("num__", "").replace("cat__", "")
        if cleaned in numeric_columns:
            return cleaned
        for prefix, base_col in zip(categorical_prefixes, categorical_columns):
            if cleaned.startswith(prefix):
                return base_col
        return cleaned

    grouped = (
        feature_scores.assign(base_feature=feature_scores["feature"].map(to_base))
        .groupby("base_feature", as_index=False)["importance_score"]
        .sum()
        .rename(columns={"base_feature": "feature"})
        .sort_values("importance_score", ascending=False)
        .reset_index(drop=True)
    )
    grouped["importance_rank"] = np.arange(1, len(grouped) + 1)
    return grouped


def shap_or_fallback_importance(
    model_name: str,
    pipeline: Pipeline,
    X_reference: pd.DataFrame,
    y_reference: pd.Series,
    numeric_columns: list[str],
    categorical_columns: list[str],
    max_rows: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, str]:
    """Compute SHAP importance when available, else fallback to model/permutation importance."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    if max_rows is not None and len(X_reference) > max_rows:
        sample_df = X_reference.sample(n=max_rows, random_state=random_state)
        y_sample = y_reference.loc[sample_df.index]
    else:
        sample_df = X_reference
        y_sample = y_reference

    transformed = preprocessor.transform(sample_df)
    feature_names = get_preprocessed_feature_names(preprocessor)

    if hasattr(transformed, "toarray"):
        transformed_dense = transformed.toarray()
    else:
        transformed_dense = np.asarray(transformed)

    transformed_df = pd.DataFrame(transformed_dense, columns=feature_names, index=sample_df.index)

    try:
        import shap

        explainer = shap.Explainer(model, transformed_df)
        shap_values = explainer(transformed_df)
        values = shap_values.values
        if values.ndim == 3:
            values = values[:, :, 1]

        importance = np.abs(values).mean(axis=0)
        raw_importance = pd.DataFrame({
            "feature": feature_names,
            "importance_score": importance,
        })
        grouped = aggregate_encoded_importance(raw_importance, numeric_columns, categorical_columns)
        return grouped, "shap"
    except Exception:
        pass

    # Fallback priority: model-native importance, then permutation importance.
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        size = min(len(feature_names), len(importances))
        raw_importance = pd.DataFrame(
            {
                "feature": feature_names[:size],
                "importance_score": importances[:size],
            }
        )
        grouped = aggregate_encoded_importance(raw_importance, numeric_columns, categorical_columns)
        return grouped, "feature_importance"

    if hasattr(model, "coef_"):
        coeffs = np.abs(np.asarray(model.coef_).ravel())
        size = min(len(feature_names), len(coeffs))
        raw_importance = pd.DataFrame(
            {
                "feature": feature_names[:size],
                "importance_score": coeffs[:size],
            }
        )
        grouped = aggregate_encoded_importance(raw_importance, numeric_columns, categorical_columns)
        return grouped, "coefficient_abs"

    permutation = permutation_importance(
        pipeline,
        sample_df,
        y_sample,
        n_repeats=5,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )
    importances = np.abs(np.asarray(permutation.importances_mean))
    base_feature_names = list(sample_df.columns)
    size = min(len(base_feature_names), len(importances))
    grouped = pd.DataFrame(
        {
            "feature": base_feature_names[:size],
            "importance_score": importances[:size],
        }
    ).sort_values("importance_score", ascending=False)
    grouped["importance_rank"] = np.arange(1, len(grouped) + 1)
    return grouped.reset_index(drop=True), "permutation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark models and export SHAP/fallback importance.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DATA_DIR / "customer_level_dataset.csv",
        help="Customer-level dataset path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Artifact output directory.",
    )
    parser.add_argument(
        "--max-explain-rows",
        type=int,
        default=None,
        help="Optional max rows for SHAP/fallback explainability per model.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_customer_dataset(args.input_path)
    feature_set = build_feature_set(df)

    X_train, X_test, y_train, y_test = train_test_split(
        feature_set.X,
        feature_set.y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=feature_set.y,
    )

    models, availability = build_models(random_state=args.random_state)

    metrics_rows: list[dict[str, Any]] = []
    model_explainability_rows: list[pd.DataFrame] = []

    shap_dir = args.output_dir / "shap_by_model"
    shap_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        preprocessor = build_preprocessor(feature_set.numeric_columns, feature_set.categorical_columns)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_score = probability_scores(pipeline, X_test)
        metrics = evaluate(y_test, y_score)
        metrics["model_name"] = model_name

        importance_df, method_used = shap_or_fallback_importance(
            model_name=model_name,
            pipeline=pipeline,
            X_reference=X_test,
            y_reference=y_test,
            numeric_columns=feature_set.numeric_columns,
            categorical_columns=feature_set.categorical_columns,
            max_rows=args.max_explain_rows,
            random_state=args.random_state,
        )
        importance_df["model_name"] = model_name
        importance_df["method_used"] = method_used
        metrics["explainability_method"] = method_used
        metrics_rows.append(metrics)
        model_explainability_rows.append(importance_df)

        model_file = shap_dir / f"{model_name.lower()}_shap_importance.csv"
        importance_df.to_csv(model_file, index=False)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by=["roc_auc", "f1", "recall"], ascending=False)
    metrics_df.to_csv(args.output_dir / "model_comparison_all_models.csv", index=False)

    combined_explain_df = pd.concat(model_explainability_rows, ignore_index=True)
    combined_explain_df.to_csv(args.output_dir / "model_shap_comparison_all_models.csv", index=False)

    save_json(availability, args.output_dir / "model_availability.json")

    # Export feature list used for modeling all values.
    pd.DataFrame(
        {
            "feature": feature_set.X.columns,
            "target_variable": "premium_customer",
            "used_in_model": True,
        }
    ).to_csv(args.output_dir / "all_model_features_used.csv", index=False)

    print("Completed model benchmark + SHAP/fallback explainability.")
    print(f"Features used: {len(feature_set.X.columns)}")
    print(f"Models trained: {len(models)}")
    print(f"Best model by ROC AUC: {metrics_df.iloc[0]['model_name']}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
