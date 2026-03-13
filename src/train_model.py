"""Train and evaluate premium customer classification models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
from feature_engineering import (
    TARGET_COLUMN,
    build_feature_set,
    build_preprocessor,
    get_preprocessed_feature_names,
    load_customer_dataset,
)
from utils import ARTIFACTS_DIR, DATA_DIR, assign_action_band, ensure_directories, save_json


def _build_candidate_models(
    random_state: int = 42, include_optional_models: bool = True
) -> dict[str, Any]:
    models: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=450,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
    }

    if include_optional_models:
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
        except Exception:
            pass

    return models


def _probability_scores(model_pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model_pipeline, "predict_proba"):
        probabilities = model_pipeline.predict_proba(X)
        if probabilities.ndim == 2:
            return probabilities[:, 1]
        return probabilities

    if hasattr(model_pipeline, "decision_function"):
        decision_values = model_pipeline.decision_function(X)
        min_val = np.min(decision_values)
        max_val = np.max(decision_values)
        if max_val == min_val:
            return np.zeros_like(decision_values)
        return (decision_values - min_val) / (max_val - min_val)

    # Last-resort fallback.
    return model_pipeline.predict(X).astype(float)


def _evaluate_model(model_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    probabilities = _probability_scores(model_pipeline, X_test)
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return metrics


def _train_and_compare_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numeric_columns: list[str],
    categorical_columns: list[str],
    random_state: int,
    include_optional_models: bool,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    model_candidates = _build_candidate_models(
        random_state=random_state,
        include_optional_models=include_optional_models,
    )
    comparison_rows: list[dict[str, Any]] = []
    trained_pipelines: dict[str, Pipeline] = {}

    for model_name, model in model_candidates.items():
        preprocessor = build_preprocessor(numeric_columns, categorical_columns)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        metrics = _evaluate_model(pipeline, X_test, y_test)
        metrics["model_name"] = model_name
        comparison_rows.append(metrics)
        trained_pipelines[model_name] = pipeline

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["roc_auc", "f1", "recall"], ascending=False
    )
    return comparison_df, trained_pipelines


def _extract_feature_importance(
    pipeline: Pipeline,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = get_preprocessed_feature_names(preprocessor)

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_).ravel())
    else:
        importances = np.zeros(len(feature_names))

    # Align potential mismatches from model internals.
    size = min(len(feature_names), len(importances))
    feature_names = feature_names[:size]
    importances = importances[:size]

    fine_grained_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    categorical_prefixes = tuple(f"{col}_" for col in categorical_columns)

    def to_base_feature(feature: str) -> str:
        if feature in numeric_columns:
            return feature
        for prefix, base_col in zip(categorical_prefixes, categorical_columns):
            if feature.startswith(prefix):
                return base_col
        return feature

    grouped_df = (
        fine_grained_df.assign(base_feature=fine_grained_df["feature"].map(to_base_feature))
        .groupby("base_feature", as_index=False)["importance"]
        .sum()
        .rename(columns={"base_feature": "feature"})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    grouped_df["importance_rank"] = np.arange(1, len(grouped_df) + 1)
    return grouped_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train premium customer classification model.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DATA_DIR / "customer_level_dataset.csv",
        help="Customer-level dataset path.",
    )
    parser.add_argument(
        "--model-output-path",
        type=Path,
        default=ARTIFACTS_DIR / "premium_customer_model.joblib",
        help="Path to save the best model artifact.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=ARTIFACTS_DIR / "model_metrics.json",
        help="Path to save model metrics JSON.",
    )
    parser.add_argument(
        "--comparison-path",
        type=Path,
        default=ARTIFACTS_DIR / "model_comparison.csv",
        help="Path to save model comparison CSV.",
    )
    parser.add_argument(
        "--scores-path",
        type=Path,
        default=ARTIFACTS_DIR / "customer_scores.csv",
        help="Path to save customer-level score output.",
    )
    parser.add_argument(
        "--feature-importance-path",
        type=Path,
        default=ARTIFACTS_DIR / "feature_importance.csv",
        help="Path to save feature importance output.",
    )
    parser.add_argument(
        "--disable-optional-models",
        action="store_true",
        help="Disable optional XGBoost/LightGBM candidates for portable sklearn-only artifacts.",
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    df = load_customer_dataset(args.input_path)
    feature_set = build_feature_set(df)

    X_train, X_test, y_train, y_test = train_test_split(
        feature_set.X,
        feature_set.y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=feature_set.y,
    )

    comparison_df, trained_pipelines = _train_and_compare_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        numeric_columns=feature_set.numeric_columns,
        categorical_columns=feature_set.categorical_columns,
        random_state=args.random_state,
        include_optional_models=not args.disable_optional_models,
    )

    if comparison_df.empty:
        raise RuntimeError("No models were trained. Check environment dependencies.")

    best_model_name = comparison_df.iloc[0]["model_name"]
    best_pipeline = trained_pipelines[best_model_name]

    # Save model artifact with metadata needed for scoring and explainability.
    model_bundle = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "feature_columns": list(feature_set.X.columns),
        "numeric_columns": feature_set.numeric_columns,
        "categorical_columns": feature_set.categorical_columns,
        "target_column": TARGET_COLUMN,
    }
    args.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, args.model_output_path)

    # Save comparisons and metrics.
    comparison_df.to_csv(args.comparison_path, index=False)
    best_row = comparison_df.iloc[0].to_dict()

    metrics_payload = {
        "best_model": best_model_name,
        "selection_metric": "roc_auc",
        "best_model_metrics": best_row,
        "all_models": comparison_df.to_dict(orient="records"),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "premium_customer_rate": float(df[TARGET_COLUMN].mean()),
    }
    save_json(metrics_payload, args.metrics_path)

    # Full-customer scoring for actioning.
    full_probabilities = _probability_scores(best_pipeline, feature_set.X)
    scored_df = df[["CustomerID", "customer_clv", "value_segment", TARGET_COLUMN]].copy()
    scored_df["premium_score"] = full_probabilities
    scored_df["predicted_premium_customer"] = (scored_df["premium_score"] >= 0.5).astype(int)
    scored_df["action_band"] = scored_df["premium_score"].apply(assign_action_band)
    scored_df = scored_df.sort_values("premium_score", ascending=False)
    scored_df.to_csv(args.scores_path, index=False)

    # Feature importance output.
    importance_df = _extract_feature_importance(
        pipeline=best_pipeline,
        numeric_columns=feature_set.numeric_columns,
        categorical_columns=feature_set.categorical_columns,
    )
    importance_df.to_csv(args.feature_importance_path, index=False)

    print(f"Best model: {best_model_name}")
    print(
        f"Optional models enabled: {not args.disable_optional_models} "
        "(XGBoost/LightGBM candidates)"
    )
    print(f"ROC AUC: {best_row['roc_auc']:.4f}")
    print(f"Saved model to {args.model_output_path}")
    print(f"Saved metrics to {args.metrics_path}")
    print(f"Saved customer scores to {args.scores_path}")


if __name__ == "__main__":
    main()
