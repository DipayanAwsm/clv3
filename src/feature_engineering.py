"""Feature engineering utilities for premium customer classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "premium_customer"
ID_COLUMN = "CustomerID"

# Columns excluded from model fitting due to leakage/identifier semantics.
EXCLUDED_FEATURE_COLUMNS = {
    ID_COLUMN,
    TARGET_COLUMN,
    "customer_clv",
    "premium_threshold",
    "value_segment",
}


@dataclass
class FeatureSet:
    X: pd.DataFrame
    y: pd.Series
    numeric_columns: list[str]
    categorical_columns: list[str]


def load_customer_dataset(path: Path) -> pd.DataFrame:
    """Load prepared customer-level dataset."""
    return pd.read_csv(path)


def build_feature_set(df: pd.DataFrame) -> FeatureSet:
    """Prepare train-ready features and target from customer dataset."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    feature_columns = [col for col in df.columns if col not in EXCLUDED_FEATURE_COLUMNS]
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    numeric_columns = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_columns = [col for col in X.columns if col not in numeric_columns]

    return FeatureSet(X=X, y=y, numeric_columns=numeric_columns, categorical_columns=categorical_columns)


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    """Build model preprocessing pipeline for numeric and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    return preprocessor


def get_preprocessed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract transformed feature names from fitted column transformer."""
    try:
        feature_names = preprocessor.get_feature_names_out()
        return [name.replace("num__", "").replace("cat__", "") for name in feature_names]
    except Exception:
        names: list[str] = []
        for transformer_name, transformer, columns in preprocessor.transformers_:
            if transformer_name == "remainder":
                continue
            if transformer_name == "num":
                names.extend(columns)
            elif transformer_name == "cat":
                encoder = transformer.named_steps.get("encoder")
                if encoder is not None:
                    names.extend(list(encoder.get_feature_names_out(columns)))
        return names
