"""Shared utilities for the premium customer analytics project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"


def ensure_directories() -> None:
    """Create output directories if they do not already exist."""
    for directory in [DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def safe_divide(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    """Safely divide values and avoid divide-by-zero or NaN propagation."""
    try:
        if denominator in (0, None) or pd.isna(denominator):
            return default
        if numerator is None or pd.isna(numerator):
            return default
        return float(numerator) / float(denominator)
    except Exception:
        return default


def mode_or_unknown(series: pd.Series, unknown_value: str = "Unknown") -> str:
    """Return the mode from a series; fallback to unknown label."""
    non_null = series.dropna()
    if non_null.empty:
        return unknown_value
    mode_values = non_null.mode()
    if mode_values.empty:
        return unknown_value
    return str(mode_values.iloc[0])


def save_json(data: dict[str, Any], path: Path) -> None:
    """Write dictionary as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def assign_value_segment(clv: float, premium_threshold: float, low_threshold: float) -> str:
    """Map CLV into business-facing customer value segment."""
    if clv >= premium_threshold:
        return "Premium"
    if clv <= low_threshold:
        return "Low / Loss-Making"
    return "Core"


def assign_action_band(score: float) -> str:
    """Map probability score to recommended action band."""
    if score >= 0.75:
        return "Retain & Grow"
    if score >= 0.45:
        return "Protect"
    return "Monitor / Re-price"


def to_serializable(value: Any) -> Any:
    """Convert numpy/pandas objects into JSON serializable python objects."""
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value
