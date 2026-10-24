"""Gemological GIA grade and physical dimension validation for diamond payloads and DataFrames."""

from __future__ import annotations

from typing import Any

import pandas as pd

from diamond_valuation.core.config import AppConfig
from diamond_valuation.core.exceptions import DataValidationError


def validate_single_diamond(payload: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    """Validate physical dimensions and GIA grades for a single diamond stone."""
    required = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise DataValidationError(f"Missing required diamond fields: {missing}")

    try:
        carat = float(payload["carat"])
        depth = float(payload["depth"])
        table = float(payload["table"])
        x = float(payload["x"])
        y = float(payload["y"])
        z = float(payload["z"])
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"Physical measurements must be numeric: {exc}") from exc

    cut = str(payload["cut"]).strip()
    color = str(payload["color"]).strip().upper()
    clarity = str(payload["clarity"]).strip().upper()

    if not (0.10 <= carat <= 10.0):
        raise DataValidationError(f"carat out of gemological bounds [0.10, 10.0]: {carat}")
    if not (40.0 <= depth <= 80.0):
        raise DataValidationError(f"depth percentage out of bounds [40.0, 80.0]: {depth}")
    if not (40.0 <= table <= 95.0):
        raise DataValidationError(f"table percentage out of bounds [40.0, 95.0]: {table}")
    if x <= 0.5 or y <= 0.5 or z <= 0.3 or x > 30.0 or y > 30.0 or z > 20.0:
        raise DataValidationError(f"Invalid physical dimensions (mm): x={x}, y={y}, z={z}")

    allowed_cut = set(config.features.categories["cut"])
    allowed_color = set(config.features.categories["color"])
    allowed_clarity = set(config.features.categories["clarity"])

    if cut not in allowed_cut:
        raise DataValidationError(f"Invalid GIA cut '{cut}'. Allowed: {config.features.categories['cut']}")
    if color not in allowed_color:
        raise DataValidationError(f"Invalid GIA color '{color}'. Allowed: {config.features.categories['color']}")
    if clarity not in allowed_clarity:
        raise DataValidationError(f"Invalid GIA clarity '{clarity}'. Allowed: {config.features.categories['clarity']}")

    return {
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z,
    }


def validate_diamond_dataframe(
    df: pd.DataFrame,
    config: AppConfig,
    require_target: bool = True,
) -> pd.DataFrame:
    """Validate a diamond DataFrame, dropping zero-dimension rows and checking GIA grades."""
    if df.empty:
        raise DataValidationError("Input diamond DataFrame cannot be empty.")

    cleaned = df.drop(columns=["id", "Unnamed: 0"], errors="ignore").copy()
    required_cols = list(config.features.numerical_columns) + list(config.features.categorical_columns)
    if require_target:
        required_cols.append(config.features.target_column)

    missing = [col for col in required_cols if col not in cleaned.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    cleaned = cleaned.dropna(subset=required_cols).reset_index(drop=True)
    # Filter out corrupted zero-dimension records common in raw gemstone datasets
    valid_dims = (cleaned["x"] > 0.5) & (cleaned["y"] > 0.5) & (cleaned["z"] > 0.3) & (cleaned["carat"] > 0.05)
    cleaned = cleaned.loc[valid_dims].reset_index(drop=True)
    if cleaned.empty:
        raise DataValidationError("No valid rows remaining after filtering zero/negative dimensions.")

    for cat_col in config.features.categorical_columns:
        allowed = set(config.features.categories[cat_col])
        invalid = set(cleaned[cat_col].astype(str).unique()) - allowed
        if invalid:
            raise DataValidationError(f"Unsupported GIA grades in '{cat_col}': {sorted(invalid)}")

    if require_target and (cleaned[config.features.target_column] <= 0).any():
        raise DataValidationError("Diamond price must be strictly positive.")

    return cleaned
