"""Deterministic offline GIA diamond dataset generator and train/test ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from diamond_valuation.core.config import AppConfig
from diamond_valuation.core.exceptions import DataIngestionError
from diamond_valuation.core.logger import get_logger, log_event
from diamond_valuation.data.validation import validate_diamond_dataframe

logger = get_logger(__name__)

CUT_MULTIPLIERS: dict[str, float] = {
    "Fair": 0.84,
    "Good": 0.91,
    "Very Good": 0.98,
    "Premium": 1.05,
    "Ideal": 1.12,
}

COLOR_MULTIPLIERS: dict[str, float] = {
    "J": 0.78,
    "I": 0.85,
    "H": 0.92,
    "G": 0.99,
    "F": 1.06,
    "E": 1.13,
    "D": 1.22,
}

CLARITY_MULTIPLIERS: dict[str, float] = {
    "I1": 0.65,
    "SI2": 0.80,
    "SI1": 0.90,
    "VS2": 0.98,
    "VS1": 1.06,
    "VVS2": 1.16,
    "VVS1": 1.26,
    "IF": 1.40,
}


def generate_synthetic_diamonds(n_rows: int = 1800, random_state: int = 42) -> pd.DataFrame:
    """Generate a deterministic gemological dataset following GIA grade multipliers and carat power laws."""
    rng = np.random.default_rng(random_state)

    cuts = rng.choice(
        list(CUT_MULTIPLIERS.keys()),
        size=n_rows,
        p=[0.04, 0.10, 0.22, 0.26, 0.38],
    )
    colors = rng.choice(
        list(COLOR_MULTIPLIERS.keys()),
        size=n_rows,
        p=[0.06, 0.10, 0.16, 0.21, 0.18, 0.17, 0.12],
    )
    clarities = rng.choice(
        list(CLARITY_MULTIPLIERS.keys()),
        size=n_rows,
        p=[0.03, 0.15, 0.23, 0.22, 0.15, 0.10, 0.07, 0.05],
    )

    carat = np.round(np.clip(rng.lognormal(mean=-0.32, sigma=0.46, size=n_rows), 0.20, 4.50), 2)
    depth = np.round(np.clip(rng.normal(61.75, 1.35, size=n_rows), 55.0, 70.0), 1)
    table = np.round(np.clip(rng.normal(57.45, 2.10, size=n_rows), 50.0, 68.0), 1)

    # Physical millimeter dimensions scale with cube-root of carat weight
    base_len = 6.45 * (carat ** (1.0 / 3.0))
    x = np.round(np.clip(base_len + rng.normal(0.0, 0.07, size=n_rows), 3.5, 11.5), 2)
    y = np.round(np.clip(x * rng.normal(1.002, 0.006, size=n_rows), 3.5, 11.5), 2)
    z = np.round(np.clip(0.5 * (x + y) * (depth / 100.0) + rng.normal(0.0, 0.03, size=n_rows), 2.1, 7.5), 2)

    prices = np.empty(n_rows, dtype=float)
    for i in range(n_rows):
        base_carat_price = 3850.0 * (carat[i] ** 1.82)
        grade_factor = (
            CUT_MULTIPLIERS[str(cuts[i])]
            * COLOR_MULTIPLIERS[str(colors[i])]
            * CLARITY_MULTIPLIERS[str(clarities[i])]
        )
        noise = rng.normal(1.0, 0.045)
        prices[i] = round(float(np.clip(base_carat_price * grade_factor * noise, 320.0, 28500.0)), 2)

    return pd.DataFrame(
        {
            "carat": carat,
            "cut": cuts,
            "color": colors,
            "clarity": clarities,
            "depth": depth,
            "table": table,
            "x": x,
            "y": y,
            "z": z,
            "price": prices,
        }
    )


def ingest_data(config: AppConfig, source_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest diamond dataset from CSV or generate deterministic offline dataset and split train/test."""
    try:
        if source_path and source_path.exists():
            df = pd.read_csv(source_path)
            log_event(logger, "loaded_external_diamond_csv", path=str(source_path), rows=len(df))
        elif config.data.raw_data_path.exists():
            df = pd.read_csv(config.data.raw_data_path)
            log_event(logger, "loaded_cached_raw_diamond_csv", path=str(config.data.raw_data_path), rows=len(df))
        else:
            df = generate_synthetic_diamonds(
                n_rows=config.data.synthetic_rows,
                random_state=config.data.random_state,
            )
            config.data.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(config.data.raw_data_path, index=False)
            log_event(logger, "generated_synthetic_diamonds", path=str(config.data.raw_data_path), rows=len(df))
    except Exception as exc:
        raise DataIngestionError(f"Failed to ingest diamond dataset: {exc}") from exc

    validated_df = validate_diamond_dataframe(df, config, require_target=True)
    min_cut_count = int(validated_df["cut"].value_counts().min())
    stratify_col = validated_df["cut"] if min_cut_count >= 2 else None

    train_df, test_df = train_test_split(
        validated_df,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=stratify_col,
    )

    config.data.train_data_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.data.train_data_path, index=False)
    test_df.to_csv(config.data.test_data_path, index=False)

    log_event(logger, "diamond_data_split_completed", train_rows=len(train_df), test_rows=len(test_df))
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
