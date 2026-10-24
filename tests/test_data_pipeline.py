"""Unit tests for GIA grade validation, offline ingestion, and physical geometry features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diamond_valuation.core.config import load_config
from diamond_valuation.core.exceptions import DataValidationError
from diamond_valuation.data.ingestion import generate_synthetic_diamonds, ingest_data
from diamond_valuation.data.preprocessing import (
    DiamondGeometryEngineer,
    build_preprocessing_pipeline,
)
from diamond_valuation.data.validation import (
    validate_diamond_dataframe,
    validate_single_diamond,
)


def test_gia_hierarchy_and_synthetic_ingestion(tmp_path: Path) -> None:
    cfg = load_config()
    assert cfg.features.categories["cut"] == ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    assert cfg.features.categories["color"] == ["J", "I", "H", "G", "F", "E", "D"]
    assert cfg.features.categories["clarity"] == [
        "I1",
        "SI2",
        "SI1",
        "VS2",
        "VS1",
        "VVS2",
        "VVS1",
        "IF",
    ]

    df = generate_synthetic_diamonds(n_rows=150, random_state=11)
    assert len(df) == 150
    assert (df["price"] > 0).all()

    # Add a corrupted zero-dimension row and verify it is filtered out
    df_corrupt = df.copy()
    df_corrupt.loc[0, "z"] = 0.0
    cleaned = validate_diamond_dataframe(df_corrupt, cfg, require_target=True)
    assert len(cleaned) == 149

    csv_path = tmp_path / "diamonds_sample.csv"
    df.to_csv(csv_path, index=False)
    train_df, test_df = ingest_data(cfg, source_path=csv_path)
    assert len(train_df) + len(test_df) == 150


def test_single_stone_validation_rejects_invalid_grades_and_bounds() -> None:
    cfg = load_config()
    valid_stone = {
        "carat": 1.05,
        "cut": "Ideal",
        "color": "E",
        "clarity": "VS1",
        "depth": 61.8,
        "table": 57.0,
        "x": 6.52,
        "y": 6.55,
        "z": 4.04,
    }
    norm = validate_single_diamond(valid_stone, cfg)
    assert norm["color"] == "E"

    bad_color = {**valid_stone, "color": "Z"}
    with pytest.raises(DataValidationError, match="Invalid GIA color"):
        validate_single_diamond(bad_color, cfg)

    bad_dim = {**valid_stone, "z": 0.0}
    with pytest.raises(DataValidationError, match="Invalid physical dimensions"):
        validate_single_diamond(bad_dim, cfg)


def test_physical_geometry_engineer_and_ordinal_pipeline() -> None:
    cfg = load_config()
    df = generate_synthetic_diamonds(n_rows=50, random_state=21)
    X_raw = df.drop(columns=["price"])

    geo = DiamondGeometryEngineer()
    engineered = geo.fit_transform(X_raw)
    assert "volume_mm3" in engineered.columns
    assert "table_to_depth_ratio" in engineered.columns
    assert "depth_deviation_pct" in engineered.columns
    assert (engineered["volume_mm3"] > 0).all()

    pipeline = build_preprocessing_pipeline(cfg)
    X_mat = pipeline.fit_transform(X_raw)
    assert X_mat.shape == (50, 14)
    assert np.isfinite(X_mat).all()
