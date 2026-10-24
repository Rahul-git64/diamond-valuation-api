"""Unit tests for diamond regression trainer, single valuation, and 500-stone batch registry."""

from __future__ import annotations

import pytest

from diamond_valuation.core.config import load_config
from diamond_valuation.core.exceptions import DataValidationError
from diamond_valuation.models.registry import ModelRegistry
from diamond_valuation.models.trainer import train_pipeline


def test_diamond_trainer_and_leaderboard() -> None:
    cfg = load_config()
    metadata = train_pipeline(cfg)
    assert "metrics" in metadata
    assert metadata["metrics"]["r2"] >= cfg.model.r2_threshold
    assert cfg.model.model_path.exists()
    assert cfg.model.preprocessor_path.exists()
    assert cfg.model.reference_stats_path.exists()


def test_registry_single_and_batch_predictions() -> None:
    cfg = load_config()
    registry = ModelRegistry(cfg)
    registry.ensure_loaded(auto_train=True)

    stone_d_if = {
        "carat": 1.20,
        "cut": "Ideal",
        "color": "D",
        "clarity": "IF",
        "depth": 61.5,
        "table": 57.0,
        "x": 6.82,
        "y": 6.85,
        "z": 4.21,
    }
    stone_j_i1 = {
        "carat": 1.20,
        "cut": "Fair",
        "color": "J",
        "clarity": "I1",
        "depth": 63.5,
        "table": 60.0,
        "x": 6.70,
        "y": 6.65,
        "z": 4.24,
    }

    res_single = registry.predict_single(stone_d_if)
    assert res_single["predicted_price_usd"] > 0
    assert res_single["volume_mm3"] > 0

    batch_res = registry.predict_batch([stone_d_if, stone_j_i1])
    assert len(batch_res) == 2
    # Verify GIA grade ordering: D/IF/Ideal must be valued higher than J/I1/Fair for same carat
    assert batch_res[0]["predicted_price_usd"] > batch_res[1]["predicted_price_usd"]

    with pytest.raises(DataValidationError, match="exceeds maximum allowed"):
        registry.predict_batch([stone_d_if] * 501)
