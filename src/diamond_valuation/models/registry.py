"""Model registry supporting single-stone and vectorized 500-stone batch valuations."""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from diamond_valuation.core.config import AppConfig
from diamond_valuation.core.exceptions import DataValidationError, InferenceError
from diamond_valuation.core.logger import get_logger, log_event
from diamond_valuation.data.validation import validate_single_diamond
from diamond_valuation.models.trainer import train_pipeline

logger = get_logger(__name__)


class ModelRegistry:
    """Manages lifecycle of preprocessor, trained estimator, metadata, and reference stats."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.preprocessor: Any = None
        self.model: Any = None
        self.metadata: dict[str, Any] = {}
        self.reference_stats: dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        return bool(self.preprocessor is not None and self.model is not None)

    def ensure_loaded(self, auto_train: bool = True) -> None:
        """Load model artifacts from disk or trigger training if missing."""
        if self.is_loaded:
            return

        m_cfg = self.config.model
        exists = (
            m_cfg.preprocessor_path.exists()
            and m_cfg.model_path.exists()
            and m_cfg.metadata_path.exists()
            and m_cfg.reference_stats_path.exists()
        )
        if not exists:
            if not auto_train:
                raise InferenceError("Model artifacts missing and auto_train is disabled.")
            log_event(logger, "auto_training_triggered", reason="missing_artifacts")
            train_pipeline(self.config)

        try:
            self.preprocessor = joblib.load(m_cfg.preprocessor_path)
            self.model = joblib.load(m_cfg.model_path)
            with m_cfg.metadata_path.open("r", encoding="utf-8") as f_meta:
                self.metadata = json.load(f_meta)
            with m_cfg.reference_stats_path.open("r", encoding="utf-8") as f_ref:
                self.reference_stats = json.load(f_ref)
        except Exception as exc:
            raise InferenceError(f"Failed to load diamond model artifacts: {exc}") from exc

    def predict_batch(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate and evaluate a batch of up to max_batch_size diamond records."""
        if not payloads:
            raise DataValidationError("Batch diamond list cannot be empty.")
        max_batch = self.config.model.max_batch_size
        if len(payloads) > max_batch:
            raise DataValidationError(
                f"Batch size ({len(payloads)}) exceeds maximum allowed ({max_batch})."
            )

        self.ensure_loaded(auto_train=True)
        validated_list = [validate_single_diamond(p, self.config) for p in payloads]
        input_df = pd.DataFrame(validated_list)

        try:
            X_trans = self.preprocessor.transform(input_df)
            raw_preds = np.maximum(self.model.predict(X_trans), 120.0)
        except Exception as exc:
            raise InferenceError(f"Batch inference execution failed: {exc}") from exc

        q90_rel = float(self.metadata.get("q90_relative_error", 0.08))
        model_name = str(self.metadata.get("model_name", "GradientBoostingRegressor"))
        model_version = str(self.metadata.get("model_version", self.config.version))

        results: list[dict[str, Any]] = []
        for stone, pred_val in zip(validated_list, raw_preds, strict=True):
            price_usd = round(float(pred_val), 2)
            price_per_carat = round(price_usd / stone["carat"], 2)
            vol_mm3 = round(stone["x"] * stone["y"] * stone["z"], 2)
            td_ratio = round(stone["table"] / stone["depth"], 4)
            lower_usd = round(max(100.0, price_usd * (1.0 - q90_rel)), 2)
            upper_usd = round(price_usd * (1.0 + q90_rel), 2)

            results.append(
                {
                    "predicted_price_usd": price_usd,
                    "price_per_carat_usd": price_per_carat,
                    "prediction_interval_usd": {"lower": lower_usd, "upper": upper_usd},
                    "volume_mm3": vol_mm3,
                    "table_to_depth_ratio": td_ratio,
                    "gia_grade_summary": f"{stone['carat']:.2f}ct {stone['color']} {stone['clarity']} ({stone['cut']})",
                    "validated_stone": stone,
                    "model_name": model_name,
                    "model_version": model_version,
                }
            )
        return results

    def predict_single(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate and predict valuation for a single diamond stone."""
        return self.predict_batch([payload])[0]
