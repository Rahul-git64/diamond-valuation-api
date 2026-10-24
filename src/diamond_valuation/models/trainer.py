"""Gemological regression model trainer, evaluator, and reference drift stats exporter."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from diamond_valuation.core.config import AppConfig
from diamond_valuation.core.exceptions import ModelTrainingError
from diamond_valuation.core.logger import get_logger, log_event
from diamond_valuation.data.ingestion import ingest_data
from diamond_valuation.data.preprocessing import build_preprocessing_pipeline

logger = get_logger(__name__)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics: RMSE, MAE, MAPE, and R2."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mape_pct": round(mape, 2),
        "r2": round(r2, 4),
    }


def build_reference_stats(train_df: pd.DataFrame, config: AppConfig) -> dict[str, Any]:
    """Export reference training distributions for carat/geometry KS/PSI and GIA grade shifts."""
    numerical_stats: dict[str, Any] = {}
    rng = np.random.default_rng(config.data.random_state)

    for col in config.features.numerical_columns:
        series = train_df[col].astype(float).to_numpy()
        sampled = rng.choice(series, size=min(len(series), 400), replace=False)
        quantiles = np.quantile(series, np.linspace(0.0, 1.0, 11)).tolist()
        numerical_stats[col] = {
            "mean": round(float(np.mean(series)), 4),
            "std": round(float(np.std(series)), 4),
            "quantiles": [round(float(q), 4) for q in quantiles],
            "sample": [round(float(x), 4) for x in sampled],
        }

    categorical_stats: dict[str, dict[str, float]] = {}
    for col in config.features.categorical_columns:
        counts = train_df[col].astype(str).value_counts(normalize=True)
        categorical_stats[col] = {str(k): round(float(v), 4) for k, v in counts.items()}

    return {
        "n_reference": len(train_df),
        "numerical": numerical_stats,
        "categorical": categorical_stats,
    }


def train_pipeline(config: AppConfig) -> dict[str, Any]:
    """Execute end-to-end training and selection across log-linear and ensemble estimators."""
    train_df, test_df = ingest_data(config)
    target_col = config.features.target_column

    X_train_raw = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(float).to_numpy()
    X_test_raw = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(float).to_numpy()

    preprocessor = build_preprocessing_pipeline(config)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    seed = config.data.random_state
    candidates: dict[str, Any] = {
        "LogRidge": TransformedTargetRegressor(
            regressor=Ridge(alpha=1.0),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=150,
            max_depth=16,
            random_state=seed,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.07,
            max_depth=4,
            subsample=0.85,
            random_state=seed,
        ),
    }

    leaderboard: dict[str, dict[str, float]] = {}
    best_name = ""
    best_model: Any = None
    best_r2 = -np.inf
    best_preds = np.zeros_like(y_test)

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = np.maximum(model.predict(X_test), 100.0)
        metrics = evaluate_predictions(y_test, preds)
        leaderboard[name] = metrics
        log_event(logger, "candidate_model_evaluated", model=name, **metrics)
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_name = name
            best_model = model
            best_preds = preds

    if best_r2 < config.model.r2_threshold:
        raise ModelTrainingError(
            f"Best model {best_name} R2 ({best_r2:.4f}) below threshold ({config.model.r2_threshold:.4f})"
        )

    rel_residuals = np.abs(y_test - best_preds) / np.maximum(best_preds, 1.0)
    q90_rel_error = round(float(np.quantile(rel_residuals, 0.90)), 4)

    config.model.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, config.model.preprocessor_path)
    joblib.dump(best_model, config.model.model_path)

    metadata: dict[str, Any] = {
        "model_name": best_name,
        "model_version": config.version,
        "trained_at": datetime.now(UTC).isoformat(),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "q90_relative_error": q90_rel_error,
        "metrics": leaderboard[best_name],
        "leaderboard": leaderboard,
    }

    with config.model.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    ref_stats = build_reference_stats(train_df, config)
    with config.model.reference_stats_path.open("w", encoding="utf-8") as handle:
        json.dump(ref_stats, handle, indent=2)

    log_event(logger, "diamond_training_completed", winner=best_name, **leaderboard[best_name])
    return metadata
