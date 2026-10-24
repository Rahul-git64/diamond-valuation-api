"""FastAPI service exposing single (/predict) and 500-stone batch (/predict/batch) valuation endpoints."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from diamond_valuation.api.monitoring import DriftSupervisor
from diamond_valuation.api.schemas import (
    BatchDiamondRequest,
    BatchDiamondResponse,
    DiamondPredictionResponse,
    DiamondStoneRequest,
    DiamondValuationItem,
    DriftReportResponse,
    HealthResponse,
)
from diamond_valuation.core.config import AppConfig, load_config
from diamond_valuation.core.exceptions import DiamondValuationError
from diamond_valuation.core.logger import get_logger, log_event
from diamond_valuation.models.registry import ModelRegistry

logger = get_logger(__name__)


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI gemological diamond valuation application."""
    cfg = config or load_config()
    registry = ModelRegistry(cfg)
    supervisor = DriftSupervisor(cfg)
    start_time = time.monotonic()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        log_event(logger, "service_startup_initiated", version=cfg.version)
        registry.ensure_loaded(auto_train=True)
        log_event(
            logger,
            "service_startup_complete",
            active_model=registry.metadata.get("model_name", "unknown"),
        )
        yield

    app = FastAPI(
        title="Diamond Valuation API Service",
        description=(
            "Gemological diamond valuation API enforcing strict GIA ordinal hierarchies, "
            "physical geometry validation, single/batch (up to 500 stones) inference, and drift supervision."
        ),
        version=cfg.version,
        lifespan=lifespan,
    )

    app.state.config = cfg
    app.state.registry = registry
    app.state.supervisor = supervisor
    app.state.start_time = start_time

    @app.get("/health", response_model=HealthResponse, tags=["Supervision"])
    def health_check() -> HealthResponse:
        """Return service health, uptime, and active model holdout metrics."""
        app.state.registry.ensure_loaded(auto_train=True)
        uptime = round(time.monotonic() - app.state.start_time, 2)
        meta = app.state.registry.metadata
        metrics = {
            k: float(v)
            for k, v in meta.get("metrics", {}).items()
            if isinstance(v, (int, float))
        }
        return HealthResponse(
            status="healthy",
            service=cfg.project_name,
            version=cfg.version,
            uptime_seconds=uptime,
            active_model=str(meta.get("model_name", "unknown")),
            model_metrics=metrics,
            buffered_requests=len(app.state.supervisor.buffer),
        )

    @app.post("/predict", response_model=DiamondPredictionResponse, tags=["Inference"])
    def predict_single_stone(request: DiamondStoneRequest) -> DiamondPredictionResponse:
        """Validate a single diamond stone, compute physical geometry, and predict USD value."""
        t0 = time.perf_counter()
        payload = request.model_dump()
        try:
            res = app.state.registry.predict_single(payload)
        except DiamondValuationError as exc:
            log_event(logger, "single_inference_failure", error=exc.message)
            raise HTTPException(status_code=422, detail=exc.message) from exc

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        app.state.supervisor.record_requests([res["validated_stone"]])

        log_event(
            logger,
            "single_valuation_completed",
            carat=request.carat,
            cut=request.cut,
            color=request.color,
            clarity=request.clarity,
            predicted_price_usd=res["predicted_price_usd"],
            latency_ms=latency_ms,
        )

        return DiamondPredictionResponse(
            predicted_price_usd=res["predicted_price_usd"],
            price_per_carat_usd=res["price_per_carat_usd"],
            prediction_interval_usd=res["prediction_interval_usd"],
            volume_mm3=res["volume_mm3"],
            table_to_depth_ratio=res["table_to_depth_ratio"],
            gia_grade_summary=res["gia_grade_summary"],
            model_name=res["model_name"],
            model_version=res["model_version"],
            latency_ms=latency_ms,
        )

    @app.post("/predict/batch", response_model=BatchDiamondResponse, tags=["Inference"])
    def predict_batch_stones(request: BatchDiamondRequest) -> BatchDiamondResponse:
        """Evaluate up to 500 diamond stones in a single vectorized batch request."""
        t0 = time.perf_counter()
        raw_stones = [s.model_dump() for s in request.stones]
        try:
            batch_results = app.state.registry.predict_batch(raw_stones)
        except DiamondValuationError as exc:
            log_event(logger, "batch_inference_failure", error=exc.message)
            raise HTTPException(status_code=422, detail=exc.message) from exc

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        app.state.supervisor.record_requests([r["validated_stone"] for r in batch_results])

        items = [
            DiamondValuationItem(
                predicted_price_usd=r["predicted_price_usd"],
                price_per_carat_usd=r["price_per_carat_usd"],
                prediction_interval_usd=r["prediction_interval_usd"],
                volume_mm3=r["volume_mm3"],
                table_to_depth_ratio=r["table_to_depth_ratio"],
                gia_grade_summary=r["gia_grade_summary"],
                model_name=r["model_name"],
                model_version=r["model_version"],
            )
            for r in batch_results
        ]
        total_val = round(sum(i.predicted_price_usd for i in items), 2)
        mean_val = round(total_val / max(len(items), 1), 2)

        log_event(
            logger,
            "batch_valuation_completed",
            count=len(items),
            total_portfolio_value_usd=total_val,
            latency_ms=latency_ms,
        )

        return BatchDiamondResponse(
            count=len(items),
            total_portfolio_value_usd=total_val,
            mean_price_usd=mean_val,
            valuations=items,
            latency_ms=latency_ms,
        )

    @app.get("/monitor/drift", response_model=DriftReportResponse, tags=["Supervision"])
    def monitor_drift() -> DriftReportResponse:
        """Evaluate carat/geometry KS/PSI drift and GIA grade L1 shift."""
        app.state.registry.ensure_loaded(auto_train=True)
        report = app.state.supervisor.evaluate(app.state.registry.reference_stats)
        log_event(
            logger,
            "drift_evaluation_executed",
            status=report["status"],
            sample_size=report["sample_size"],
            overall_drift_detected=report["overall_drift_detected"],
        )
        return DriftReportResponse(**report)

    return app


app = create_app()
