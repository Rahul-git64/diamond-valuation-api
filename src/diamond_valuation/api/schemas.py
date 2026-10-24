"""Pydantic v2 request and response schemas for single and batch diamond valuation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DiamondStoneRequest(BaseModel):
    """Input physical dimensions and GIA grades for a single diamond stone."""

    carat: float = Field(..., ge=0.10, le=10.0, description="Carat weight of the diamond")
    cut: str = Field(..., description="GIA cut grade: Fair, Good, Very Good, Premium, Ideal")
    color: str = Field(..., description="GIA color grade: J (lowest) to D (colorless)")
    clarity: str = Field(..., description="GIA clarity grade: I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF")
    depth: float = Field(..., ge=40.0, le=80.0, description="Total depth percentage")
    table: float = Field(..., ge=40.0, le=95.0, description="Table width percentage")
    x: float = Field(..., gt=0.5, le=30.0, description="Length in millimeters")
    y: float = Field(..., gt=0.5, le=30.0, description="Width in millimeters")
    z: float = Field(..., gt=0.3, le=20.0, description="Depth in millimeters")


class BatchDiamondRequest(BaseModel):
    """High-throughput batch valuation payload supporting up to 500 stones."""

    stones: list[DiamondStoneRequest] = Field(..., min_length=1, max_length=500)


class DiamondValuationItem(BaseModel):
    """Single stone valuation result with geometry telemetry and 90% interval."""

    predicted_price_usd: float
    price_per_carat_usd: float
    prediction_interval_usd: dict[str, float]
    volume_mm3: float
    table_to_depth_ratio: float
    gia_grade_summary: str
    model_name: str
    model_version: str


class DiamondPredictionResponse(DiamondValuationItem):
    """Single stone valuation response with inference latency."""

    latency_ms: float


class BatchDiamondResponse(BaseModel):
    """Aggregated response for batch diamond valuation."""

    count: int
    total_portfolio_value_usd: float
    mean_price_usd: float
    valuations: list[DiamondValuationItem]
    latency_ms: float


class HealthResponse(BaseModel):
    """Service health and active model metrics."""

    status: str
    service: str
    version: str
    uptime_seconds: float
    active_model: str
    model_metrics: dict[str, float]
    buffered_requests: int


class NumericalDriftMetric(BaseModel):
    """KS-test and PSI drift telemetry for a numerical gemological feature."""

    ks_statistic: float
    ks_p_value: float
    psi: float
    mean_reference: float
    mean_current: float
    drift_detected: bool


class CategoricalDriftMetric(BaseModel):
    """L1 distribution shift telemetry for a GIA categorical grade."""

    l1_divergence: float
    drift_detected: bool


class DriftReportResponse(BaseModel):
    """Covariate drift report across carat weight, geometry, and GIA grades."""

    status: str
    overall_drift_detected: bool
    sample_size: int
    reference_size: int
    numerical_features: dict[str, NumericalDriftMetric]
    categorical_features: dict[str, CategoricalDriftMetric]
