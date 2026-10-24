"""Custom exception hierarchy for gemological diamond valuation pipeline."""

from __future__ import annotations


class DiamondValuationError(Exception):
    """Base exception for all diamond valuation pipeline errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(DiamondValuationError):
    """Raised when YAML configuration cannot be loaded or validated."""


class DataIngestionError(DiamondValuationError):
    """Raised when diamond dataset ingestion or synthetic generation fails."""


class DataValidationError(DiamondValuationError):
    """Raised when GIA ordinal grades or physical geometry bounds are violated."""


class FeatureEngineeringError(DiamondValuationError):
    """Raised when physical geometry feature computation fails."""


class ModelTrainingError(DiamondValuationError):
    """Raised when regression training fails holdout quality gates."""


class InferenceError(DiamondValuationError):
    """Raised when single or batch valuation inference fails."""
