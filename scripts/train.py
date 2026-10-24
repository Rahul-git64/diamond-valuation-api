"""CLI script to execute gemological diamond valuation training pipeline."""

from __future__ import annotations

from diamond_valuation.core.config import load_config
from diamond_valuation.core.logger import get_logger, log_event
from diamond_valuation.models.trainer import train_pipeline

logger = get_logger(__name__)


def main() -> None:
    """Load config and train diamond valuation models."""
    config = load_config()
    metadata = train_pipeline(config)
    log_event(
        logger,
        "cli_diamond_training_finished",
        model_name=metadata["model_name"],
        metrics=metadata["metrics"],
    )


if __name__ == "__main__":
    main()
