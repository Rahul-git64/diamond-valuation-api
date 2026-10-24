"""Integration tests for FastAPI single/batch diamond valuation and drift supervision."""

from __future__ import annotations

from fastapi.testclient import TestClient

from diamond_valuation.api.main import create_app
from diamond_valuation.core.config import load_config


def test_single_and_batch_valuation_endpoints() -> None:
    cfg = load_config()
    app = create_app(cfg)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    stone = {
        "carat": 1.10,
        "cut": "Ideal",
        "color": "E",
        "clarity": "VVS2",
        "depth": 61.6,
        "table": 57.0,
        "x": 6.62,
        "y": 6.65,
        "z": 4.09,
    }
    single_resp = client.post("/predict", json=stone)
    assert single_resp.status_code == 200
    s_body = single_resp.json()
    assert s_body["predicted_price_usd"] > 0
    assert s_body["volume_mm3"] > 0

    batch_resp = client.post("/predict/batch", json={"stones": [stone, {**stone, "carat": 1.50, "x": 7.3, "y": 7.3, "z": 4.5}]})
    assert batch_resp.status_code == 200
    b_body = batch_resp.json()
    assert b_body["count"] == 2
    assert b_body["total_portfolio_value_usd"] > s_body["predicted_price_usd"]


def test_drift_supervision_progression() -> None:
    cfg = load_config()
    app = create_app(cfg)
    client = TestClient(app)

    init_drift = client.get("/monitor/drift")
    assert init_drift.status_code == 200
    assert init_drift.json()["status"] == "insufficient_data"

    stones = [
        {
            "carat": c,
            "cut": "Premium",
            "color": "G",
            "clarity": "VS1",
            "depth": 61.8,
            "table": 58.0,
            "x": 6.2,
            "y": 6.2,
            "z": 3.8,
        }
        for c in [0.7, 0.9, 1.1, 1.3, 1.5, 1.7]
    ]
    batch_r = client.post("/predict/batch", json={"stones": stones})
    assert batch_r.status_code == 200

    drift_after = client.get("/monitor/drift")
    assert drift_after.status_code == 200
    d_body = drift_after.json()
    assert d_body["sample_size"] == 6
    assert d_body["status"] in {"healthy", "drift_detected"}
    assert "carat" in d_body["numerical_features"]
    assert "color" in d_body["categorical_features"]
