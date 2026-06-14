import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from service.main import app

client = TestClient(app)

HIGH_RISK = {
    "zone_class": "alpha",
    "operational_load": "high",
    "time_period": "night",
    "sensor_mode": "active",
    "pressure_delta_pa": 5.0,
    "temperature_c": 27.5,
    "humidity_pct": 68.0,
    "particle_density": 20000.0,
    "occupancy_count": 10,
    "maintenance_lag_days": 14,
    "airflow_pct": 60.0,
}

LOW_RISK = {
    "zone_class": "delta",
    "operational_load": "low",
    "time_period": "day",
    "sensor_mode": "active",
    "pressure_delta_pa": 23.0,
    "temperature_c": 22.0,
    "humidity_pct": 40.0,
    "particle_density": 1000.0,
    "occupancy_count": 1,
    "maintenance_lag_days": 0,
    "airflow_pct": 97.0,
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_high_risk_predicts_anomaly():
    resp = client.post("/predict", json=HIGH_RISK)
    assert resp.status_code == 200
    data = resp.json()
    assert data["anomaly_predicted"] is True


def test_low_risk_predicts_no_anomaly():
    resp = client.post("/predict", json=LOW_RISK)
    assert resp.status_code == 200
    data = resp.json()
    assert data["anomaly_predicted"] is False


def test_risk_band_low():
    resp = client.post("/predict", json=LOW_RISK)
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_band"] == "low"
    assert data["anomaly_probability"] < 0.30


def test_risk_band_high():
    resp = client.post("/predict", json=HIGH_RISK)
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_band"] == "high"
    assert data["anomaly_probability"] >= 0.60


def test_invalid_zone_class():
    payload = {**HIGH_RISK, "zone_class": "omega"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_out_of_range_pressure():
    payload = {**HIGH_RISK, "pressure_delta_pa": 1.0}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_high_risk_score_greater_than_low_risk():
    high_resp = client.post("/predict", json=HIGH_RISK)
    low_resp = client.post("/predict", json=LOW_RISK)
    assert high_resp.status_code == 200
    assert low_resp.status_code == 200
    assert high_resp.json()["anomaly_probability"] > low_resp.json()["anomaly_probability"]
