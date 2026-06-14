---
title: IoT Anomaly Detector
emoji: 📡
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: true
license: mit
short_description: IoT sensor anomaly detection with live risk gauge UI.
---

# IoT Sensor Anomaly Detection

A FastAPI microservice that uses machine learning to predict whether an industrial sensor zone is heading toward an anomaly. Feed it a snapshot of environmental and operational readings — pressure delta, temperature, humidity, particle density, airflow, maintenance lag, and zone metadata — and it returns an anomaly probability, a risk band, and a binary prediction.

The model is a scikit-learn pipeline with logistic regression and gradient boosting trained head-to-head on 6000 synthetic samples with realistic risk relationships. The winner by 5-fold cross-validated ROC-AUC is saved to `model/model.joblib` at train time. The service exposes a single `/predict` endpoint and a browser-based dashboard at `/`.

## Tech Stack

- **ML**: scikit-learn (logistic regression, gradient boosting, StandardScaler, OneHotEncoder)
- **API**: FastAPI + Pydantic v2 + uvicorn
- **Dashboard**: vanilla JS, SVG arc gauge, dark theme
- **Containerization**: Docker (single-stage, Python 3.12-slim)

## Quick Start

```bash
python data_gen.py          # generates data/anomaly_samples.csv
python train.py             # trains model, saves model/model.joblib + metrics
uvicorn service.main:app --reload --port 8000
# open http://localhost:8000
```

## API Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "zone_class": "alpha",
    "operational_load": "high",
    "time_period": "night",
    "sensor_mode": "active",
    "pressure_delta_pa": 7.5,
    "temperature_c": 27.0,
    "humidity_pct": 65.0,
    "particle_density": 17000.0,
    "occupancy_count": 8,
    "maintenance_lag_days": 12,
    "airflow_pct": 63.0
  }'
```

Response:
```json
{"anomaly_probability": 0.9142, "anomaly_predicted": true, "risk_band": "high", "threshold": 0.5}
```

## Deployment

Build and run with Docker:

```bash
docker build -t iot-anomaly-detector .
docker run -p 8000:8000 iot-anomaly-detector
```

The Dockerfile runs `data_gen.py` and `train.py` at image build time, so the container ships with a ready-to-serve model. No external dependencies or database required.
