from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import aiofiles
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from service.schemas import SensorReading, PredictionResponse

app = FastAPI(title="IoT Sensor Anomaly Detection", version="1.0.0")

MODEL_PATH = Path(__file__).parent.parent / "model" / "model.joblib"
WEB_DIR = Path(__file__).parent / "web"

_model = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

@app.get("/", response_class=HTMLResponse)
async def index():
    async with aiofiles.open(WEB_DIR / "index.html", mode="r") as f:
        return await f.read()

@app.get("/health")
def health():
    model_loaded = MODEL_PATH.exists()
    return {"status": "ok", "model_loaded": model_loaded}

@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    model = get_model()
    data = pd.DataFrame([reading.model_dump()])
    prob = float(model.predict_proba(data)[0][1])
    predicted = prob >= 0.5
    if prob < 0.30:
        band = "low"
    elif prob < 0.60:
        band = "medium"
    else:
        band = "high"
    return PredictionResponse(
        anomaly_probability=round(prob, 4),
        anomaly_predicted=predicted,
        risk_band=band,
        threshold=0.5,
    )
