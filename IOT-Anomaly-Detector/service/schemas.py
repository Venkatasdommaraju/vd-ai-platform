from pydantic import BaseModel, Field
from typing import Literal

class SensorReading(BaseModel):
    zone_class: Literal["alpha", "beta", "gamma", "delta"]
    operational_load: Literal["low", "standard", "high"]
    time_period: Literal["day", "night"]
    sensor_mode: Literal["active", "passive", "contact", "mobile"]
    pressure_delta_pa: float = Field(..., ge=5.0, le=25.0)
    temperature_c: float = Field(..., ge=18.0, le=28.0)
    humidity_pct: float = Field(..., ge=30.0, le=70.0)
    particle_density: float = Field(..., ge=500.0, le=20000.0)
    occupancy_count: int = Field(..., ge=0, le=10)
    maintenance_lag_days: int = Field(..., ge=0, le=14)
    airflow_pct: float = Field(..., ge=60.0, le=100.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "zone_class": "alpha",
                "operational_load": "high",
                "time_period": "night",
                "sensor_mode": "active",
                "pressure_delta_pa": 8.0,
                "temperature_c": 26.5,
                "humidity_pct": 62.0,
                "particle_density": 14200.0,
                "occupancy_count": 7,
                "maintenance_lag_days": 11,
                "airflow_pct": 68.0,
            }
        }
    }

class PredictionResponse(BaseModel):
    anomaly_probability: float
    anomaly_predicted: bool
    risk_band: Literal["low", "medium", "high"]
    threshold: float = 0.5
