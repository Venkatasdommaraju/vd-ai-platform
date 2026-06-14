import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N = 6000

zone_classes = rng.choice(["alpha", "beta", "gamma", "delta"], size=N, p=[0.25, 0.25, 0.25, 0.25])
operational_loads = rng.choice(["low", "standard", "high"], size=N, p=[0.30, 0.45, 0.25])
time_periods = rng.choice(["day", "night"], size=N, p=[0.60, 0.40])
sensor_modes = rng.choice(["active", "passive", "contact", "mobile"], size=N, p=[0.35, 0.25, 0.25, 0.15])

pressure_delta_pa = rng.uniform(5, 25, size=N)
temperature_c = rng.uniform(18, 28, size=N)
humidity_pct = rng.uniform(30, 70, size=N)
particle_density = rng.uniform(500, 20000, size=N)
occupancy_count = rng.integers(0, 11, size=N)
maintenance_lag_days = rng.integers(0, 15, size=N)
airflow_pct = rng.uniform(60, 100, size=N)

# Normalise each feature to a 0-1 risk score
# pressure: low pressure = high risk (invert scale 5-25 → 1-0)
pressure_risk = (25 - pressure_delta_pa) / 20.0

# temperature: extremes risky, centre (23 °C) safe
temp_risk = np.abs(temperature_c - 23.0) / 5.0

# humidity: high humidity risky (30-70 → 0-1 at 55+)
humidity_risk = np.clip((humidity_pct - 40) / 30.0, 0, 1)

# particle density: high = risky (500-20000 → 0-1)
particle_risk = (particle_density - 500) / 19500.0

# maintenance lag: high = risky (0-14 → 0-1)
maint_risk = maintenance_lag_days / 14.0

# airflow: low airflow = risky (invert 60-100 → 1-0)
airflow_risk = (100 - airflow_pct) / 40.0

# zone class: alpha most critical
zone_mult = np.where(zone_classes == "alpha", 1.0,
            np.where(zone_classes == "beta", 0.7,
            np.where(zone_classes == "gamma", 0.45, 0.25)))

load_mult = np.where(operational_loads == "high", 1.0,
            np.where(operational_loads == "standard", 0.65, 0.35))

time_mod = np.where(time_periods == "night", 1.15, 1.0)

composite = (
    0.20 * pressure_risk +
    0.08 * temp_risk +
    0.10 * humidity_risk +
    0.25 * particle_risk +
    0.22 * maint_risk +
    0.15 * airflow_risk
)

prob = composite * zone_mult * load_mult * time_mod
prob = np.clip(prob, 0.0, 0.95)

anomaly = rng.binomial(1, prob).astype(int)

df = pd.DataFrame({
    "zone_class": zone_classes,
    "operational_load": operational_loads,
    "time_period": time_periods,
    "sensor_mode": sensor_modes,
    "pressure_delta_pa": np.round(pressure_delta_pa, 2),
    "temperature_c": np.round(temperature_c, 2),
    "humidity_pct": np.round(humidity_pct, 2),
    "particle_density": np.round(particle_density, 1),
    "occupancy_count": occupancy_count,
    "maintenance_lag_days": maintenance_lag_days,
    "airflow_pct": np.round(airflow_pct, 2),
    "anomaly": anomaly,
})

out_dir = Path(__file__).parent / "data"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "anomaly_samples.csv"
df.to_csv(out_path, index=False)

print(f"Dataset saved to {out_path}")
print(f"Total samples: {len(df)}")
print(f"Anomaly rate: {df['anomaly'].mean():.1%}")
print(f"Anomaly count: {df['anomaly'].sum()} / {len(df)}")
print("\nZone class distribution:")
print(df.groupby("zone_class")["anomaly"].mean().round(3).to_string())
print("\nOperational load distribution:")
print(df.groupby("operational_load")["anomaly"].mean().round(3).to_string())
