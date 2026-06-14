from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

NUMERIC_FEATURES = [
    "pressure_delta_pa",
    "temperature_c",
    "humidity_pct",
    "particle_density",
    "occupancy_count",
    "maintenance_lag_days",
    "airflow_pct",
]

CATEGORICAL_FEATURES = [
    "zone_class",
    "operational_load",
    "time_period",
    "sensor_mode",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "anomaly"

def build_preprocessor():
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
