
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = Path("data/kerala_train_sample.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE, low_memory=False)

# Prepare target: arrival delay in minutes.
df["delay_arrival_min"] = pd.to_numeric(
    df["delay_arrival_min"], errors="coerce"
)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["seq"] = pd.to_numeric(df["seq"], errors="coerce")
df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")

# Convert scheduled arrival time into minutes after midnight.
def time_to_minutes(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    try:
        parts = value.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        return (hours * 60 + minutes) % (24 * 60)
    except (ValueError, IndexError):
        return np.nan

df["scheduled_arrival_minutes"] = df["sch_arrival"].apply(
    time_to_minutes
)

# Keep only rows with a usable target and date.
df = df.dropna(
    subset=[
        "date",
        "train_no",
        "station_code",
        "seq",
        "scheduled_arrival_minutes",
        "delay_arrival_min",
    ]
).copy()

# Exclude impossible/extreme values only for basic data quality.
df = df[
    df["delay_arrival_min"].between(-180, 720)
].copy()

# Sort chronologically so later dates can be held out for testing.
df = df.sort_values("date").reset_index(drop=True)

features = [
    "train_no",
    "station_code",
    "seq",
    "distance_km",
    "scheduled_arrival_minutes",
]

target = "delay_arrival_min"

# Split by date, not randomly, to test on later dates.
unique_dates = sorted(df["date"].dt.date.unique())

if len(unique_dates) < 2:
    raise ValueError("Not enough distinct dates to make a date-based split.")

split_index = max(1, int(len(unique_dates) * 0.8))
split_index = min(split_index, len(unique_dates) - 1)
test_start_date = unique_dates[split_index]

train_df = df[df["date"].dt.date < test_start_date].copy()
test_df = df[df["date"].dt.date >= test_start_date].copy()

if train_df.empty or test_df.empty:
    raise ValueError("Training or testing data is empty.")

categorical_features = ["train_no", "station_code"]
numeric_features = [
    "seq",
    "distance_km",
    "scheduled_arrival_minutes",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ),
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ),
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

print("Training the train-delay model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Compare against a simple baseline: always predict training-set mean.
baseline_predictions = np.full(len(y_test), y_train.mean())
baseline_mae = mean_absolute_error(y_test, baseline_predictions)

joblib.dump(model, MODEL_DIR / "train_delay_model.joblib", compress=3)

metrics = {
    "training_rows": int(len(train_df)),
    "testing_rows": int(len(test_df)),
    "test_start_date": str(test_start_date),
    "mae_minutes": round(float(mae), 3),
    "rmse_minutes": round(float(rmse), 3),
    "r2": round(float(r2), 3),
    "baseline_mae_minutes": round(float(baseline_mae), 3),
    "trains": int(df["train_no"].nunique()),
    "stations": int(df["station_code"].nunique()),
}

with open(MODEL_DIR / "train_model_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\n--- TRAIN MODEL RESULTS ---")
for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nSaved model: models/train_delay_model.joblib")
print("Saved metrics: models/train_model_metrics.json")
