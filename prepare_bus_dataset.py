import os
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

print("Loading GTFS data...")
routes = pd.read_csv(DATA_DIR / "bus_routes.csv")
trips = pd.read_csv(DATA_DIR / "trips.txt")
stop_times = pd.read_csv(DATA_DIR / "stop_times.txt")
stops = pd.read_csv(DATA_DIR / "stops.txt")

# Filter trips belonging to bus routes
bus_route_ids = set(routes["route_id"].unique())
bus_trips = trips[trips["route_id"].isin(bus_route_ids)].copy()

# Map routes
route_dict = routes.set_index("route_id")["route_long_name"].fillna("").to_dict()
routes_short = routes.set_index("route_id")["route_short_name"].fillna("").to_dict()

# Map stops
stop_info = stops.set_index("stop_id")[["stop_name", "stop_lat", "stop_lon"]].to_dict("index")

def time_to_min(val):
    try:
        parts = str(val).strip().split(":")
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2]) if len(parts) > 2 else 0.0
        return h * 60 + m + s / 60.0
    except Exception:
        return np.nan

print("Building consecutive segments...")
stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

# Group by trip
records = []
trip_groups = stop_times.groupby("trip_id")

trip_metadata = bus_trips.set_index("trip_id").to_dict("index")

for trip_id, group in trip_groups:
    if trip_id not in trip_metadata:
        continue
    meta = trip_metadata[trip_id]
    route_id = meta["route_id"]
    r_name = route_dict.get(route_id) or routes_short.get(route_id) or str(route_id)
    headsign = meta.get("trip_headsign", "")
    direction_id = meta.get("direction_id", 0)

    group_rows = group.to_dict("records")
    for i in range(len(group_rows) - 1):
        s_curr = group_rows[i]
        s_next = group_rows[i + 1]

        o_id = s_curr["stop_id"]
        d_id = s_next["stop_id"]

        if o_id not in stop_info or d_id not in stop_info:
            continue

        o_info = stop_info[o_id]
        d_info = stop_info[d_id]

        o_dep = s_curr["departure_time"]
        d_arr = s_next["arrival_time"]

        t_start = time_to_min(o_dep)
        t_end = time_to_min(d_arr)

        if pd.isna(t_start) or pd.isna(t_end):
            continue

        sched_min = t_end - t_start
        if sched_min <= 0:
            sched_min = 2.0  # reasonable minimum segment default

        # Create realistic travel time variation based on time of day (rush hours)
        # Peak morning: 8:00 - 10:30 (480 - 630 min), peak evening: 17:00 - 20:00 (1020 - 1200 min)
        rush_multiplier = 1.0
        if (480 <= t_start <= 630) or (1020 <= t_start <= 1200):
            rush_multiplier = 1.25 + 0.15 * np.sin((t_start % 180) / 180.0 * np.pi)
        elif 360 <= t_start <= 480 or 1200 <= t_start <= 1320:
            rush_multiplier = 1.10

        # Small stochastic variation
        rng = np.random.default_rng(seed=(i * 1000 + int(t_start)) % 1000000)
        noise = rng.normal(1.0, 0.08)
        actual_min = max(1.0, sched_min * rush_multiplier * noise)

        records.append({
            "trip_id": trip_id,
            "route_id": route_id,
            "route_name": r_name,
            "trip_headsign": headsign,
            "direction_id": int(direction_id) if pd.notna(direction_id) else 0,
            "origin_stop_id": o_id,
            "destination_stop_id": d_id,
            "origin_stop_name": o_info["stop_name"],
            "destination_stop_name": d_info["stop_name"],
            "origin_lat": o_info["stop_lat"],
            "origin_lon": o_info["stop_lon"],
            "destination_lat": d_info["stop_lat"],
            "destination_lon": d_info["stop_lon"],
            "stop_sequence": int(s_curr["stop_sequence"]),
            "origin_departure": o_dep,
            "destination_arrival": d_arr,
            "scheduled_travel_minutes": round(sched_min, 2),
            "actual_travel_minutes": round(actual_min, 2),
            "departure_minutes": round(t_start, 2),
        })

df_training = pd.DataFrame(records)
output_path = DATA_DIR / "routewise_training.csv"
df_training.to_csv(output_path, index=False)
print(f"Saved {len(df_training)} segment records to {output_path}")

# Now train the XGBoost travel time prediction model
print("Training XGBoost travel time model...")
categorical_columns = [
    "route_id",
    "trip_headsign",
    "origin_stop_id",
    "destination_stop_id",
    "direction_id"
]

X_df = df_training.copy()

for col in categorical_columns:
    X_df[col] = X_df[col].astype("category").cat.codes

FEATURES = [
    "route_id",
    "trip_headsign",
    "origin_stop_id",
    "destination_stop_id",
    "direction_id",
    "origin_lat",
    "origin_lon",
    "destination_lat",
    "destination_lon",
    "stop_sequence",
    "departure_minutes"
]

X = X_df[FEATURES]
y = X_df["actual_travel_minutes"]

split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = xgb.XGBRegressor(
    n_estimators=120,
    learning_rate=0.08,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"XGBoost Model Results: MAE = {mae:.3f} min, RMSE = {rmse:.3f} min, R2 = {r2:.3f}")

model_file = MODELS_DIR / "travel_time_model.json"
model.save_model(model_file)
print(f"Saved model to {model_file}")
