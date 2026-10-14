from pathlib import Path
import pandas as pd
import xgboost as xgb

BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "models" / "travel_time_model.json"
TRAINING_FILE = BASE_DIR / "data" / "routewise_training.csv"

# Load training data to recreate the exact category mappings
training_df = pd.read_csv(TRAINING_FILE)

categorical_columns = [
    "route_id",
    "trip_headsign",
    "origin_stop_id",
    "destination_stop_id",
    "direction_id",
]

# Recreate the same category mappings used during training
category_maps = {}

for column in categorical_columns:
    categories = training_df[column].astype("category").cat.categories
    category_maps[column] = {
        value: code for code, value in enumerate(categories)
    }

# Load trained XGBoost model
model = xgb.XGBRegressor()
model.load_model(str(MODEL_FILE))

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
    "departure_minutes",
]


def time_to_minutes(time_string):
    parts = str(time_string).strip().split(":")
    h = int(parts[0])
    m = int(parts[1])
    s = float(parts[2]) if len(parts) > 2 else 0.0
    return h * 60 + m + s / 60.0


def predict_travel_time(
    route_id,
    trip_headsign,
    origin_stop_id,
    destination_stop_id,
    direction_id,
    origin_lat,
    origin_lon,
    destination_lat,
    destination_lon,
    stop_sequence,
    departure_time,
):
    values = {
        "route_id": route_id,
        "trip_headsign": trip_headsign,
        "origin_stop_id": origin_stop_id,
        "destination_stop_id": destination_stop_id,
        "direction_id": direction_id,
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "destination_lat": destination_lat,
        "destination_lon": destination_lon,
        "stop_sequence": stop_sequence,
        "departure_minutes": time_to_minutes(departure_time),
    }

    # Convert categorical values using the SAME mappings
    for column in categorical_columns:
        if values[column] not in category_maps[column]:
            # Graceful fallback for unseen category
            values[column] = -1
        else:
            values[column] = category_maps[column][values[column]]

    X = pd.DataFrame([values])[FEATURES]
    prediction = model.predict(X)[0]

    return max(0.5, float(prediction))