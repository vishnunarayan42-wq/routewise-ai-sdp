import datetime
import json
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
import pandas as pd

# Lazy imports for ML models
_model_predictor = None
_metro_planner = None
_train_model = None
_df_cache = None

BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "data" / "routewise_training.csv"
METRO_STOPS_FILE = BASE_DIR / "data" / "kochimetro_gtfs" / "stops.txt"
TRAIN_MODEL_PATH = BASE_DIR / "models" / "train_delay_model.joblib"
METRICS_FILE = BASE_DIR / "models" / "train_model_metrics.json"

app = Flask(__name__, static_folder="../public", static_url_path="")


def get_df():
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(DATA_FILE)
        _df_cache["stop_sequence"] = pd.to_numeric(_df_cache["stop_sequence"], errors="coerce")
        _df_cache["scheduled_travel_minutes"] = pd.to_numeric(
            _df_cache["scheduled_travel_minutes"], errors="coerce"
        )
    return _df_cache


def get_model_predictor():
    global _model_predictor
    if _model_predictor is None:
        import model_predictor
        _model_predictor = model_predictor
    return _model_predictor


def get_metro_planner():
    global _metro_planner
    if _metro_planner is None:
        import metro_planner
        _metro_planner = metro_planner
    return _metro_planner


def get_train_model():
    global _train_model
    if _train_model is None:
        import joblib
        _train_model = joblib.load(TRAIN_MODEL_PATH)
    return _train_model


def time_to_min(val):
    parts = str(val).strip().split(":")
    h = int(parts[0])
    m = int(parts[1])
    s = float(parts[2]) if len(parts) > 2 else 0.0
    return h * 60 + m + s / 60.0


def format_min(minutes):
    minutes = int(round(float(minutes)))
    h = (minutes // 60) % 24
    m = minutes % 60
    suffix = "AM" if h < 12 else "PM"
    disp_h = h % 12 or 12
    return f"{disp_h}:{m:02d} {suffix}"


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "app": "RouteWise AI", "version": "1.0.0"})


@app.route("/api/stops")
def get_stops():
    try:
        df = get_df()
        stops = set(df["origin_stop_name"].dropna().astype(str).str.strip())
        stops.update(df["destination_stop_name"].dropna().astype(str).str.strip())

        if METRO_STOPS_FILE.exists():
            metro_df = pd.read_csv(METRO_STOPS_FILE)
            if "stop_name" in metro_df.columns:
                stops.update(metro_df["stop_name"].dropna().astype(str).str.strip())

        sorted_stops = sorted(s for s in stops if s)
        return jsonify({"stops": sorted_stops, "count": len(sorted_stops)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/plan", methods=["POST"])
def plan_route():
    try:
        data = request.get_json(force=True)
        source = str(data.get("source", "")).strip()
        destination = str(data.get("destination", "")).strip()

        dep_hour = int(data.get("departure_hour", 8))
        dep_min = int(data.get("departure_minute", 30))
        dep_period = str(data.get("departure_period", "AM")).upper()

        dl_hour = int(data.get("deadline_hour", 9))
        dl_min = int(data.get("deadline_minute", 30))
        dl_period = str(data.get("deadline_period", "AM")).upper()

        mode_filter = str(data.get("mode_filter", "All"))

        if not source or not destination:
            return jsonify({"error": "Please provide both source and destination stops."}), 400
        if source.lower() == destination.lower():
            return jsonify({"error": "Starting stop and destination must be different."}), 400

        dep_hour_24 = dep_hour % 12 + (12 if dep_period == "PM" else 0)
        dep_minutes = dep_hour_24 * 60 + dep_min

        dl_hour_24 = dl_hour % 12 + (12 if dl_period == "PM" else 0)
        deadline_minutes = dl_hour_24 * 60 + dl_min

        df = get_df()
        predictor = get_model_predictor()
        metro = get_metro_planner()

        bus_results = []
        if "Bus" in mode_filter or "All" in mode_filter:
            source_trips = set(df.loc[df["origin_stop_name"] == source, "trip_id"].dropna())
            destination_trips = set(df.loc[df["destination_stop_name"] == destination, "trip_id"].dropna())
            candidates = source_trips.intersection(destination_trips)

            for trip_id in candidates:
                trip = df[df["trip_id"] == trip_id].copy()
                if trip.empty:
                    continue
                trip = trip.sort_values("stop_sequence").reset_index(drop=True)
                s_rows = trip[trip["origin_stop_name"] == source]
                d_rows = trip[trip["destination_stop_name"] == destination]

                if s_rows.empty or d_rows.empty:
                    continue

                for _, s_row in s_rows.iterrows():
                    s_seq = s_row["stop_sequence"]
                    if pd.isna(s_seq):
                        continue
                    v_dest = d_rows[d_rows["stop_sequence"] >= s_seq]
                    if v_dest.empty:
                        continue
                    d_row = v_dest.sort_values("stop_sequence").iloc[0]
                    d_seq = d_row["stop_sequence"]
                    if d_seq <= s_seq:
                        continue

                    journey = trip[(trip["stop_sequence"] >= s_seq) & (trip["stop_sequence"] <= d_seq)]
                    if journey.empty:
                        continue

                    total_pred = 0.0
                    failed = False
                    for _, seg in journey.iterrows():
                        sched_m = seg["scheduled_travel_minutes"]
                        try:
                            p = predictor.predict_travel_time(
                                seg["route_id"],
                                seg["trip_headsign"],
                                seg["origin_stop_id"],
                                seg["destination_stop_id"],
                                seg["direction_id"],
                                seg["origin_lat"],
                                seg["origin_lon"],
                                seg["destination_lat"],
                                seg["destination_lon"],
                                seg["stop_sequence"],
                                seg["origin_departure"],
                            )
                            total_pred += float(p)
                        except Exception:
                            failed = True
                            if pd.notna(sched_m):
                                total_pred += float(sched_m)
                            else:
                                total_pred = float("nan")
                                break

                    if pd.isna(total_pred):
                        continue

                    o_dep = s_row["origin_departure"]
                    try:
                        dep_m = time_to_min(o_dep)
                    except Exception:
                        continue

                    if dep_m < dep_minutes:
                        continue

                    arr_m = dep_m + total_pred
                    bus_results.append({
                        "mode": "Bus",
                        "route_name": s_row["route_name"],
                        "trip_id": trip_id,
                        "departure_time": format_min(dep_m),
                        "departure_minutes": dep_m,
                        "travel_minutes": round(total_pred, 1),
                        "arrival_time": format_min(arr_m),
                        "arrival_minutes": arr_m,
                        "before_deadline": arr_m <= deadline_minutes,
                        "basis": "ML Prediction (XGBoost)" if not failed else "Scheduled Fallback",
                    })

        metro_results = []
        if "Metro" in mode_filter or "All" in mode_filter:
            m_df = metro.get_metro_journeys(source, destination)
            if not m_df.empty:
                for _, row in m_df.iterrows():
                    try:
                        dep_m = time_to_min(row["departure"])
                        if dep_m < dep_minutes:
                            continue
                        arr_m = time_to_min(row["arrival"])
                        travel_m = float(row["travel_minutes"])
                        metro_results.append({
                            "mode": "Metro",
                            "route_name": f"Kochi Metro ({row['trip_id']})",
                            "trip_id": row["trip_id"],
                            "departure_time": format_min(dep_m),
                            "departure_minutes": dep_m,
                            "travel_minutes": round(travel_m, 1),
                            "arrival_time": format_min(arr_m),
                            "arrival_minutes": arr_m,
                            "before_deadline": arr_m <= deadline_minutes,
                            "basis": "GTFS Timetable",
                        })
                    except Exception:
                        continue

        all_journeys = bus_results + metro_results
        all_journeys.sort(key=lambda x: (not x["before_deadline"], x["arrival_minutes"]))

        feasible = [j for j in all_journeys if j["before_deadline"]]
        recommended = feasible[0] if feasible else None

        return jsonify({
            "source": source,
            "destination": destination,
            "departure": format_min(dep_minutes),
            "deadline": format_min(deadline_minutes),
            "recommended": recommended,
            "journeys": all_journeys,
            "count": len(all_journeys),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict-train", methods=["POST"])
def predict_train():
    try:
        data = request.get_json(force=True)
        train_no = str(data.get("train_no", "12623"))
        station = str(data.get("station_code", "ERS")).strip().upper()
        h = int(data.get("scheduled_hour", 12))
        m = int(data.get("scheduled_minute", 0))
        seq = int(data.get("sequence", 20))
        dist = float(data.get("distance_km", 700.0))

        sched_min = (h % 24) * 60 + m
        sched_time_str = format_min(sched_min)

        model = get_train_model()
        features = pd.DataFrame([{
            "train_no": int(train_no),
            "station_code": station,
            "seq": int(seq),
            "distance_km": float(dist),
            "scheduled_arrival_minutes": sched_min,
        }])

        pred_delay = float(model.predict(features)[0])
        pred_delay = max(-60.0, min(360.0, pred_delay))

        est_min = (sched_min + pred_delay) % 1440
        est_time_str = format_min(est_min)

        if pred_delay <= 5:
            risk = "🟢 On Time / Minimal Delay"
            alert = "success"
        elif pred_delay <= 25:
            risk = "🟡 Moderate Delay Risk"
            alert = "warning"
        else:
            risk = "🔴 High Delay Expected"
            alert = "danger"

        return jsonify({
            "train_no": train_no,
            "station_code": station,
            "scheduled_time": sched_time_str,
            "predicted_delay_min": round(pred_delay, 1),
            "estimated_time": est_time_str,
            "risk_status": risk,
            "alert_level": alert,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics")
def get_metrics():
    train_metrics = {
        "mae_minutes": 23.01,
        "rmse_minutes": 41.16,
        "r2": 0.172,
        "baseline_mae_minutes": 26.05,
    }
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE, "r") as f:
                train_metrics = json.load(f)
        except Exception:
            pass

    return jsonify({
        "bus_model": {
            "name": "Bus Segment Travel-Time Predictor",
            "algorithm": "XGBoost Regressor",
            "mae_minutes": 0.35,
            "rmse_minutes": 0.54,
            "r2_score": 0.512,
            "training_records": 45621,
        },
        "train_model": {
            "name": "Kerala Train Delay Predictor",
            "algorithm": "Random Forest Regressor (Pipeline)",
            "mae_minutes": train_metrics.get("mae_minutes", 23.01),
            "rmse_minutes": train_metrics.get("rmse_minutes", 41.16),
            "r2_score": train_metrics.get("r2", 0.172),
            "baseline_mae_minutes": train_metrics.get("baseline_mae_minutes", 26.05),
            "training_records": train_metrics.get("training_rows", 30569),
        },
        "network": {
            "bus_routes": 465,
            "metro_stations": 25,
            "train_lines": 6,
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
