import csv
import json
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="../public", static_url_path="")

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# In-memory caches for fast serverless responses
_bus_data = None
_bus_model = None
_train_model = None
_metro_data = None


# =========================================================
# HELPER TIME UTILITIES
# =========================================================

def time_to_min(val):
    try:
        parts = str(val).strip().split(":")
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2]) if len(parts) > 2 else 0.0
        return h * 60 + m + s / 60.0
    except Exception:
        return 0.0


def format_min(minutes):
    minutes = int(round(float(minutes)))
    h = (minutes // 60) % 24
    m = minutes % 60
    suffix = "AM" if h < 12 else "PM"
    disp_h = h % 12 or 12
    return f"{disp_h}:{m:02d} {suffix}"


# =========================================================
# PURE PYTHON ML INFERENCE ENGINE
# =========================================================

def load_bus_model():
    global _bus_model
    if _bus_model is not None:
        return _bus_model

    path = MODELS_DIR / "travel_time_model.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    learner = data["learner"]
    base_str = learner["learner_model_param"]["base_score"].strip("[]")
    base_score = float(base_str)
    trees = learner["gradient_booster"]["model"]["trees"]

    # Pre-parse tree arrays for maximum speed
    parsed_trees = []
    for t in trees:
        parsed_trees.append((
            t["left_children"],
            t["right_children"],
            t["split_indices"],
            t["split_conditions"],
            t["base_weights"],
        ))

    _bus_model = {
        "base_score": base_score,
        "trees": parsed_trees,
    }
    return _bus_model


def predict_bus_segment(features):
    m = load_bus_model()
    tot = m["base_score"]
    for lefts, rights, indices, splits, weights in m["trees"]:
        node = 0
        while lefts[node] != -1:
            if features[indices[node]] < splits[node]:
                node = lefts[node]
            else:
                node = rights[node]
        tot += weights[node]
    return max(0.5, tot)


def load_train_model():
    global _train_model
    if _train_model is not None:
        return _train_model

    path = MODELS_DIR / "train_delay_model.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    _train_model = data
    return _train_model


def predict_train_delay(train_no, station_code, seq, distance_km, sched_min):
    m = load_train_model()
    cats = m["categories"]
    scale_mean = m["scale_mean"]
    scale_scale = m["scale_scale"]
    trees = m["trees"]

    vec = [0.0] * 19
    t_str = str(train_no)
    if t_str in cats[0]:
        vec[cats[0].index(t_str)] = 1.0
    s_str = str(station_code).upper()
    if s_str in cats[1]:
        vec[len(cats[0]) + cats[1].index(s_str)] = 1.0

    offset = len(cats[0]) + len(cats[1])
    vec[offset + 0] = (float(seq) - scale_mean[0]) / scale_scale[0]
    vec[offset + 1] = (float(distance_km) - scale_mean[1]) / scale_scale[1]
    vec[offset + 2] = (float(sched_min) - scale_mean[2]) / scale_scale[2]

    tot = 0.0
    for tr in trees:
        node = 0
        lefts = tr["l"]
        rights = tr["r"]
        features = tr["f"]
        thresh = tr["t"]
        vals = tr["v"]
        while lefts[node] != -1:
            if vec[features[node]] <= thresh[node]:
                node = lefts[node]
            else:
                node = rights[node]
        tot += vals[node]
    return tot / len(trees)


# =========================================================
# DATA LOADERS & INDEXING (ZERO DEPENDENCY)
# =========================================================

def load_bus_data():
    global _bus_data
    if _bus_data is not None:
        return _bus_data

    path = DATA_DIR / "routewise_training.csv"
    trips_map = {}
    source_to_trips = {}
    dest_to_trips = {}
    all_stops = set()

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row["trip_id"]
            if tid not in trips_map:
                trips_map[tid] = []

            orig = row["origin_stop_name"].strip()
            dest = row["destination_stop_name"].strip()
            all_stops.add(orig)
            all_stops.add(dest)

            if orig not in source_to_trips:
                source_to_trips[orig] = set()
            source_to_trips[orig].add(tid)

            if dest not in dest_to_trips:
                dest_to_trips[dest] = set()
            dest_to_trips[dest].add(tid)

            trips_map[tid].append({
                "trip_id": tid,
                "route_id": row["route_id"],
                "route_name": row["route_name"],
                "trip_headsign": row["trip_headsign"],
                "direction_id": int(float(row["direction_id"] or 0)),
                "origin_stop_id": row["origin_stop_id"],
                "destination_stop_id": row["destination_stop_id"],
                "origin_stop_name": orig,
                "destination_stop_name": dest,
                "origin_lat": float(row["origin_lat"] or 0),
                "origin_lon": float(row["origin_lon"] or 0),
                "destination_lat": float(row["destination_lat"] or 0),
                "destination_lon": float(row["destination_lon"] or 0),
                "stop_sequence": int(float(row["stop_sequence"] or 0)),
                "origin_departure": row["origin_departure"],
                "destination_arrival": row["destination_arrival"],
                "scheduled_travel_minutes": float(row["scheduled_travel_minutes"] or 2.0),
            })

    # Sort segments by stop_sequence
    for tid in trips_map:
        trips_map[tid].sort(key=lambda s: s["stop_sequence"])

    _bus_data = {
        "trips_map": trips_map,
        "source_to_trips": source_to_trips,
        "dest_to_trips": dest_to_trips,
        "stops": all_stops,
    }
    return _bus_data


def load_metro_data():
    global _metro_data
    if _metro_data is not None:
        return _metro_data

    metro_dir = DATA_DIR / "kochimetro_gtfs"
    stops_file = metro_dir / "stops.txt"
    st_file = metro_dir / "stop_times.txt"

    stops = []
    metro_stop_names = set()
    with open(stops_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s_name = row["stop_name"].strip()
            stops.append((row["stop_id"], s_name))
            metro_stop_names.add(s_name)

    stop_times_by_trip = {}
    with open(st_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = row["trip_id"]
            if tid not in stop_times_by_trip:
                stop_times_by_trip[tid] = []
            stop_times_by_trip[tid].append({
                "stop_id": row["stop_id"],
                "seq": int(row["stop_sequence"]),
                "arr": row["arrival_time"],
                "dep": row["departure_time"],
            })

    for tid in stop_times_by_trip:
        stop_times_by_trip[tid].sort(key=lambda x: x["seq"])

    _metro_data = {
        "stops": stops,
        "metro_stop_names": metro_stop_names,
        "stop_times": stop_times_by_trip,
    }
    return _metro_data


def get_metro_journeys(source, destination):
    aliases = {
        "edappally": "Edapally",
        "edappally toll": "Edapally",
        "maharajas college": "Maharaja's College",
        "ernakulam south": "Ernakulam South",
        "ernakulam north": "Lissie",
        "kaloor bus stand": "Kaloor",
        "vyttila hub": "Vyttila",
        "aluva bus stand": "Aluva",
    }
    s_clean = aliases.get(source.strip().lower(), source.strip())
    d_clean = aliases.get(destination.strip().lower(), destination.strip())

    m_data = load_metro_data()
    s_ids = {s_id for s_id, name in m_data["stops"] if s_clean.lower() in name.lower()}
    d_ids = {s_id for s_id, name in m_data["stops"] if d_clean.lower() in name.lower()}

    if not s_ids or not d_ids:
        return []

    journeys = []
    for tid, st_list in m_data["stop_times"].items():
        s_match = [x for x in st_list if x["stop_id"] in s_ids]
        d_match = [x for x in st_list if x["stop_id"] in d_ids]

        if not s_match or not d_match:
            continue

        for start in s_match:
            laters = [x for x in d_match if x["seq"] > start["seq"]]
            if not laters:
                continue
            end = laters[0]
            t_dep = time_to_min(start["dep"])
            t_arr = time_to_min(end["arr"])
            travel_mins = round(t_arr - t_dep, 1)

            journeys.append({
                "trip_id": tid,
                "source": source,
                "destination": destination,
                "departure": start["dep"],
                "departure_minutes": t_dep,
                "arrival": end["arr"],
                "arrival_minutes": t_arr,
                "travel_minutes": travel_mins,
            })
    return journeys


# =========================================================
# FLASK HTTP ROUTES
# =========================================================

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "app": "RouteWise AI", "version": "1.0.0"})


@app.route("/api/stops")
def get_stops():
    try:
        b_data = load_bus_data()
        m_data = load_metro_data()
        all_s = sorted(b_data["stops"].union(m_data["metro_stop_names"]))
        return jsonify({"stops": all_s, "count": len(all_s)})
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

        b_data = load_bus_data()
        bus_results = []

        if "Bus" in mode_filter or "All" in mode_filter:
            s_trips = b_data["source_to_trips"].get(source, set())
            d_trips = b_data["dest_to_trips"].get(destination, set())
            candidates = s_trips.intersection(d_trips)

            for tid in candidates:
                segs = b_data["trips_map"].get(tid, [])
                # Find origin and destination indices in trip
                o_idx = None
                d_idx = None
                for idx, seg in enumerate(segs):
                    if seg["origin_stop_name"] == source and o_idx is None:
                        o_idx = idx
                    if seg["destination_stop_name"] == destination and o_idx is not None:
                        d_idx = idx
                        break

                if o_idx is None or d_idx is None or d_idx < o_idx:
                    continue

                sub_journey = segs[o_idx:d_idx + 1]
                if not sub_journey:
                    continue

                first_seg = sub_journey[0]
                dep_m = time_to_min(first_seg["origin_departure"])
                if dep_m < dep_minutes:
                    continue

                total_pred = 0.0
                failed = False
                for seg in sub_journey:
                    try:
                        feat = [
                            hash(seg["route_id"]) % 1000,
                            hash(seg["trip_headsign"]) % 1000,
                            hash(seg["origin_stop_id"]) % 1000,
                            hash(seg["destination_stop_id"]) % 1000,
                            seg["direction_id"],
                            seg["origin_lat"],
                            seg["origin_lon"],
                            seg["destination_lat"],
                            seg["destination_lon"],
                            seg["stop_sequence"],
                            dep_m,
                        ]
                        pred = predict_bus_segment(feat)
                        total_pred += pred
                    except Exception:
                        failed = True
                        total_pred += seg["scheduled_travel_minutes"]

                arr_m = dep_m + total_pred
                bus_results.append({
                    "mode": "Bus",
                    "route_name": first_seg["route_name"],
                    "trip_id": tid,
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
            m_list = get_metro_journeys(source, destination)
            for m in m_list:
                if m["departure_minutes"] < dep_minutes:
                    continue
                metro_results.append({
                    "mode": "Metro",
                    "route_name": f"Kochi Metro ({m['trip_id']})",
                    "trip_id": m["trip_id"],
                    "departure_time": format_min(m["departure_minutes"]),
                    "departure_minutes": m["departure_minutes"],
                    "travel_minutes": m["travel_minutes"],
                    "arrival_time": format_min(m["arrival_minutes"]),
                    "arrival_minutes": m["arrival_minutes"],
                    "before_deadline": m["arrival_minutes"] <= deadline_minutes,
                    "basis": "GTFS Timetable",
                })

        all_j = bus_results + metro_results
        all_j.sort(key=lambda x: (not x["before_deadline"], x["arrival_minutes"]))

        feasible = [j for j in all_j if j["before_deadline"]]
        recommended = feasible[0] if feasible else None

        return jsonify({
            "source": source,
            "destination": destination,
            "departure": format_min(dep_minutes),
            "deadline": format_min(deadline_minutes),
            "recommended": recommended,
            "journeys": all_j,
            "count": len(all_j),
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

        pred_delay = predict_train_delay(train_no, station, seq, dist, sched_min)
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
    return jsonify({
        "bus_model": {
            "name": "Bus Segment Travel-Time Predictor",
            "algorithm": "XGBoost Regressor (120 Trees)",
            "mae_minutes": 0.35,
            "rmse_minutes": 0.54,
            "r2_score": 0.512,
            "training_records": 45621,
        },
        "train_model": {
            "name": "Kerala Train Delay Predictor",
            "algorithm": "Random Forest Regressor (30 Trees, Pipeline)",
            "mae_minutes": 21.06,
            "rmse_minutes": 38.45,
            "r2_score": 0.290,
            "baseline_mae_minutes": 26.05,
            "training_records": 30569,
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
