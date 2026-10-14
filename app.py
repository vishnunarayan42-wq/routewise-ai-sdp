import datetime
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from model_predictor import predict_travel_time
from metro_planner import get_metro_journeys
from train_prediction import render_train_prediction


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RouteWise AI • Smart Transit Planner & ETA Prediction",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "routewise_training.csv"
METRO_STOPS_FILE = BASE_DIR / "data" / "kochimetro_gtfs" / "stops.txt"
METRICS_FILE = BASE_DIR / "models" / "train_model_metrics.json"


# =========================================================
# CSS — MODERN ROUTEWISE DESIGN
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f4f7fc;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1240px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 32px 36px;
        border-radius: 20px;
        background: linear-gradient(135deg, #0d253f, #134e5e, #1a7a8a);
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(13, 37, 63, 0.15);
    }

    .hero-label {
        color: #9fe2ec;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero h1 {
        color: white;
        font-size: clamp(28px, 4.5vw, 42px);
        font-weight: 800;
        margin: 8px 0 10px 0;
    }

    .hero p {
        color: #e3f5f8;
        font-size: 16px;
        margin-bottom: 0;
        max-width: 850px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 750;
        color: #16324f;
        margin-top: 14px;
        margin-bottom: 8px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #028090, #00a896);
        color: white;
        border: none;
        border-radius: 12px;
        min-height: 48px;
        font-size: 16px;
        font-weight: 700;
        transition: 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #026875;
        border: none;
        color: white;
    }

    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e1e9f0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
    }

    .badge-success {
        background: #e6f7ed;
        color: #0e8a4a;
    }

    .badge-warning {
        background: #fff8e6;
        color: #b26a00;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. Please run prepare_bus_dataset.py."
        )

    data = pd.read_csv(DATA_FILE)

    required_columns = [
        "trip_id",
        "route_id",
        "route_name",
        "trip_headsign",
        "direction_id",
        "origin_stop_id",
        "destination_stop_id",
        "origin_stop_name",
        "destination_stop_name",
        "origin_lat",
        "origin_lon",
        "destination_lat",
        "destination_lon",
        "stop_sequence",
        "origin_departure",
        "destination_arrival",
        "scheduled_travel_minutes",
    ]

    missing = [c for c in required_columns if c not in data.columns]
    if missing:
        raise ValueError(
            "The training dataset is missing required columns: " + ", ".join(missing)
        )

    data["stop_sequence"] = pd.to_numeric(data["stop_sequence"], errors="coerce")
    data["scheduled_travel_minutes"] = pd.to_numeric(
        data["scheduled_travel_minutes"], errors="coerce"
    )

    return data


try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load the transit dataset: {error}")
    st.stop()


# =========================================================
# TIME FUNCTIONS
# =========================================================

def time_to_minutes(time_string):
    """Convert a GTFS time such as 08:30:00 to minutes."""
    if pd.isna(time_string):
        raise ValueError("Time is missing.")

    parts = str(time_string).strip().split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid time format: {time_string}")

    hour = int(parts[0])
    minute = int(parts[1])
    return hour * 60 + minute


def format_time(minutes):
    """Display minutes after midnight as a 12-hour time."""
    minutes = int(round(float(minutes)))
    hour = (minutes // 60) % 24
    minute = minutes % 60
    suffix = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour}:{minute:02d} {suffix}"


def format_schedule_time(time_string):
    """Safely format a scheduled time."""
    try:
        return format_time(time_to_minutes(time_string))
    except (ValueError, TypeError, AttributeError):
        return str(time_string)


def parse_departure_minutes(time_string):
    """Convert a scheduled departure to minutes, or NaN."""
    try:
        return time_to_minutes(time_string)
    except (ValueError, TypeError, AttributeError):
        return float("nan")


# =========================================================
# HEADER — ROUTEWISE AI
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">Intelligent Mobility • Kochi & Kerala Transit</div>
        <h1>🚌 RouteWise AI</h1>
        <p>
            AI-based multi-modal transit planner comparing Kochi Buses and Metro 
            with Machine Learning ETA predictions and Kerala railway delay forecasting.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar with quick project context
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?w=500&auto=format&fit=crop&q=60",
        caption="RouteWise AI Smart Mobility",
        use_container_width=True,
    )
    st.markdown("### 📌 Navigation & Tools")
    st.markdown(
        """
        - **Kochi Journey Planner**: Multi-modal bus + metro search with ML travel time.
        - **Kerala Train Delays**: Historical delay prediction for major express trains.
        - **Model Analytics**: Evaluation metrics (MAE, RMSE, R²).
        - **Project Overview**: System architecture & EEE project details.
        """
    )
    st.markdown("---")
    st.markdown(
        "**Project By:**\n"
        "- Vishnu Narayan A B (Roll 61)\n"
        "- Vivek Noorudheen (Roll 62)\n\n"
        "**Degree:** B.Tech EEE • ML Project"
    )


# =========================================================
# TOP TABS
# =========================================================

tab_planner, tab_trains, tab_analytics, tab_about = st.tabs([
    "🚌 🚇 Kochi Journey Planner",
    "🚆 Kerala Train Delays",
    "📊 Model Analytics",
    "ℹ️ About RouteWise",
])


# =========================================================
# TAB 1: KOCHI JOURNEY PLANNER
# =========================================================

with tab_planner:
    # Build complete stop list
    all_stops = set(df["origin_stop_name"].dropna().astype(str).str.strip())
    all_stops.update(df["destination_stop_name"].dropna().astype(str).str.strip())

    if METRO_STOPS_FILE.exists():
        try:
            metro_stops_df = pd.read_csv(METRO_STOPS_FILE)
            if "stop_name" in metro_stops_df.columns:
                all_stops.update(
                    metro_stops_df["stop_name"].dropna().astype(str).str.strip()
                )
        except Exception:
            pass

    all_stops = sorted(s for s in all_stops if s)

    st.markdown(
        '<div class="section-title">📍 Plan Your Journey in Kochi</div>',
        unsafe_allow_html=True,
    )

    # Preset quick-test buttons for common commutes
    st.caption("Quick Select Common Commutes:")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    preset_source = None
    preset_destination = None

    if p_col1.button("Aluva ➔ Edapally", use_container_width=True):
        preset_source = "Aluva"
        preset_destination = "Edapally"
    if p_col2.button("Kalamassery ➔ MG Road", use_container_width=True):
        preset_source = "Kalamassery"
        preset_destination = "M.G Road"
    if p_col3.button("Kaloor ➔ Vyttila", use_container_width=True):
        preset_source = "Kaloor"
        preset_destination = "Vyttila"
    if p_col4.button("Edapally ➔ Maharajas", use_container_width=True):
        preset_source = "Edapally"
        preset_destination = "Maharaja's College"

    col1, col2 = st.columns(2)

    with col1:
        default_src_idx = all_stops.index(preset_source) if preset_source in all_stops else None
        source = st.selectbox(
            "Starting Stop / Station (From)",
            options=all_stops,
            index=default_src_idx,
            placeholder="Type or select starting stop...",
            key="source_stop",
        )

    with col2:
        default_dst_idx = all_stops.index(preset_destination) if preset_destination in all_stops else None
        destination = st.selectbox(
            "Destination Stop / Station (To)",
            options=all_stops,
            index=default_dst_idx,
            placeholder="Type or select destination stop...",
            key="destination_stop",
        )

    time_col1, time_col2 = st.columns(2)

    with time_col1:
        st.markdown("##### 🕒 Departure Time")
        dep_c1, dep_c2, dep_c3 = st.columns(3)
        with dep_c1:
            departure_hour = st.selectbox(
                "Hour",
                options=list(range(1, 13)),
                index=7,
                key="departure_hour",
            )
        with dep_c2:
            departure_minute = st.selectbox(
                "Minute",
                options=list(range(60)),
                index=30,
                format_func=lambda m: f"{m:02d}",
                key="departure_minute",
            )
        with dep_c3:
            departure_period = st.selectbox(
                "AM/PM",
                options=["AM", "PM"],
                index=0,
                key="departure_period",
            )

    with time_col2:
        st.markdown("##### ⏰ Required Arrival Deadline")
        dl_c1, dl_c2, dl_c3 = st.columns(3)
        with dl_c1:
            deadline_hour = st.selectbox(
                "Deadline Hour",
                options=list(range(1, 13)),
                index=9,
                key="deadline_hour",
            )
        with dl_c2:
            deadline_minute = st.selectbox(
                "Deadline Minute",
                options=list(range(60)),
                index=30,
                format_func=lambda m: f"{m:02d}",
                key="deadline_minute",
            )
        with dl_c3:
            deadline_period = st.selectbox(
                "Deadline AM/PM",
                options=["AM", "PM"],
                index=0,
                key="deadline_period",
            )

    mode_filter = st.radio(
        "Transit Filter",
        ["All Modes (Compare Bus & Metro)", "🚌 Bus Only", "🚇 Metro Only"],
        horizontal=True,
    )

    # Calculate 24-hr minutes
    dl_hour_24 = deadline_hour % 12 + (12 if deadline_period == "PM" else 0)
    arrival_deadline = datetime.time(dl_hour_24, deadline_minute)
    deadline_minutes = dl_hour_24 * 60 + deadline_minute

    dep_hour_24 = departure_hour % 12 + (12 if departure_period == "PM" else 0)
    departure_time = datetime.time(dep_hour_24, departure_minute)
    departure_minutes = dep_hour_24 * 60 + departure_minute

    # Complete bus journey search helper
    def find_complete_bus_trips(src, dst):
        results = []
        source_trips = set(df.loc[df["origin_stop_name"] == src, "trip_id"].dropna())
        destination_trips = set(df.loc[df["destination_stop_name"] == dst, "trip_id"].dropna())
        candidate_trip_ids = source_trips.intersection(destination_trips)

        for trip_id in candidate_trip_ids:
            trip = df[df["trip_id"] == trip_id].copy()
            if trip.empty:
                continue

            trip = trip.sort_values("stop_sequence").reset_index(drop=True)
            source_rows = trip[trip["origin_stop_name"] == src]
            destination_rows = trip[trip["destination_stop_name"] == dst]

            if source_rows.empty or destination_rows.empty:
                continue

            for _, source_row in source_rows.iterrows():
                src_seq = source_row["stop_sequence"]
                if pd.isna(src_seq):
                    continue

                valid_destinations = destination_rows[destination_rows["stop_sequence"] >= src_seq]
                if valid_destinations.empty:
                    continue

                dest_row = valid_destinations.sort_values("stop_sequence").iloc[0]
                dest_seq = dest_row["stop_sequence"]
                if dest_seq <= src_seq:
                    continue

                journey = trip[
                    (trip["stop_sequence"] >= src_seq)
                    & (trip["stop_sequence"] <= dest_seq)
                ].copy()

                if journey.empty:
                    continue

                total_predicted_minutes = 0.0
                prediction_failed = False

                for _, segment in journey.iterrows():
                    scheduled_minutes = segment["scheduled_travel_minutes"]
                    try:
                        predicted = predict_travel_time(
                            segment["route_id"],
                            segment["trip_headsign"],
                            segment["origin_stop_id"],
                            segment["destination_stop_id"],
                            segment["direction_id"],
                            segment["origin_lat"],
                            segment["origin_lon"],
                            segment["destination_lat"],
                            segment["destination_lon"],
                            segment["stop_sequence"],
                            segment["origin_departure"],
                        )
                        total_predicted_minutes += float(predicted)
                    except Exception:
                        prediction_failed = True
                        if pd.notna(scheduled_minutes):
                            total_predicted_minutes += float(scheduled_minutes)
                        else:
                            total_predicted_minutes = float("nan")
                            break

                if pd.isna(total_predicted_minutes):
                    continue

                departure_string = source_row["origin_departure"]
                try:
                    dep_mins_trip = time_to_minutes(departure_string)
                except Exception:
                    continue

                predicted_arrival_minutes = dep_mins_trip + total_predicted_minutes

                results.append({
                    "trip_id": trip_id,
                    "route_id": source_row["route_id"],
                    "route_name": source_row["route_name"],
                    "trip_headsign": source_row["trip_headsign"],
                    "from": src,
                    "to": dst,
                    "departure": departure_string,
                    "departure_minutes": dep_mins_trip,
                    "predicted_minutes": total_predicted_minutes,
                    "predicted_arrival_minutes": predicted_arrival_minutes,
                    "predicted_arrival": format_time(predicted_arrival_minutes),
                    "scheduled_arrival": dest_row["destination_arrival"],
                    "segments": len(journey),
                    "prediction_failed": prediction_failed,
                })

        if not results:
            return pd.DataFrame()

        res_df = pd.DataFrame(results).drop_duplicates(subset=["trip_id", "departure"])
        return res_df.sort_values("predicted_arrival_minutes").reset_index(drop=True)

    st.markdown("")
    search_clicked = st.button(
        "🔎 Find Best Routes & Compare ETA",
        type="primary",
        use_container_width=True,
    )

    if search_clicked:
        if not source or not destination:
            st.warning("Please specify both starting stop (From) and destination stop (To).")
        elif source == destination:
            st.warning("Starting stop and destination must be different.")
        else:
            with st.spinner("Analyzing bus networks and metro schedules with ML models..."):
                bus_results = pd.DataFrame()
                metro_results = pd.DataFrame()

                if "Bus" in mode_filter or "All" in mode_filter:
                    try:
                        bus_results = find_complete_bus_trips(source, destination)
                        if not bus_results.empty:
                            bus_results["departure_minutes"] = bus_results["departure"].apply(
                                parse_departure_minutes
                            )
                            bus_results = bus_results[
                                bus_results["departure_minutes"] >= departure_minutes
                            ].copy()
                            bus_results["before_deadline"] = (
                                bus_results["predicted_arrival_minutes"] <= deadline_minutes
                            )
                    except Exception as err:
                        st.error(f"Bus routing error: {err}")

                if "Metro" in mode_filter or "All" in mode_filter:
                    try:
                        metro_results = get_metro_journeys(source, destination)
                        if not metro_results.empty:
                            metro_results["departure_minutes"] = metro_results["departure"].apply(
                                parse_departure_minutes
                            )
                            metro_results = metro_results[
                                metro_results["departure_minutes"] >= departure_minutes
                            ].copy()
                    except Exception as err:
                        st.warning(f"Metro lookup error: {err}")

            # Assemble comparisons
            comparison_rows = []

            if not bus_results.empty:
                for _, bus in bus_results.iterrows():
                    comparison_rows.append({
                        "Mode": "Bus",
                        "Route / Trip": bus.get("route_name", "Bus"),
                        "Departure": format_schedule_time(bus["departure"]),
                        "Journey Time (min)": round(float(bus["predicted_minutes"]), 1),
                        "Arrival": bus["predicted_arrival"],
                        "Arrival Minutes": float(bus["predicted_arrival_minutes"]),
                        "Time Basis": "ML Prediction (XGBoost)" if not bus.get("prediction_failed") else "Scheduled Fallback",
                    })

            if not metro_results.empty:
                for _, metro in metro_results.iterrows():
                    try:
                        metro_arr_mins = time_to_minutes(metro["arrival"])
                        travel_mins = float(metro["travel_minutes"])
                        comparison_rows.append({
                            "Mode": "Metro",
                            "Route / Trip": f"Kochi Metro ({metro['trip_id']})",
                            "Departure": format_schedule_time(metro["departure"]),
                            "Journey Time (min)": round(travel_mins, 1),
                            "Arrival": format_schedule_time(metro["arrival"]),
                            "Arrival Minutes": metro_arr_mins,
                            "Time Basis": "GTFS Timetable",
                        })
                    except Exception:
                        continue

            st.markdown("---")
            st.markdown("### 📋 Journey Comparison Results")

            if comparison_rows:
                comp_df = pd.DataFrame(comparison_rows)
                comp_df["Before Deadline"] = comp_df["Arrival Minutes"] <= deadline_minutes
                comp_df = comp_df.sort_values(
                    ["Before Deadline", "Arrival Minutes"],
                    ascending=[False, True]
                ).reset_index(drop=True)

                feasible_df = comp_df[comp_df["Before Deadline"]]

                if not feasible_df.empty:
                    top = feasible_df.iloc[0]
                    st.success(
                        f"⭐ **Optimal Recommendation:** Take the **{top['Mode']}** — *{top['Route / Trip']}*. "
                        f"Departs at **{top['Departure']}**, arriving at **{top['Arrival']}** "
                        f"({top['Journey Time (min)']} mins). Meets your {arrival_deadline.strftime('%I:%M %p')} deadline!"
                    )
                else:
                    st.warning(
                        f"⚠️ None of the available trips are expected to arrive before your deadline "
                        f"({arrival_deadline.strftime('%I:%M %p')}). Consider an earlier departure."
                    )

                # Cards layout
                for idx, journey in comp_df.iterrows():
                    is_top = (not feasible_df.empty and journey["Arrival Minutes"] == feasible_df.iloc[0]["Arrival Minutes"] and journey["Before Deadline"])
                    mode = str(journey["Mode"])
                    icon = "🚌" if mode == "Bus" else "🚇"

                    with st.container(border=True):
                        c_left, c_right = st.columns([3, 1])
                        with c_left:
                            if is_top:
                                st.markdown("#### ⭐ Recommended Journey")
                            else:
                                st.markdown(f"#### {icon} {mode} Service")

                            st.write(f"**Route:** {journey['Route / Trip']}")
                            st.write(f"**Departure:** {journey['Departure']} ➔ **Arrival:** {journey['Arrival']}")
                            st.caption(f"Duration: **{journey['Journey Time (min)']} min** • Basis: {journey['Time Basis']}")

                        with c_right:
                            st.metric("Estimated Arrival", str(journey["Arrival"]))
                            if bool(journey["Before Deadline"]):
                                st.markdown('<span class="badge-pill badge-success">✓ On Deadline</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="badge-pill badge-warning">⚠ Late Arrival</span>', unsafe_allow_html=True)

                # Show detailed table if bus options exist
                if not bus_results.empty:
                    with st.expander("🔍 Detailed Bus Segments & Predictions Table"):
                        display_b = bus_results[[
                            "route_name", "departure", "predicted_arrival", 
                            "predicted_minutes", "scheduled_arrival", "segments", "before_deadline"
                        ]].copy()
                        display_b["departure"] = display_b["departure"].apply(format_schedule_time)
                        display_b["scheduled_arrival"] = display_b["scheduled_arrival"].apply(format_schedule_time)
                        display_b.columns = [
                            "Route", "Scheduled Dep", "ML-Predicted Arrival",
                            "ML Travel Time (min)", "GTFS Arrival", "Stops", "Meets Deadline"
                        ]
                        st.dataframe(display_b, use_container_width=True, hide_index=True)

            else:
                st.info(
                    f"No direct scheduled trips found connecting **{source}** to **{destination}** "
                    f"at or after {departure_time.strftime('%I:%M %p')}. "
                    "Try selecting major hubs such as *Aluva*, *Edapally*, *Kalamassery*, or *MG Road*."
                )


# =========================================================
# TAB 2: KERALA TRAIN DELAYS
# =========================================================

with tab_trains:
    render_train_prediction()


# =========================================================
# TAB 3: MODEL ANALYTICS
# =========================================================

with tab_analytics:
    st.markdown('<div class="section-title">📊 Machine Learning Models & Metrics</div>', unsafe_allow_html=True)
    st.caption("Performance evaluation and historical dataset summaries for the transit prediction models.")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("### 🚌 Bus Travel-Time Model (XGBoost)")
        st.markdown(
            """
            - **Algorithm:** Extreme Gradient Boosting (XGBoost Regressor)
            - **Target:** Segment travel time in minutes (`actual_travel_minutes`)
            - **Features:** `route_id`, `trip_headsign`, `origin_stop_id`, `destination_stop_id`, `direction_id`, `origin_lat`, `origin_lon`, `destination_lat`, `destination_lon`, `stop_sequence`, `departure_minutes`
            """
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", "0.35 min")
        c2.metric("RMSE", "0.54 min")
        c3.metric("R² Score", "0.512")

    with col_m2:
        st.markdown("### 🚆 Kerala Train Delay Model (Random Forest)")
        train_metrics = {
            "mae_minutes": 23.01,
            "rmse_minutes": 41.16,
            "r2": 0.172,
            "baseline_mae_minutes": 26.05,
            "training_rows": 30569,
            "testing_rows": 10672,
        }
        if METRICS_FILE.exists():
            try:
                with open(METRICS_FILE, "r") as f:
                    train_metrics = json.load(f)
            except Exception:
                pass

        st.markdown(
            f"""
            - **Algorithm:** Random Forest Regressor (150 trees, Pipeline with One-Hot & Scaling)
            - **Target:** Arrival delay in minutes (`delay_arrival_min`)
            - **Training Records:** {train_metrics.get('training_rows', 30569):,} rows across 10 Kerala stations
            """
        )
        t1, t2, t3 = st.columns(3)
        t1.metric("Model MAE", f"{train_metrics.get('mae_minutes', 23.01)} min")
        t2.metric("Baseline MAE", f"{train_metrics.get('baseline_mae_minutes', 26.05)} min")
        t3.metric("R² Score", f"{train_metrics.get('r2', 0.172)}")

    st.markdown("---")
    st.markdown("### 🗄️ Transit Dataset Summary")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Kochi Bus Routes", "465 routes")
    s2.metric("Trained Bus Segments", "45,621 segments")
    s3.metric("Kochi Metro Stations", "25 stations")
    s4.metric("Kerala Train Runs", "41,304 records")


# =========================================================
# TAB 4: ABOUT ROUTEWISE AI
# =========================================================

with tab_about:
    st.markdown('<div class="section-title">ℹ️ About RouteWise AI</div>', unsafe_allow_html=True)
    st.markdown(
        """
        ### 🎯 Problem Statement
        Passengers frequently must reach a destination (e.g. Cochin International Airport, railway terminals, 
        or office hubs) within strict time limits. While transit timetables show scheduled times, 
        actual arrival times fluctuate due to traffic congestion, bottlenecks, weather, and operational delays.
        RouteWise AI solves this by predicting realistic travel times and optimizing route choice against 
        deadline requirements.

        ### ⚙️ System Architecture
        1. **Data Ingestion & GTFS Pipeline:**
           - Processes Kochi Open GTFS feeds (buses, metro schedules, and stops).
           - Fetches historical Indian Railways Kerala running logs via Hugging Face.
        2. **Machine Learning Layer:**
           - **XGBoost Regressor:** Predicts segment travel times accounting for rush hours and time of day.
           - **Random Forest Pipeline:** Predicts train delays across Kerala stations.
        3. **Multi-Modal Route Optimization:**
           - Identifies candidate bus routes and Kochi Metro services.
           - Compares predicted arrival against the user's hard deadline to select the optimal option.
        4. **Interactive Dashboard:**
           - Streamlit-powered responsive web interface for desktop and mobile commuters.

        ### 👥 Project Contributors
        - **Vishnu Narayan A B** (Roll No. 61)
        - **Vivek Noorudheen** (Roll No. 62)
        - *Department of Electrical and Electronics Engineering (EEE)*
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown(
    "<center><small style='color: #64748b;'>"
    "RouteWise AI • AI-Based Bus & Train ETA Prediction and Route Optimization<br>"
    "Developed by Vishnu Narayan A B & Vivek Noorudheen | B.Tech EEE"
    "</small></center>",
    unsafe_allow_html=True,
)
