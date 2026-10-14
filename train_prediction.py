import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "train_delay_model.joblib"

TRAIN_INFO = {
    "12623": "12623 • MAS TVC SF Mail (Chennai → Thiruvananthapuram)",
    "12624": "12624 • TVC MAS SF Mail (Thiruvananthapuram → Chennai)",
    "16345": "16345 • Netravati Express (LTT Mumbai → Thiruvananthapuram)",
    "16346": "16346 • Netravati Express (Thiruvananthapuram → LTT Mumbai)",
    "16629": "16629 • Malabar Express (Mangaluru → Thiruvananthapuram)",
    "16630": "16630 • Malabar Express (Thiruvananthapuram → Mangaluru)",
}

KERALA_STATION_NAMES = {
    "TVC": "TVC — Thiruvananthapuram Central",
    "QLN": "QLN — Kollam Junction",
    "ALLP": "ALLP — Alappuzha",
    "KTYM": "KTYM — Kottayam",
    "ERS": "ERS — Ernakulam Junction (South)",
    "ERN": "ERN — Ernakulam Town (North)",
    "TCR": "TCR — Thrissur",
    "PGT": "PGT — Palakkad Junction",
    "CLT": "CLT — Kozhikode Main",
    "CAN": "CAN — Kannur",
}

STATION_DEFAULTS = {
    "TVC": {"seq": 32, "dist": 920.0},
    "QLN": {"seq": 28, "dist": 855.0},
    "ALLP": {"seq": 24, "dist": 770.0},
    "KTYM": {"seq": 23, "dist": 760.0},
    "ERS": {"seq": 21, "dist": 700.0},
    "ERN": {"seq": 20, "dist": 698.0},
    "TCR": {"seq": 17, "dist": 625.0},
    "PGT": {"seq": 14, "dist": 550.0},
    "CLT": {"seq": 10, "dist": 435.0},
    "CAN": {"seq": 6, "dist": 345.0},
}


@st.cache_resource
def load_train_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def render_train_prediction():
    st.markdown(
        '<div class="section-title">🚆 Kerala Train Delay & ETA Predictor</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Estimate train arrival delays across major Kerala railway stations using our "
        "trained Machine Learning Random Forest model. Based on historical Indian Railways running records."
    )

    col1, col2 = st.columns(2)

    with col1:
        train_no = st.selectbox(
            "Select Train",
            options=list(TRAIN_INFO.keys()),
            format_func=lambda x: TRAIN_INFO[x],
            key="train_number_input",
        )

        station = st.selectbox(
            "Kerala Station",
            options=list(KERALA_STATION_NAMES.keys()),
            format_func=lambda x: KERALA_STATION_NAMES[x],
            key="train_station_input",
        )

    default_meta = STATION_DEFAULTS.get(station, {"seq": 15, "dist": 500.0})

    with col2:
        scheduled_time = st.time_input(
            "Scheduled Arrival Time",
            value=datetime.time(12, 0),
            key="train_scheduled_time",
        )

        c_seq, c_dist = st.columns(2)
        with c_seq:
            sequence = st.number_input(
                "Station Sequence",
                min_value=1,
                max_value=200,
                value=default_meta["seq"],
                step=1,
                key="train_sequence",
            )
        with c_dist:
            distance = st.number_input(
                "Distance (km)",
                min_value=0.0,
                max_value=5000.0,
                value=float(default_meta["dist"]),
                step=10.0,
                key="train_distance",
            )

    st.markdown("")
    predict_clicked = st.button(
        "⚡ Predict Train Delay & ETA",
        type="primary",
        key="predict_train_delay",
        use_container_width=True,
    )

    if predict_clicked:
        try:
            model = load_train_model()

            scheduled_minutes = scheduled_time.hour * 60 + scheduled_time.minute

            features = pd.DataFrame([{
                "train_no": int(train_no),
                "station_code": station,
                "seq": int(sequence),
                "distance_km": float(distance),
                "scheduled_arrival_minutes": scheduled_minutes,
            }])

            predicted_delay = float(model.predict(features)[0])
            predicted_delay = max(-60.0, min(360.0, predicted_delay))

            estimated_minutes = (scheduled_minutes + predicted_delay) % 1440
            est_h = int(estimated_minutes // 60)
            est_m = int(round(estimated_minutes % 60))
            if est_m == 60:
                est_m = 0
                est_h = (est_h + 1) % 24
            estimated_time = datetime.time(est_h, est_m)

            st.markdown("---")
            st.markdown("### 📊 Prediction Results")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Scheduled Arrival", scheduled_time.strftime("%I:%M %p"))
            with m2:
                delay_str = f"{predicted_delay:+.1f} min"
                st.metric("Predicted Delay", delay_str, delta=f"{predicted_delay:+.1f} min", delta_color="inverse")
            with m3:
                st.metric("ML Estimated Arrival", estimated_time.strftime("%I:%M %p"))
            with m4:
                if predicted_delay <= 5:
                    status_text = "🟢 On Time / Minimal Delay"
                elif predicted_delay <= 25:
                    status_text = "🟡 Moderate Delay Risk"
                else:
                    status_text = "🔴 High Delay Expected"
                st.metric("Risk Status", status_text)

            if predicted_delay > 15:
                st.warning(
                    f"⚠️ **High Delay Alert**: Train {train_no} is estimated to run **{predicted_delay:.1f} minutes behind schedule** "
                    f"at {KERALA_STATION_NAMES[station]}. Plan connecting transit accordingly."
                )
            elif predicted_delay < -1:
                st.info(
                    f"ℹ️ **Early Arrival**: Train is estimated to arrive approximately **{abs(predicted_delay):.1f} minutes ahead of schedule**."
                )
            else:
                st.success(
                    f"✅ **On-Schedule**: Train {train_no} is expected to arrive within **{abs(predicted_delay):.1f} minutes** of scheduled time."
                )

        except FileNotFoundError:
            st.error("Train model artifact not found. Please verify models/train_delay_model.joblib.")
        except Exception as error:
            st.error(f"Error making prediction: {error}")
