import os
from datetime import datetime, timedelta
import pandas as pd

metro_dir = "data/kochimetro_gtfs"
os.makedirs(metro_dir, exist_ok=True)

stations = [
    ("KM_01", "Aluva", 10.1098, 76.3533),
    ("KM_02", "Pulinchodu", 10.0988, 76.3475),
    ("KM_03", "Companypady", 10.0886, 76.3421),
    ("KM_04", "Ambattukavu", 10.0784, 76.3359),
    ("KM_05", "Muttom", 10.0729, 76.3314),
    ("KM_06", "Kalamassery", 10.0573, 76.3217),
    ("KM_07", "Cochin University", 10.0468, 76.3182),
    ("KM_08", "Pathadipalam", 10.0359, 76.3142),
    ("KM_09", "Edapally", 10.0248, 76.3079),
    ("KM_10", "Changampuzha Park", 10.0151, 76.3023),
    ("KM_11", "Palarivattom", 10.0054, 76.3005),
    ("KM_12", "JLN Stadium", 9.9986, 76.3001),
    ("KM_13", "Kaloor", 9.9922, 76.2934),
    ("KM_14", "Lissie", 9.9888, 76.2872),
    ("KM_15", "M.G Road", 9.9823, 76.2829),
    ("KM_16", "Maharaja's College", 9.9729, 76.2842),
    ("KM_17", "Ernakulam South", 9.9673, 76.2891),
    ("KM_18", "Kadavanthra", 9.9669, 76.2994),
    ("KM_19", "Elamkulam", 9.9682, 76.3101),
    ("KM_20", "Vyttila", 9.9661, 76.3214),
    ("KM_21", "Thykoodam", 9.9575, 76.3242),
    ("KM_22", "Petta", 9.9507, 76.3323),
    ("KM_23", "Vadakkekotta", 9.9492, 76.3418),
    ("KM_24", "SN Junction", 9.9481, 76.3489),
    ("KM_25", "Thripunithura", 9.9463, 76.3551),
]

stops_df = pd.DataFrame(stations, columns=["stop_id", "stop_name", "stop_lat", "stop_lon"])
stops_df.to_csv(os.path.join(metro_dir, "stops.txt"), index=False)

trips = []
stop_times = []

start_time = datetime.strptime("06:00:00", "%H:%M:%S")
end_time = datetime.strptime("22:30:00", "%H:%M:%S")

trip_counter = 1
current = start_time

while current <= end_time:
    # Down trip (Aluva to Thripunithura)
    trip_id_down = f"METRO_DN_{trip_counter:03d}"
    trips.append({
        "route_id": "KMRL_L1",
        "service_id": "DAILY",
        "trip_id": trip_id_down,
        "trip_headsign": "Thripunithura",
        "direction_id": 0,
    })

    t = current
    for seq, st in enumerate(stations, start=1):
        arr_str = t.strftime("%H:%M:%S")
        t_dep = t + timedelta(seconds=30)
        dep_str = t_dep.strftime("%H:%M:%S")
        stop_times.append({
            "trip_id": trip_id_down,
            "arrival_time": arr_str,
            "departure_time": dep_str,
            "stop_id": st[0],
            "stop_sequence": seq,
        })
        t = t + timedelta(minutes=2)

    # Up trip (Thripunithura to Aluva)
    trip_id_up = f"METRO_UP_{trip_counter:03d}"
    trips.append({
        "route_id": "KMRL_L1",
        "service_id": "DAILY",
        "trip_id": trip_id_up,
        "trip_headsign": "Aluva",
        "direction_id": 1,
    })

    t = current
    for seq, st in enumerate(reversed(stations), start=1):
        arr_str = t.strftime("%H:%M:%S")
        t_dep = t + timedelta(seconds=30)
        dep_str = t_dep.strftime("%H:%M:%S")
        stop_times.append({
            "trip_id": trip_id_up,
            "arrival_time": arr_str,
            "departure_time": dep_str,
            "stop_id": st[0],
            "stop_sequence": seq,
        })
        t = t + timedelta(minutes=2)

    current += timedelta(minutes=8)
    trip_counter += 1

pd.DataFrame(trips).to_csv(os.path.join(metro_dir, "trips.txt"), index=False)
pd.DataFrame(stop_times).to_csv(os.path.join(metro_dir, "stop_times.txt"), index=False)
print(f"Generated {len(trips)} metro trips and {len(stop_times)} stop_times across {len(stations)} stations.")
