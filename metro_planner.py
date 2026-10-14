from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent / "data" / "kochimetro_gtfs"
_cached_metro_data = None


def load_metro_data():
    global _cached_metro_data
    if _cached_metro_data is not None:
        return _cached_metro_data

    stops = pd.read_csv(DATA_DIR / "stops.txt")
    stop_times = pd.read_csv(DATA_DIR / "stop_times.txt")
    trips = pd.read_csv(DATA_DIR / "trips.txt")

    stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"])
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

    _cached_metro_data = (stops, stop_times, trips)
    return _cached_metro_data


def get_metro_journeys(source, destination):
    # Match common spelling differences between bus and Metro data.
    station_aliases = {
        "edappally": "Edapally",
        "edappally toll": "Edapally",
        "maharajas college": "Maharaja's College",
        "ernakulam south": "Ernakulam South",
        "ernakulam north": "Lissie",
        "kaloor bus stand": "Kaloor",
        "vyttila hub": "Vyttila",
        "aluva bus stand": "Aluva",
    }

    src_clean = source.strip().lower()
    dest_clean = destination.strip().lower()

    source = station_aliases.get(src_clean, source.strip())
    destination = station_aliases.get(dest_clean, destination.strip())

    stops, stop_times, trips = load_metro_data()

    source_stops = stops[
        stops["stop_name"].str.contains(source, case=False, na=False)
    ]
    destination_stops = stops[
        stops["stop_name"].str.contains(destination, case=False, na=False)
    ]

    if source_stops.empty or destination_stops.empty:
        return pd.DataFrame()

    source_ids = source_stops["stop_id"].tolist()
    destination_ids = destination_stops["stop_id"].tolist()

    journeys = []

    for trip_id, group in stop_times.groupby("trip_id"):
        group = group.sort_values("stop_sequence")

        source_rows = group[group["stop_id"].isin(source_ids)]
        destination_rows = group[group["stop_id"].isin(destination_ids)]

        if source_rows.empty or destination_rows.empty:
            continue

        for _, start in source_rows.iterrows():
            later_stops = destination_rows[
                destination_rows["stop_sequence"] > start["stop_sequence"]
            ]

            if later_stops.empty:
                continue

            end = later_stops.iloc[0]

            start_seconds = pd.to_timedelta(start["departure_time"]).total_seconds()
            end_seconds = pd.to_timedelta(end["arrival_time"]).total_seconds()

            journeys.append({
                "trip_id": trip_id,
                "source": source,
                "destination": destination,
                "departure": start["departure_time"],
                "arrival": end["arrival_time"],
                "travel_minutes": round((end_seconds - start_seconds) / 60, 1),
            })

    return pd.DataFrame(journeys).drop_duplicates()
