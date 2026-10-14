
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

TRAIN_NUMBERS = ["12623", "12624", "16345", "16346", "16629", "16630"]

KERALA_STATIONS = {
    "TVC", "QLN", "ALLP", "KTYM", "ERS",
    "ERN", "TCR", "PGT", "CLT", "CAN",
}

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "kerala_train_sample.csv"

frames = []

for train_no in TRAIN_NUMBERS:
    print(f"Reading train {train_no}...")

    try:
        path = hf_hub_download(
            repo_id="yadavsaurabh/indian-railways-runs",
            filename=f"trains/{train_no}.csv",
            repo_type="dataset",
        )

        df = pd.read_csv(path, low_memory=False)

        # Keep only selected Kerala stations.
        df["station_code"] = df["station_code"].astype(str).str.strip()
        df = df[df["station_code"].isin(KERALA_STATIONS)].copy()

        if not df.empty:
            frames.append(df)

    except Exception as error:
        print(f"Could not read train {train_no}: {error}")

if not frames:
    print("No matching Kerala station records found.")
    raise SystemExit(1)

data = pd.concat(frames, ignore_index=True)

# Keep records with a usable actual arrival and known arrival delay.
data["delay_arrival_min"] = pd.to_numeric(
    data["delay_arrival_min"], errors="coerce"
)

training_data = data.dropna(
    subset=["date", "train_no", "seq", "station_code",
            "sch_arrival", "act_arrival", "delay_arrival_min"]
).copy()

training_data = training_data[
    training_data["reported"].eq(1)
    & training_data["halts_here"].eq(1)
].copy()

training_data.to_csv(OUTPUT_FILE, index=False)

print("\n--- DATASET SUMMARY ---")
print("Kerala station records:", len(data))
print("Rows with usable arrival delay:", len(training_data))
print("Trains represented:", training_data["train_no"].nunique())
print("Stations represented:", training_data["station_code"].nunique())
print("\nStations found:")
print(
    training_data[["station_code", "train_name"]]
    .drop_duplicates()
    .sort_values("station_code")
    .to_string(index=False)
)
print(f"\nSaved to: {OUTPUT_FILE}")
