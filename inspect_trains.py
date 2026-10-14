
import pandas as pd
from huggingface_hub import hf_hub_download

trains = ["12623", "12624", "16345", "16346", "16629", "16630"]

for train in trains:
    print(f"\n--- TRAIN {train} ---")

    try:
        path = hf_hub_download(
            repo_id="yadavsaurabh/indian-railways-runs",
            filename=f"trains/{train}.csv",
            repo_type="dataset",
        )

        df = pd.read_csv(path, nrows=2)

        print("Columns:", df.columns.tolist())
        print(df.to_string(index=False))

    except Exception as error:
        print("Could not inspect this train:", error)
