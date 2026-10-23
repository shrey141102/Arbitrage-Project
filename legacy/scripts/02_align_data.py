import os
import pandas as pd

DATA_DIR = "data"
PAIRS = ["EURUSD", "GBPUSD", "EURGBP"]
DATE_STR = "20261014_13"
OUTPUT_FILE = os.path.join(DATA_DIR, f"aligned_grid_{DATE_STR}.parquet")

def build_aligned_grid(freq: str = "1s") -> pd.DataFrame:
    raw_dfs = {}
    for pair in PAIRS:
        file_path = os.path.join(DATA_DIR, f"{pair}_{DATE_STR}.parquet")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing {file_path}. Run data_collection.py first.")
        
        df = pd.read_parquet(file_path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last")
        raw_dfs[pair] = df

    # Find common 1-hour overlap window
    start_time = max(df["time"].min() for df in raw_dfs.values()).floor(freq)
    end_time = min(df["time"].max() for df in raw_dfs.values()).ceil(freq)
    
    # 3,600 uniform 1-second discrete timestamps
    master_grid = pd.DataFrame({"time": pd.date_range(start_time, end_time, freq=freq)})
    
    aligned = master_grid.copy()
    for pair, df in raw_dfs.items():
        # Rename columns to pair-specific headers
        renamed_df = df.rename(columns={
            "bid": f"{pair}_bid",
            "ask": f"{pair}_ask",
            "bid_vol": f"{pair}_bid_vol",
            "ask_vol": f"{pair}_ask_vol",
        })
        
        # Backward merge_asof: at time t, only use the latest tick <= t (zero look-ahead)
        aligned = pd.merge_asof(
            aligned,
            renamed_df,
            on="time",
            direction="backward",
            tolerance=pd.Timedelta("60s")
        )

    # Drop any initial rows if a pair had no tick before the start boundary
    aligned = aligned.dropna().reset_index(drop=True)
    aligned.to_parquet(OUTPUT_FILE, index=False)
    
    print("\n================ ALIGNED GRID VERIFICATION ================")
    print(f"File saved to        : {OUTPUT_FILE}")
    print(f"Total time steps (t) : {len(aligned):,} seconds")
    print(f"Start timestamp      : {aligned['time'].min()}")
    print(f"End timestamp        : {aligned['time'].max()}")
    print(f"Columns              : {list(aligned.columns)}")
    print("\nSample (first 3 seconds):")
    cols_to_show = ["time", "EURUSD_bid", "EURUSD_ask", "GBPUSD_bid", "GBPUSD_ask", "EURGBP_bid", "EURGBP_ask"]
    print(aligned[cols_to_show].head(3))
    print("===========================================================\n")
    return aligned

if __name__ == "__main__":
    build_aligned_grid(freq="1s")