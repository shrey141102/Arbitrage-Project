import os
import lzma
import struct
import time
from datetime import datetime, timedelta
import requests
import pandas as pd

OUTPUT_DIR = "data"
PAIRS = ["EURUSD", "GBPUSD", "EURGBP"]

# Use a persistent session with browser headers
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "*/*",
    "Connection": "keep-alive"
})

def fetch_hour(pair: str, dt: datetime, max_retries: int = 5) -> pd.DataFrame:
    """Downloads and unpacks 1 hour of Dukascopy .bi5 tick data with robust retries."""
    url = (
        f"https://datafeed.dukascopy.com/datafeed/{pair}/"
        f"{dt.year}/{dt.month - 1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"
    )

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[{pair}] {dt.strftime('%H:00')} UTC (Attempt {attempt}/{max_retries})...")
            r = session.get(url, timeout=45)
            
            if r.status_code == 200 and len(r.content) > 0:
                raw = lzma.decompress(r.content)
                scale = 1e3 if pair.endswith("JPY") else 1e5
                
                records = [
                    (dt + timedelta(milliseconds=ms), bid / scale, ask / scale, bid_vol, ask_vol)
                    for ms, ask, bid, ask_vol, bid_vol in struct.iter_unpack(">IIIff", raw)
                ]
                df = pd.DataFrame(records, columns=["time", "bid", "ask", "bid_vol", "ask_vol"])
                print(f"  -> Successfully unpacked {len(df):,} ticks.")
                return df
                
            elif r.status_code == 404:
                print(f"  -> 404 Not Found.")
                return pd.DataFrame()
            else:
                print(f"  -> Server returned status {r.status_code}, retrying...")

        except (requests.RequestException, lzma.LZMAError) as e:
            print(f"  -> Network/Decompress error: {e}")

        # Exponential backoff: 2s, 4s, 8s...
        time.sleep(2 ** attempt)

    print(f"  -> FAILED after {max_retries} attempts.")
    return pd.DataFrame()

def collect_synchronized_hour(target_date: datetime, hour: int):
    """Fetches a single, fully-overlapping 1-hour window across all pairs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dt = target_date.replace(hour=hour, minute=0, second=0, microsecond=0)
    
    print(f"\nTargeting Synchronized Window: {dt.strftime('%Y-%m-%d %H:00')} UTC across {PAIRS}\n")
    
    results = {}
    for pair in PAIRS:
        out_file = os.path.join(OUTPUT_DIR, f"{pair}_{dt.strftime('%Y%m%d_%H')}.parquet")
        
        # Check if already saved and valid
        if os.path.exists(out_file):
            df = pd.read_parquet(out_file)
            if len(df) > 0:
                print(f"[{pair}] Found cached file: {len(df):,} ticks.")
                results[pair] = df
                continue

        df = fetch_hour(pair, dt)
        if not df.empty:
            df.to_parquet(out_file, index=False)
            results[pair] = df
        else:
            print(f"\nCRITICAL: Could not fetch {pair} for hour {hour}. Try another hour or date.")
            return False
        
        time.sleep(1)

    print("\n================ DATA VERIFICATION ================")
    all_valid = True
    for pair, df in results.items():
        if df.empty:
            all_valid = False
            print(f"{pair:7s} | MISSING")
        else:
            spread = (df["ask"] - df["bid"]).mean()
            print(f"{pair:7s} | {len(df):,} ticks | Avg Spread: {spread:.5f} | Start: {df['time'].min()} | End: {df['time'].max()}")
    print("===================================================\n")
    return all_valid

if __name__ == "__main__":
    # Target 13:00 UTC (1 PM to 2 PM London/NY overlap)
    # Notice: GBPUSD and EURGBP already succeeded for 13:00 earlier! 
    # This run will cache/fetch cleanly so all 3 pairs match 13:00 UTC.
    TARGET_DATE = datetime(2026, 10, 14)
    TARGET_HOUR = 13

    success = collect_synchronized_hour(TARGET_DATE, TARGET_HOUR)
    if success:
        print("Ready for Step 3 (Timestamp Alignment).")