import time
from pathlib import Path
import polars as pl

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TABLES_DIR = BASE_DIR / "reports" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_FILE = DATA_DIR / "all_discovered_arbitrage_signals.parquet"


def analyze_arbitrage_signals():
    if not SIGNALS_FILE.exists():
        raise FileNotFoundError(f"Could not find {SIGNALS_FILE}")

    t0 = time.perf_counter()
    lf = pl.scan_parquet(SIGNALS_FILE)

    # 1. Overall & Institutional Filtering
    total_raw = lf.select(pl.len()).collect().item()
    inst_lf = lf.filter(pl.col("institutional_net_bps") > 0.0)
    total_inst = inst_lf.select(pl.len()).collect().item()

    print("======================================================================")
    print("       INSTITUTIONAL FX TRIANGULAR ARBITRAGE: EMPIRICAL SUMMARY       ")
    print("======================================================================")
    print(f"Total 100ms Raw Executable Signals (> 0.0 bps) : {total_raw:>12,}")
    print(f"Total 100ms Institutional Signals  (> 0.3 bps) : {total_inst:>12,} ({total_inst/total_raw*100:.1f}%)")

    # 2. Top Currency Triangles Leaderboard (Institutional Only)
    by_cycle = (
        inst_lf.group_by("cycle")
        .agg([
            pl.len().alias("inst_ticks_100ms"),
            pl.col("institutional_net_bps").mean().round(3).alias("mean_net_bps"),
            pl.col("institutional_net_bps").median().round(3).alias("median_net_bps"),
            pl.col("institutional_net_bps").quantile(0.95).round(3).alias("p95_net_bps"),
            pl.col("institutional_net_bps").max().round(3).alias("max_net_bps"),
        ])
        .sort("inst_ticks_100ms", descending=True)
        .collect()
    )
    print("\n--- TOP 10 CURRENCY TRIANGLES (NET OF 0.3 BPS INSTITUTIONAL FEES) ---")
    print(by_cycle.head(10))

    # 3. De-duplicating Consecutive 100ms Ticks into Distinct "Arbitrage Windows"
    # If the same cycle fires at 12:00:00.100, 12:00:00.200, 12:00:00.300 (gap <= 500ms),
    # that is ONE continuous arbitrage window of duration 300ms.
    windows_df = (
        inst_lf.sort(["folder", "cycle", "time"])
        .with_columns([
            pl.col("time").diff().over(["folder", "cycle"]).dt.total_milliseconds().alias("gap_ms")
        ])
        .with_columns([
            ((pl.col("gap_ms").is_null()) | (pl.col("gap_ms") > 500)).cast(pl.Int64).alias("is_new_window")
        ])
        .with_columns([
            pl.col("is_new_window").cum_sum().over(["folder", "cycle"]).alias("window_id")
        ])
        .group_by(["folder", "cycle", "window_id"])
        .agg([
            pl.col("time").min().alias("start_time"),
            pl.col("time").max().alias("end_time"),
            pl.len().alias("ticks_in_window"),
            pl.col("institutional_net_bps").max().round(3).alias("peak_net_bps"),
            pl.col("institutional_net_bps").mean().round(3).alias("avg_net_bps"),
        ])
        .with_columns([
            ((pl.col("end_time") - pl.col("start_time")).dt.total_milliseconds() + 100).alias("duration_ms")
        ])
        .collect()
    )

    total_windows = windows_df.height
    sub_second_pct = (windows_df.filter(pl.col("duration_ms") <= 1000).height / total_windows) * 100
    single_tick_pct = (windows_df.filter(pl.col("duration_ms") == 100).height / total_windows) * 100
    median_dur = windows_df["duration_ms"].median()
    mean_dur = windows_df["duration_ms"].mean()

    print("\n--- ARBITRAGE HALF-LIFE & WINDOW DURATION MICROSTRUCTURE ---")
    print(f"Distinct Independent Arbitrage Windows : {total_windows:>12,}")
    print(f"Single-Tick Windows (Lasting <= 100ms) : {single_tick_pct:>11.2f}%")
    print(f"Sub-Second Windows  (Lasting <= 1.0s)  : {sub_second_pct:>11.2f}%")
    print(f"Median Window Duration                 : {median_dur:>9.0f} ms")
    print(f"Mean Window Duration                   : {mean_dur:>9.1f} ms")

    print("\n--- TOP 15 DISTINCT INDEPENDENT ARBITRAGE EVENTS (DE-CLUSTERED) ---")
    top_events = windows_df.sort("peak_net_bps", descending=True).select([
        "folder", "start_time", "cycle", "duration_ms", "ticks_in_window", "peak_net_bps"
    ]).head(15)
    print(top_events)

    # 4. Monthly Regime Aggregation (across all 29 Kaggle folders)
    by_month = (
        lf.group_by("folder")
        .agg([
            pl.len().alias("raw_ticks_100ms"),
            (pl.col("institutional_net_bps") > 0.0).sum().alias("inst_ticks_100ms"),
            pl.col("raw_bid_ask_bps").max().round(3).alias("max_raw_bps"),
            pl.col("institutional_net_bps").max().round(3).alias("max_inst_bps"),
            pl.col("institutional_net_bps").filter(pl.col("institutional_net_bps") > 0.0).mean().round(3).alias("mean_inst_bps"),
        ])
        .sort("inst_ticks_100ms", descending=True)
        .collect()
    )

    # Save FULL unbiased artifacts for downstream Statistical Tests (S1-S7) & EDA
    by_cycle.write_csv(TABLES_DIR / "summary_by_triangle.csv")
    by_month.write_csv(TABLES_DIR / "summary_by_month.csv")
    # Also write a copy to DATA_DIR for backward compatibility
    by_cycle.write_csv(DATA_DIR / "summary_by_triangle.csv")
    by_month.write_csv(DATA_DIR / "summary_by_month.csv")
    windows_df.write_parquet(DATA_DIR / "all_distinct_windows.parquet")
    windows_df.sort("peak_net_bps", descending=True).head(1000).write_parquet(DATA_DIR / "top_1000_distinct_windows.parquet")

    print(f"\nSaved full unbiased population ({windows_df.height:,} windows) -> {DATA_DIR / 'all_distinct_windows.parquet'}")
    print(f"Saved monthly regime summary ({by_month.height} months) -> {TABLES_DIR / 'summary_by_month.csv'}")
    print(f"Completed in {time.perf_counter() - t0:.2f}s.")
    print("======================================================================\n")

if __name__ == "__main__":
    analyze_arbitrage_signals()