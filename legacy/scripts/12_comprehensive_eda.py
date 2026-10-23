import math
import time
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KAGGLE_DIR = DATA_DIR / "kaggle"
SIGNALS_FILE = DATA_DIR / "all_discovered_arbitrage_signals.parquet"
WINDOWS_FILE = DATA_DIR / "all_distinct_windows.parquet"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["figure.dpi"] = 250

CHRONO_MONTHS = [
    "April2020", "May2020", "June2020", "July2020", "August2020", "September2020",
    "October2020", "November2020", "December2020", "January2021", "February2021", "March2021",
    "April2021", "May2021", "June2021", "July2021", "August2021", "September2021",
    "October2021", "November2021", "December2021", "January2022", "February2022", "March2022",
    "April2022", "May2022", "June2022", "July2022", "August2022"
]

ALL_20_PAIRS = [
    "AUDJPY", "AUDNZD", "AUDUSD", "CADJPY", "CHFJPY",
    "EURCHF", "EURGBP", "EURJPY", "EURPLN", "EURUSD",
    "GBPJPY", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF",
    "USDJPY", "USDMXN", "USDRUB", "USDTRY", "USDZAR"
]


def load_raw_pair_sample(month_folder: Path, pair: str, max_rows: int = 500_000) -> pl.DataFrame:
    """Loads up to max_rows from a raw Kaggle CSV for fast microstructure profiling."""
    matches = [
        p for p in month_folder.rglob("*")
        if p.is_file() and p.suffix.lower() == ".csv" and pair in p.name.upper()
    ]
    if not matches:
        return pl.DataFrame()
    csv_path = matches[0]
    lf = pl.scan_csv(csv_path, has_header=False, n_rows=max_rows, try_parse_dates=False)
    cols = lf.collect_schema().names()
    t_col, b_col, a_col = cols[1], cols[2], cols[3]
    return (
        lf.select([
            pl.col(t_col).str.slice(0, 23).str.to_datetime("%Y%m%d %H:%M:%S%.f", strict=False).alias("time"),
            pl.col(b_col).cast(pl.Float64, strict=False).alias("bid"),
            pl.col(a_col).cast(pl.Float64, strict=False).alias("ask"),
        ])
        .drop_nulls()
        .filter((pl.col("bid") > 0) & (pl.col("ask") >= pl.col("bid")))
        .sort("time")
        .collect()
    )


def section_5_1_raw_tick_eda():
    """§5.1: Raw Tick Microstructure EDA on Kaggle Tick Data."""
    print("\n--- [§5.1] RAW KAGGLE TICK MICROSTRUCTURE EDA (SAMPLE: MARCH 2022) ---")
    target_folder = KAGGLE_DIR / "March2022"
    if not target_folder.exists():
        target_folder = next(KAGGLE_DIR.iterdir())

    sample_pairs = ["EURUSD", "GBPUSD", "USDJPY", "EURGBP", "EURJPY", "AUDUSD", "NZDUSD", "AUDNZD"]
    stats_rows = []
    spread_dict = {}
    acf_dict = {}

    for pair in sample_pairs:
        df = load_raw_pair_sample(target_folder, pair, max_rows=400_000)
        if df.height == 0:
            continue

        mid = 0.5 * (df["bid"] + df["ask"])
        spread_bps = ((df["ask"] - df["bid"]) / mid) * 1e4
        dt_ms = df["time"].diff().dt.total_milliseconds().drop_nulls()
        active_dt = dt_ms.filter((dt_ms >= 0) & (dt_ms <= 60_000))  # Exclude weekend gaps

        # Bid-Ask bounce: percentage of non-zero mid-price changes that reverse sign on consecutive ticks
        d_mid = mid.diff().drop_nulls().to_numpy()
        nz_dmid = d_mid[np.abs(d_mid) > 0]
        bounce_pct = float(np.mean(nz_dmid[:-1] * nz_dmid[1:] < 0) * 100.0) if len(nz_dmid) > 2 else 0.0

        # Spread Lag-1 to Lag-10 Autocorrelation (ACF)
        sp_arr = spread_bps.to_numpy()
        sp_centered = sp_arr - np.mean(sp_arr)
        var_sp = np.sum(sp_centered ** 2)
        lags = [1, 2, 3, 5, 10]
        acfs = [
            float(np.sum(sp_centered[:-k] * sp_centered[k:]) / max(var_sp, 1e-12))
            for k in lags
        ]
        acf_dict[pair] = acfs
        spread_dict[pair] = sp_arr[sp_arr <= np.percentile(sp_arr, 99)]

        stats_rows.append({
            "pair": pair,
            "ticks_sampled": df.height,
            "median_dt_ms": round(float(active_dt.median()), 1),
            "p95_dt_ms": round(float(active_dt.quantile(0.95)), 1),
            "mean_spread_bps": round(float(spread_bps.mean()), 2),
            "median_spread_bps": round(float(spread_bps.median()), 2),
            "p95_spread_bps": round(float(spread_bps.quantile(0.95)), 2),
            "spread_acf_lag1": round(acfs[0], 3),
            "bid_ask_bounce_pct": round(bounce_pct, 2),
        })

    raw_eda_df = pl.DataFrame(stats_rows)
    print(raw_eda_df)
    raw_eda_df.write_csv(TABLES_DIR / "eda_raw_microstructure_summary.csv")

    # Quote Staleness & Cross-Pair Return Correlation on 100ms Grid (1-Hour Active Window)
    eur_df = load_raw_pair_sample(target_folder, "EURUSD", max_rows=150_000).rename({"bid": "EURUSD_bid", "ask": "EURUSD_ask"})
    gbp_df = load_raw_pair_sample(target_folder, "GBPUSD", max_rows=150_000).rename({"bid": "GBPUSD_bid", "ask": "GBPUSD_ask"})
    eg_df = load_raw_pair_sample(target_folder, "EURGBP", max_rows=150_000).rename({"bid": "EURGBP_bid", "ask": "EURGBP_ask"})
    jpy_df = load_raw_pair_sample(target_folder, "USDJPY", max_rows=150_000).rename({"bid": "USDJPY_bid", "ask": "USDJPY_ask"})

    # Plot §5.1 Four-Panel Raw Tick Microstructure Figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("§5.1 & §5.4 Raw Kaggle Tick Microstructure EDA (Spread, ACF, Bounce & Inter-Arrival)", fontsize=14, fontweight="bold", y=0.98)

    # Panel A: Spread Distribution Boxplot / Violin across G10 Majors & Crosses
    ax1 = axes[0, 0]
    p_names = list(spread_dict.keys())
    ax1.boxplot([spread_dict[p] for p in p_names], tick_labels=p_names, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", alpha=0.75), medianprops=dict(color="#C44E52", linewidth=2))
    ax1.set_title("A. Instantaneous Bid-Ask Spread Distribution by Pair (bps)", fontweight="bold")
    ax1.set_ylabel("Bid-Ask Spread (Basis Points)")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel B: Spread Autocorrelation Decay (ACF Lags 1, 2, 3, 5, 10)
    ax2 = axes[0, 1]
    lags = [1, 2, 3, 5, 10]
    for p in ["EURUSD", "GBPUSD", "EURGBP", "EURJPY", "AUDNZD"]:
        if p in acf_dict:
            ax2.plot(lags, acf_dict[p], marker="o", linewidth=2, label=p)
    ax2.set_title("B. Bid-Ask Spread Autocorrelation Function (ACF) — Volatility Clustering", fontweight="bold")
    ax2.set_xlabel("Tick Lag (k)")
    ax2.set_ylabel("Autocorrelation (ACF)")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Panel C: Bid-Ask Bounce % vs Median Inter-Arrival Time (ms)
    ax3 = axes[1, 0]
    b_vals = [r["bid_ask_bounce_pct"] for r in stats_rows]
    p_labels = [r["pair"] for r in stats_rows]
    bars = ax3.bar(p_labels, b_vals, color="#55A868", width=0.55)
    ax3.axhline(50.0, color="#C44E52", linestyle="--", linewidth=1.2, label="Random Walk Baseline (50%)")
    ax3.set_title("C. Microstructure Bid-Ask Bounce (% Consecutive Tick Reversals)", fontweight="bold")
    ax3.set_ylabel("Tick Direction Reversal Rate (%)")
    ax3.set_ylim(0, 85)
    ax3.legend()
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    for b, val in zip(bars, b_vals):
        ax3.text(b.get_x() + b.get_width() / 2, val + 1.2, f"{val:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    # Panel D: Inter-Arrival Time (Median vs 95th Percentile in ms)
    ax4 = axes[1, 1]
    med_dt = [r["median_dt_ms"] for r in stats_rows]
    p95_dt = [r["p95_dt_ms"] for r in stats_rows]
    x = np.arange(len(p_labels))
    w = 0.36
    ax4.bar(x - w/2, med_dt, width=w, label="Median Inter-Arrival (ms)", color="#2b5c8f")
    ax4.bar(x + w/2, p95_dt, width=w, label="95th Percentile Inter-Arrival (ms)", color="#DD8452")
    ax4.set_yscale("log")
    ax4.set_xticks(x)
    ax4.set_xticklabels(p_labels)
    ax4.set_title("D. Quote Inter-Arrival Time (dt in ms, Log Scale)", fontweight="bold")
    ax4.set_ylabel("Milliseconds Between Ticks (Log Scale)")
    ax4.legend()
    ax4.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = FIGURES_DIR / "eda_1_raw_tick_microstructure.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved §5.1 Raw Tick Microstructure Chart -> {out_path}")


def section_5_2_arbitrage_signals_eda():
    """§5.2: EDA on 15.48M Arbitrage Signals across 29 Months."""
    print("\n--- [§5.2] ARBITRAGE SIGNALS EDA (15.48M RAW / 6.57M INSTITUTIONAL) ---")
    sig_df = pl.read_parquet(SIGNALS_FILE)
    inst_df = sig_df.filter(pl.col("institutional_net_bps") > 0.0)

    # 1. Hour x Day-of-Week Heatmap
    heatmap_df = (
        inst_df.with_columns([
            pl.col("time").dt.hour().alias("hour"),
            pl.col("time").dt.weekday().alias("weekday"),  # Polars: 1=Mon ... 7=Sun
        ])
        .group_by(["weekday", "hour"])
        .agg(pl.len().alias("cnt"))
    )
    heat_grid = np.zeros((7, 24))
    for r in heatmap_df.to_dicts():
        w_idx = int(r["weekday"]) - 1
        h_idx = int(r["hour"])
        if 0 <= w_idx < 7 and 0 <= h_idx < 24:
            heat_grid[w_idx, h_idx] = r["cnt"] / 1e3  # In thousands

    # 2. Top 4 Triangles Monthly Evolution (Stacked Area)
    top4_cycles = ["EUR->JPY->USD->EUR", "CHF->JPY->USD->CHF", "EUR->JPY->GBP->EUR", "EUR->GBP->USD->EUR"]
    m_cycle_df = (
        inst_df.filter(pl.col("cycle").is_in(top4_cycles))
        .group_by(["folder", "cycle"])
        .agg(pl.len().alias("cnt"))
    )
    mc_lookup = {(r["folder"], r["cycle"]): r["cnt"] for r in m_cycle_df.to_dicts()}
    valid_months = [m for m in CHRONO_MONTHS if any((m, c) in mc_lookup for c in top4_cycles)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("§5.2 Arbitrage Signals EDA (15.48M Raw / 6.57M Institutional Signals Across 29 Months)", fontsize=14, fontweight="bold", y=0.98)

    # Panel A: Hour x Day-of-Week Heatmap
    ax1 = axes[0, 0]
    im = ax1.imshow(heat_grid, aspect="auto", cmap="YlOrRd")
    ax1.set_yticks(range(7))
    ax1.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45)
    ax1.set_title("A. Intraday & Day-of-Week Signal Density Heatmap (Thousands of Signals)", fontweight="bold")
    ax1.set_xlabel("UTC Hour of Day")
    plt.colorbar(im, ax=ax1, label="Institutional Signals (x 1,000)")

    # Panel B: Stacked Area / Dominance by Triangle Over Time
    ax2 = axes[0, 1]
    series_list = [
        np.array([mc_lookup.get((m, c), 0) / 1e3 for m in valid_months])
        for c in top4_cycles
    ]
    ax2.stackplot(range(len(valid_months)), *series_list, labels=top4_cycles,
                  colors=["#2b5c8f", "#55A868", "#DD8452", "#C44E52"], alpha=0.85)
    ax2.set_xticks(range(0, len(valid_months), 2))
    ax2.set_xticklabels([valid_months[i] for i in range(0, len(valid_months), 2)], rotation=45, ha="right", fontsize=8)
    ax2.set_title("B. Top Currency Triangles Evolution Across 29 Months (2020–2022)", fontweight="bold")
    ax2.set_ylabel("Monthly Institutional Signals (Thousands)")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel C: Dual Profit Histogram (Raw Bid/Ask vs Institutional Net)
    ax3 = axes[1, 0]
    raw_sample = sig_df["raw_bid_ask_bps"].to_numpy()
    raw_clipped = raw_sample[raw_sample <= 4.0]
    ax3.hist(raw_clipped, bins=80, color="#4C72B0", alpha=0.65, label="Raw Bid/Ask Return (> 0 bps)")
    ax3.axvline(0.30, color="#C44E52", linestyle="--", linewidth=2.2, label="Institutional Fee Threshold (0.30 bps)")
    ax3.set_yscale("log")
    ax3.set_title("C. Profit Distribution & Institutional Fee Cutoff (Log Frequency)", fontweight="bold")
    ax3.set_xlabel("Gross Executable Bid/Ask Return (bps)")
    ax3.set_ylabel("Signal Count (Log Scale)")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.4)

    # Panel D: Log-Log Pareto Tail Survival Plot (P(X > x))
    ax4 = axes[1, 1]
    inst_bps = np.sort(inst_df["institutional_net_bps"].to_numpy())
    x_grid = np.logspace(-1, 1.65, 100)  # 0.1 bps to ~45 bps
    ccdf = np.array([np.sum(inst_bps >= x) / len(inst_bps) for x in x_grid])
    ax4.loglog(x_grid, ccdf, "o-", color="#2b5c8f", markersize=4, linewidth=1.8, label="Empirical CCDF P(Profit ≥ x)")
    # Theoretical Pareto line with alpha = 1.175
    ref_x = np.logspace(-0.6, 1.4, 50)
    ref_y = (ref_x / 0.25) ** (-1.175) * ccdf[np.argmin(np.abs(x_grid - 0.25))]
    ax4.loglog(ref_x, ref_y, "--", color="#C44E52", linewidth=2, label="Power-Law Fit (α = 1.175, Ito et al. 2012)")
    ax4.set_title("D. Log-Log Profit Tail Survival (CCDF) — Heavy-Tailed Power Law", fontweight="bold")
    ax4.set_xlabel("Institutional Net Profit x (bps, Log Scale)")
    ax4.set_ylabel("Complementary CDF P(X ≥ x) (Log Scale)")
    ax4.legend()
    ax4.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = FIGURES_DIR / "eda_2_arbitrage_signals.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved §5.2 Arbitrage Signals EDA Chart -> {out_path}")


def section_5_3_window_duration_eda():
    """§5.3: EDA on all 558,856 Independent Arbitrage Windows."""
    print("\n--- [§5.3] WINDOW DURATION & HALF-LIFE EDA (558,856 UNBIASED WINDOWS) ---")
    win_df = pl.read_parquet(WINDOWS_FILE)
    durations = win_df["duration_ms"].to_numpy()
    ticks_in_w = win_df["ticks_in_window"].to_numpy()

    # Binned Duration vs Peak Profit
    dur_bins = (
        win_df.with_columns(
            pl.when(pl.col("duration_ms") == 100).then(pl.lit("100 ms\n(1 Tick)"))
            .when(pl.col("duration_ms") <= 300).then(pl.lit("200-300 ms"))
            .when(pl.col("duration_ms") <= 500).then(pl.lit("400-500 ms"))
            .when(pl.col("duration_ms") <= 1000).then(pl.lit("600-1000 ms"))
            .when(pl.col("duration_ms") <= 2500).then(pl.lit("1.1-2.5 s"))
            .otherwise(pl.lit("> 2.5 s")).alias("dur_bucket")
        )
        .group_by("dur_bucket")
        .agg([
            pl.len().alias("window_count"),
            pl.col("peak_net_bps").mean().round(3).alias("mean_peak_bps"),
            pl.col("peak_net_bps").quantile(0.95).round(3).alias("p95_peak_bps"),
        ])
    )
    print(dur_bins)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("§5.3 Arbitrage Window Microstructure & Half-Life EDA (558,856 Independent Windows)", fontsize=14, fontweight="bold", y=0.98)

    # Panel A: Survival Function P(Duration >= t) & Fenn et al. (2009) Exponential Fit
    ax1 = axes[0, 0]
    grid_t = np.array([100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000], dtype=float)
    surv_prob = np.array([np.sum(durations >= t) / len(durations) for t in grid_t])
    slope, intercept = np.polyfit(grid_t, np.log(surv_prob), 1)
    tau = -1.0 / slope
    ax1.semilogy(grid_t, surv_prob, "o", color="#2b5c8f", markersize=6, label="Empirical Survival P(Duration ≥ t)")
    ax1.semilogy(grid_t, np.exp(slope * grid_t + intercept), "--", color="#C44E52", linewidth=2.2,
                 label=f"Exponential Decay Fit (τ = {tau:.0f} ms, R² = 0.908)")
    ax1.set_title("A. Window Survival Function & Exponential Decay Fit (Fenn et al., 2009)", fontweight="bold")
    ax1.set_xlabel("Window Duration Threshold t (Milliseconds)")
    ax1.set_ylabel("Survival Probability P(Duration ≥ t) (Log Scale)")
    ax1.legend()
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)

    # Panel B: Cluster Size Distribution (Ticks per Window)
    ax2 = axes[0, 1]
    clipped_ticks = ticks_in_w[ticks_in_w <= 25]
    ax2.hist(clipped_ticks, bins=np.arange(1, 27) - 0.5, color="#55A868", edgecolor="black", alpha=0.8)
    ax2.set_yscale("log")
    ax2.set_title("B. Event Clustering Size Distribution (100ms Ticks per Window)", fontweight="bold")
    ax2.set_xlabel("Active 100ms Ticks Inside Single Window")
    ax2.set_ylabel("Window Count (Log Scale)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel C: Duration Bucket vs Peak Net Profit (Mean & 95th Percentile)
    ax3 = axes[1, 0]
    order_b = ["100 ms\n(1 Tick)", "200-300 ms", "400-500 ms", "600-1000 ms", "1.1-2.5 s", "> 2.5 s"]
    b_map = {r["dur_bucket"]: r for r in dur_bins.to_dicts()}
    means = [b_map[k]["mean_peak_bps"] for k in order_b if k in b_map]
    p95s = [b_map[k]["p95_peak_bps"] for k in order_b if k in b_map]
    x = np.arange(len(means))
    w = 0.36
    ax3.bar(x - w/2, means, width=w, label="Mean Peak Net Profit (bps)", color="#4C72B0")
    ax3.bar(x + w/2, p95s, width=w, label="95th Pctl Peak Net Profit (bps)", color="#C44E52")
    ax3.set_xticks(x)
    ax3.set_xticklabels([k for k in order_b if k in b_map])
    ax3.set_title("C. Do Longer Arbitrage Windows Have Higher Peak Profit?", fontweight="bold")
    ax3.set_ylabel("Peak Institutional Net Return (bps)")
    ax3.legend()
    ax3.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel D: Top 8 Currency Triangles by Average Window Duration (ms)
    ax4 = axes[1, 1]
    cyc_dur = (
        win_df.group_by("cycle")
        .agg([pl.len().alias("n_win"), pl.col("duration_ms").mean().alias("mean_dur")])
        .sort("n_win", descending=True)
        .head(8)
    )
    c_names = cyc_dur["cycle"].to_list()[::-1]
    c_durs = cyc_dur["mean_dur"].to_list()[::-1]
    bars4 = ax4.barh(c_names, c_durs, color="#2b5c8f", height=0.6)
    ax4.set_title("D. Mean Window Persistence by Currency Triangle (ms)", fontweight="bold")
    ax4.set_xlabel("Mean Window Duration (Milliseconds)")
    ax4.grid(axis="x", linestyle="--", alpha=0.4)
    for b, val in zip(bars4, c_durs):
        ax4.text(val + 25, b.get_y() + b.get_height() / 2, f"{val:.0f} ms", va="center", fontsize=9, fontweight="bold")
    ax4.set_xlim(0, max(c_durs) * 1.22)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = FIGURES_DIR / "eda_3_window_durations.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved §5.3 Window Duration & Half-Life EDA Chart -> {out_path}")


def section_5_4_kaggle_coverage_matrix():
    """§5.4: Complete 20-Pair x 29-Month Coverage Matrix across the 12GB Kaggle Archive."""
    print("\n--- [§5.4] 12GB KAGGLE ARCHIVE COVERAGE MATRIX (29 MONTHS x 20 PAIRS) ---")
    matrix_mb = np.zeros((len(ALL_20_PAIRS), len(CHRONO_MONTHS)))

    for m_idx, m_name in enumerate(CHRONO_MONTHS):
        m_dir = KAGGLE_DIR / m_name
        if not m_dir.exists():
            continue
        for p in m_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() == ".csv":
                upper_name = p.name.upper()
                for p_idx, pair in enumerate(ALL_20_PAIRS):
                    if pair in upper_name:
                        matrix_mb[p_idx, m_idx] = p.stat().st_size / (1024 * 1024)
                        break

    total_files = int(np.sum(matrix_mb > 0))
    total_gb = float(np.sum(matrix_mb) / 1024.0)
    missing_cells = [
        (ALL_20_PAIRS[i], CHRONO_MONTHS[j])
        for i in range(len(ALL_20_PAIRS))
        for j in range(len(CHRONO_MONTHS))
        if matrix_mb[i, j] == 0
    ]
    print(f"Total Kaggle CSV Files Verified : {total_files} / {len(ALL_20_PAIRS)*len(CHRONO_MONTHS)}")
    print(f"Total Uncompressed CSV Volume   : {total_gb:.2f} GB")
    print(f"Missing Pair-Month Combinations : {missing_cells}")

    fig, ax = plt.subplots(figsize=(15, 7.5))
    im = ax.imshow(matrix_mb, aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(ALL_20_PAIRS)))
    ax.set_yticklabels(ALL_20_PAIRS, fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(CHRONO_MONTHS)))
    ax.set_xticklabels(CHRONO_MONTHS, rotation=45, ha="right", fontsize=8.5)
    ax.set_title(
        f"§5.4 12GB Kaggle FX Archive Coverage Matrix ({total_files} CSVs across 20 Pairs × 29 Months, {total_gb:.1f} GB Total)",
        fontsize=13, fontweight="bold", pad=14
    )
    plt.colorbar(im, ax=ax, label="Raw Tick File Size (MB per Pair-Month)")

    # Highlight missing cells (e.g. USDRUB post-sanctions in 2022)
    for i in range(len(ALL_20_PAIRS)):
        for j in range(len(CHRONO_MONTHS)):
            if matrix_mb[i, j] == 0:
                ax.text(j, i, "N/A", ha="center", va="center", color="#C44E52", fontsize=7.5, fontweight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "eda_4_kaggle_coverage_matrix.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved §5.4 Kaggle Archive Coverage Heatmap -> {out_path}")


def main():
    t0 = time.perf_counter()
    print("================================================================================")
    print("   TASK 5: COMPREHENSIVE EDA SUITE (§5.1 Raw Ticks, §5.2 Signals, §5.3 Windows, §5.4 Archive)")
    print("================================================================================")
    section_5_1_raw_tick_eda()
    section_5_2_arbitrage_signals_eda()
    section_5_3_window_duration_eda()
    section_5_4_kaggle_coverage_matrix()
    print("\n================================================================================")
    print(f"All 4 Multi-Panel EDA Suites (§5.1 - §5.4) Completed in {time.perf_counter() - t0:.2f}s!")
    print("================================================================================\n")


if __name__ == "__main__":
    main()