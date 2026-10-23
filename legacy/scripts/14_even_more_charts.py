from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_FILE = DATA_DIR / "all_discovered_arbitrage_signals.parquet"

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["figure.dpi"] = 300

def generate_final_charts():
    if not SIGNALS_FILE.exists():
        print(f"Parquet file not found at {SIGNALS_FILE}.")
        return

    df = pl.read_parquet(SIGNALS_FILE)
    inst_df = df.filter(pl.col("institutional_net_bps") > 0.0)

    # -------------------------------------------------------------
    # CHART 7: Intra-Day Session Density by UTC Hour
    # -------------------------------------------------------------
    # Extract UTC hour from timestamp
    hourly_df = (
        inst_df.with_columns([
            pl.col("time").dt.hour().alias("utc_hour")
        ])
        .group_by("utc_hour")
        .agg(pl.len().alias("signal_count"))
        .sort("utc_hour")
        .collect() if isinstance(inst_df, pl.LazyFrame) else
        inst_df.with_columns([
            pl.col("time").dt.hour().alias("utc_hour")
        ])
        .group_by("utc_hour")
        .agg(pl.len().alias("signal_count"))
        .sort("utc_hour")
    )

    hours = list(range(24))
    counts_dict = {row["utc_hour"]: row["signal_count"] for row in hourly_df.to_dicts()}
    counts = [counts_dict.get(h, 0) for h in hours]

    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Color-code trading sessions
    bar_colors = []
    for h in hours:
        if 0 <= h < 8:
            bar_colors.append("#85b3d1")  # Asian / Tokyo Session
        elif 8 <= h < 13:
            bar_colors.append("#4C72B0")  # London Morning
        elif 13 <= h < 17:
            bar_colors.append("#C44E52")  # London / NY Crossover (Peak Volatility)
        elif 17 <= h < 21:
            bar_colors.append("#55A868")  # New York Afternoon
        else:
            bar_colors.append("#B07AA1")  # Sydney / Early Tokyo Rollover

    bars = ax.bar([str(h).zfill(2) + ":00" for h in hours], counts, color=bar_colors, width=0.65)
    ax.set_title("G. Intra-Day Microstructure: Institutional Arbitrage Density by UTC Hour", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("UTC Hour of Day (Session Crossover Highlighted in Red)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Institutional Signal Count (> 0.3 bps)", fontsize=11, fontweight="bold")
    ax.set_xticklabels([str(h).zfill(2) + ":00" for h in hours], rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate peak session
    ax.annotate(
        "London / NY Crossover\n(Peak Liquidity & Rollover Lag)",
        xy=(14, counts[14]), xytext=(15, counts[14] * 1.15),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#ffcccc", ec="#C44E52", lw=1)
    )

    plt.tight_layout()
    out7 = FIGURES_DIR / "chart_7_session_density.png"
    plt.savefig(out7)
    plt.close()
    print(f"Saved: {out7}")

    # -------------------------------------------------------------
    # CHART 8: Cumulative Distribution Function (CDF) of Net Profit
    # -------------------------------------------------------------
    net_bps = inst_df.select("institutional_net_bps").to_series().to_numpy()
    sorted_bps = np.sort(net_bps)
    cdf = np.arange(1, len(sorted_bps) + 1) / len(sorted_bps)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_bps, cdf * 100, color="#2b5c8f", linewidth=2.5)
    ax.set_title("H. Cumulative Distribution Function (CDF) of Institutional Net Profit (bps)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Net Return After 0.3 bps Fee (Basis Points)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative Percentage of Signals (%)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 5)  # Focus on the primary 0 to 5 bps institutional core
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Annotate median and 95th percentile
    med_val = np.median(net_bps)
    p95_val = np.percentile(net_bps, 95)
    ax.axvline(med_val, color="#55A868", linestyle="--", linewidth=1.5, label=f"Median Net Profit: +{med_val:.2f} bps")
    ax.axvline(p95_val, color="#C44E52", linestyle="--", linewidth=1.5, label=f"95th Percentile: +{p95_val:.2f} bps")
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    out8 = FIGURES_DIR / "chart_8_profit_cdf.png"
    plt.savefig(out8)
    plt.close()
    print(f"Saved: {out8}")

    print("\nFinal charts successfully generated and saved to reports/figures/")

if __name__ == "__main__":
    generate_final_charts()