from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def generate_charts():
    by_cycle = pl.read_csv(DATA_DIR / "summary_by_triangle.csv").head(8)
    top_windows = pl.read_parquet(DATA_DIR / "top_1000_distinct_windows.parquet")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=200)
    fig.suptitle(
        "High-Frequency FX Triangular Arbitrage (2020–2022, 340M+ Ticks across 29 Months)",
        fontsize=15, fontweight="bold", y=0.98
    )

    # Panel A: Microstructure Filtering Funnel
    ax1 = axes[0, 0]
    stages = [
        "Raw Executable\nBid/Ask (>0 bps)",
        "Institutional Net\n(>0.3 bps Fee)",
        "De-Clustered\nIndependent Windows",
        "Windows Lasting\n> 1.0 Second"
    ]
    counts = [15_487_220, 6_568_432, 558_856, int(558_856 * (1 - 0.7055))]
    colors = ["#4C72B0", "#2b5c8f", "#55A868", "#C44E52"]
    bars = ax1.bar(stages, counts, color=colors, width=0.55)
    ax1.set_yscale("log")
    ax1.set_title("A. Microstructure & De-Clustering Funnel (Log Scale)", fontweight="bold")
    ax1.set_ylabel("Count (Log Scale)")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, c in zip(bars, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, c * 1.18,
            f"{c:,}", ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
    ax1.set_ylim(1e4, 5e7)

    # Panel B: Top Currency Triangles Leaderboard
    ax2 = axes[0, 1]
    cycles = by_cycle["cycle"].to_list()[::-1]
    ticks_m = (by_cycle["inst_ticks_100ms"] / 1e6).to_list()[::-1]
    p95 = by_cycle["p95_net_bps"].to_list()[::-1]
    bars2 = ax2.barh(cycles, ticks_m, color="#2b5c8f", height=0.6)
    ax2.set_title("B. Top Currency Triangles (Institutional Signals > 0.3 bps)", fontweight="bold")
    ax2.set_xlabel("Institutional 100ms Signals (Millions)")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    for bar, val, p in zip(bars2, ticks_m, p95):
        ax2.text(
            val + 0.04, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}M (95th: +{p:.2f} bps)", va="center", fontsize=8.5
        )
    ax2.set_xlim(0, max(ticks_m) * 1.38)

    # Panel C: Arbitrage Window Duration / Half-Life Breakdown
    ax3 = axes[1, 0]
    dur_buckets = ["100 ms\n(1 Tick)", "200–400 ms\n(Median=400ms)", "500–1000 ms\n(Sub-Second)", "> 1000 ms\n(> 1 Second)"]
    bucket_pcts = [35.05, 18.45, 17.05, 29.45]
    b_colors = ["#C44E52", "#DD8452", "#CCB974", "#4C72B0"]
    bars3 = ax3.bar(dur_buckets, bucket_pcts, color=b_colors, width=0.55)
    ax3.set_title("C. Arbitrage Window Half-Life (70.55% Vanish in <= 1.0s)", fontweight="bold")
    ax3.set_ylabel("Share of 558,856 Independent Windows (%)")
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, pct in zip(bars3, bucket_pcts):
        ax3.text(
            bar.get_x() + bar.get_width() / 2, pct + 0.8,
            f"{pct:.2f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold"
        )
    ax3.set_ylim(0, 42)

    # Panel D: Bellman-Ford O(V*E) vs Brute-Force O(V^k) Scaling
    ax4 = axes[1, 1]
    k_labels = ["k = 3\n(60 perms)", "k = 4\n(120 perms)", "k = 5\n(120 perms)"]
    brute_times = [0.0089, 0.0177, 0.0180]
    bf_times = [0.0009, 0.0009, 0.0009]
    x = np.arange(len(k_labels))
    w = 0.35
    ax4.bar(x - w/2, brute_times, width=w, label="Brute Force O(V^k)", color="#C44E52")
    ax4.bar(x + w/2, bf_times, width=w, label="Bellman-Ford O(V·E)", color="#55A868")
    ax4.set_xticks(x)
    ax4.set_xticklabels(k_labels)
    ax4.set_title("D. Algorithmic Complexity: Bellman-Ford vs. Brute Force", fontweight="bold")
    ax4.set_ylabel("Execution Time per Graph Snapshot (ms)")
    ax4.legend()
    ax4.grid(axis="y", linestyle="--", alpha=0.4)
    for i, (b, bf) in enumerate(zip(brute_times, bf_times)):
        ax4.text(i - w/2, b + 0.0005, f"{b/bf:.1f}x slower", ha="center", fontsize=8.5, fontweight="bold")
    ax4.set_ylim(0, 0.022)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    FIGURES_DIR = BASE_DIR / "reports" / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_img = FIGURES_DIR / "capstone_empirical_results.png"
    plt.savefig(out_img)
    print(f"Saved publication-grade 4-panel chart to: {out_img}")


if __name__ == "__main__":
    generate_charts()