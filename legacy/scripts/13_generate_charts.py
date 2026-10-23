from pathlib import Path
from itertools import permutations
import time
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Global Styling for Publication Quality
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["figure.dpi"] = 300


def benchmark_algorithms_live(vertex_count: int, n_iters: int = 2000):
    rng = np.random.default_rng(42)
    weights = rng.uniform(-0.0005, 0.001, size=(vertex_count, vertex_count))
    np.fill_diagonal(weights, 0.0)
    edges = [
        (u, v, float(weights[u, v]))
        for u in range(vertex_count)
        for v in range(vertex_count)
        if u != v
    ]

    start = time.perf_counter()
    for _ in range(n_iters):
        distances = [0.0] * vertex_count
        for _ in range(vertex_count - 1):
            updated = False
            for u, v, weight in edges:
                if distances[u] + weight < distances[v] - 1e-12:
                    distances[v] = distances[u] + weight
                    updated = True
            if not updated:
                break
    bellman_ford_ms = (time.perf_counter() - start) / n_iters * 1000.0

    brute_force_ms = []
    permutation_counts = []
    for cycle_length in (3, 4, 5):
        cycles = list(permutations(range(vertex_count), cycle_length))
        permutation_counts.append(len(cycles))
        start = time.perf_counter()
        for _ in range(n_iters):
            for cycle in cycles:
                sum(weights[cycle[i], cycle[(i + 1) % cycle_length]] for i in range(cycle_length))
        brute_force_ms.append((time.perf_counter() - start) / n_iters * 1000.0)
    return [bellman_ford_ms] * 3, brute_force_ms, permutation_counts


def generate_standalone_charts():
    by_cycle = pl.read_csv(DATA_DIR / "summary_by_triangle.csv").head(8)
    signals = pl.scan_parquet(DATA_DIR / "all_discovered_arbitrage_signals.parquet")
    windows = pl.read_parquet(DATA_DIR / "all_distinct_windows.parquet")
    total_raw_signals = signals.select(pl.len()).collect().item()
    total_institutional_signals = (
        signals.filter(pl.col("institutional_net_bps") > 0.0)
        .select(pl.len())
        .collect()
        .item()
    )
    total_windows = windows.height
    windows_over_one_second = windows.filter(pl.col("duration_ms") > 1000).height
    counts = [
        total_raw_signals,
        total_institutional_signals,
        total_windows,
        windows_over_one_second,
    ]
    
    # -------------------------------------------------------------
    # CHART 1: The Microstructure & De-Clustering Funnel
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    stages = [
        "Raw Executable\nBid/Ask (>0 bps)",
        "Institutional Net\n(>0.3 bps Fee)",
        "De-Clustered\nIndependent Windows",
        "Windows Lasting\n> 1.0 Second"
    ]
    colors = ["#4C72B0", "#2b5c8f", "#55A868", "#C44E52"]
    
    bars = ax.bar(stages, counts, color=colors, width=0.5)
    ax.set_yscale("log")
    ax.set_title("A. Microstructure Filtering & De-Clustering Funnel", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Signal Count (Log Scale)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, c * 1.15,
            f"{c:,}", ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
    ax.set_ylim(max(min(counts) * 0.2, 1), max(counts) * 3.5)
    plt.tight_layout()
    out1 = FIGURES_DIR / "chart_1_funnel.png"
    plt.savefig(out1)
    plt.close()
    print(f"Saved: {out1}")

    # -------------------------------------------------------------
    # CHART 2: Top Currency Triangles Leaderboard
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 7))
    cycles = by_cycle["cycle"].to_list()[::-1]
    ticks_m = (by_cycle["inst_ticks_100ms"] / 1e6).to_list()[::-1]
    p95 = by_cycle["p95_net_bps"].to_list()[::-1]
    
    bars2 = ax.barh(cycles, ticks_m, color="#2b5c8f", height=0.6)
    ax.set_title("B. Top FX Currency Triangles by Institutional Volume (>0.3 bps)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Institutional 100ms Signals (Millions)", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    for bar, val, p in zip(bars2, ticks_m, p95):
        ax.text(
            val + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}M  (95th Pctl: +{p:.2f} bps)", va="center", fontsize=10, fontweight="bold"
        )
    ax.set_xlim(0, max(ticks_m) * 1.45)
    plt.tight_layout()
    out2 = FIGURES_DIR / "chart_2_triangles.png"
    plt.savefig(out2)
    plt.close()
    print(f"Saved: {out2}")

    # -------------------------------------------------------------
    # CHART 3: Arbitrage Window Half-Life / Duration Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    if total_windows == 0:
        raise ValueError("Cannot build duration chart: no independent windows were found.")
    bucket_counts = [
        windows.filter(pl.col("duration_ms") == 100).height,
        windows.filter(pl.col("duration_ms").is_between(200, 400)).height,
        windows.filter(pl.col("duration_ms").is_between(500, 1000)).height,
        windows_over_one_second,
    ]
    bucket_pcts = [count / total_windows * 100.0 for count in bucket_counts]
    median_duration_ms = int(windows["duration_ms"].median())
    dur_buckets = [
        "100 ms\n(Single Tick)",
        f"200–400 ms\n(Median = {median_duration_ms}ms)",
        "500–1000 ms\n(Sub-Second)",
        "> 1000 ms\n(> 1 Second)",
    ]
    b_colors = ["#C44E52", "#DD8452", "#CCB974", "#4C72B0"]
    
    bars3 = ax.bar(dur_buckets, bucket_pcts, color=b_colors, width=0.55)
    ax.set_title(
        f"C. Arbitrage Window Duration Half-Life "
        f"({100.0 - bucket_pcts[-1]:.2f}% Vanish in ≤ 1.0s)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylabel(f"Share of {total_windows:,} Independent Windows (%)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar, pct in zip(bars3, bucket_pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, pct + 1.0,
            f"{pct:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
    ax.set_ylim(0, max(bucket_pcts) * 1.2)
    plt.tight_layout()
    out3 = FIGURES_DIR / "chart_3_halflife.png"
    plt.savefig(out3)
    plt.close()
    print(f"Saved: {out3}")

    # -------------------------------------------------------------
    # CHART 4: Algorithmic Complexity (Bellman-Ford vs Brute Force)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    bf_times, brute_times, permutation_counts = benchmark_algorithms_live(5)
    k_labels = [
        f"k = {k} ({label}-Hop)\n({count} permutations)"
        for k, label, count in zip((3, 4, 5), ("3", "4", "5"), permutation_counts)
    ]
    x = np.arange(len(k_labels))
    w = 0.35
    
    ax.bar(x - w/2, brute_times, width=w, label="Brute Force O(V^k)", color="#C44E52")
    ax.bar(x + w/2, bf_times, width=w, label="Bellman-Ford O(V·E)", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(k_labels, fontsize=11)
    ax.set_title("D. Algorithmic Scaling: Bellman-Ford vs. Brute-Force Permutation", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Execution Time per Graph Snapshot (ms)", fontsize=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    for i, (b, bf) in enumerate(zip(brute_times, bf_times)):
        ax.text(i - w/2, b * 1.03, f"{b/bf:.1f}x slower", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(brute_times) * 1.25)
    plt.tight_layout()
    out4 = FIGURES_DIR / "chart_4_scaling.png"
    plt.savefig(out4)
    plt.close()
    print(f"Saved: {out4}")

    print("\nAll 4 standalone charts successfully generated and saved to reports/figures/")

if __name__ == "__main__":
    generate_standalone_charts()