from pathlib import Path
from itertools import permutations
import time
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def benchmark_algorithms_live(currencies: list[str], n_iters: int = 2000):
    """Benchmark Bellman-Ford and simple cycle enumeration on one live-sized graph."""
    vertex_count = len(currencies)
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
            best_weight = 0.0
            for cycle in cycles:
                cycle_weight = sum(
                    weights[cycle[i], cycle[(i + 1) % cycle_length]]
                    for i in range(cycle_length)
                )
                if cycle_weight < best_weight:
                    best_weight = cycle_weight
        brute_force_ms.append((time.perf_counter() - start) / n_iters * 1000.0)

    return [bellman_ford_ms] * 3, brute_force_ms, permutation_counts


def generate_charts():
    signals = pl.scan_parquet(DATA_DIR / "all_discovered_arbitrage_signals.parquet")
    windows = pl.read_parquet(DATA_DIR / "all_distinct_windows.parquet")
    by_cycle = pl.read_csv(DATA_DIR / "summary_by_triangle.csv").head(8)

    total_raw_signals = signals.select(pl.len()).collect().item()
    total_institutional_signals = (
        signals
        .filter(pl.col("institutional_net_bps") > 0.0)
        .select(pl.len())
        .collect()
        .item()
    )
    total_windows = windows.height
    windows_over_one_second = windows.filter(
        pl.col("duration_ms") > 1000
    ).height
    counts = [
        total_raw_signals,
        total_institutional_signals,
        total_windows,
        windows_over_one_second,
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=200)
    fig.suptitle(
        f"High-Frequency FX Triangular Arbitrage "
        f"({total_raw_signals:,} Raw Signals, {total_windows:,} Independent Windows)",
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
    ax1.set_ylim(max(min(counts) * 0.2, 1), max(counts) * 3.5)

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
    if total_windows == 0:
        raise ValueError("Cannot build duration chart: no independent windows were found.")
    duration_buckets = [
        windows.filter(pl.col("duration_ms") == 100).height,
        windows.filter(pl.col("duration_ms").is_between(200, 400)).height,
        windows.filter(pl.col("duration_ms").is_between(500, 1000)).height,
        windows_over_one_second,
    ]
    bucket_pcts = [count / total_windows * 100.0 for count in duration_buckets]
    median_duration_ms = int(windows["duration_ms"].median())
    sub_second_pct = 100.0 - bucket_pcts[-1]
    dur_buckets = [
        "100 ms\n(1 Tick)",
        f"200–400 ms\n(Median={median_duration_ms}ms)",
        "500–1000 ms\n(Sub-Second)",
        "> 1000 ms\n(> 1 Second)",
    ]
    b_colors = ["#C44E52", "#DD8452", "#CCB974", "#4C72B0"]
    bars3 = ax3.bar(dur_buckets, bucket_pcts, color=b_colors, width=0.55)
    ax3.set_title(
        f"C. Arbitrage Window Half-Life "
        f"({sub_second_pct:.2f}% Vanish in <= 1.0s)",
        fontweight="bold",
    )
    ax3.set_ylabel(f"Share of {total_windows:,} Independent Windows (%)")
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, pct in zip(bars3, bucket_pcts):
        ax3.text(
            bar.get_x() + bar.get_width() / 2, pct + 0.8,
            f"{pct:.2f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold"
        )
    ax3.set_ylim(0, max(bucket_pcts) * 1.2)

    # Panel D: Bellman-Ford O(V*E) vs Brute-Force O(V^k) Scaling
    ax4 = axes[1, 1]
    benchmark_currencies = sorted({
        currency
        for cycle in by_cycle["cycle"].to_list()
        for currency in cycle.split("->")[:-1]
    })[:5]
    if len(benchmark_currencies) < 5:
        raise ValueError("At least five currencies are required for the complexity benchmark.")
    bf_times, brute_times, permutation_counts = benchmark_algorithms_live(
        benchmark_currencies
    )
    k_labels = [
        f"k = {k}\n({count} perms)"
        for k, count in zip((3, 4, 5), permutation_counts)
    ]
    x = np.arange(len(k_labels))
    w = 0.35
    ax4.bar(x - w/2, brute_times, width=w, label="Brute Force O(V^k)", color="#C44E52")
    ax4.bar(x + w/2, bf_times, width=w, label="Bellman-Ford O(V·E)", color="#55A868")
    ax4.set_xticks(x)
    ax4.set_xticklabels(k_labels)
    ax4.set_title("D. Live Algorithmic Complexity: Bellman-Ford vs. Brute Force", fontweight="bold")
    ax4.set_ylabel("Execution Time per Graph Snapshot (ms)")
    ax4.legend()
    ax4.grid(axis="y", linestyle="--", alpha=0.4)
    for i, (b, bf) in enumerate(zip(brute_times, bf_times)):
        ax4.text(i - w/2, b * 1.03, f"{b/bf:.1f}x slower", ha="center", fontsize=8.5, fontweight="bold")
    ax4.set_ylim(0, max(brute_times) * 1.25)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    FIGURES_DIR = BASE_DIR / "reports" / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_img = FIGURES_DIR / "capstone_empirical_results.png"
    plt.savefig(out_img)
    print(f"Saved publication-grade 4-panel chart to: {out_img}")


if __name__ == "__main__":
    generate_charts()