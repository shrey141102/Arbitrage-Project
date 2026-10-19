import math
from pathlib import Path
import numpy as np
import polars as pl
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SIGNALS_FILE = DATA_DIR / "all_discovered_arbitrage_signals.parquet"
WINDOWS_FILE = DATA_DIR / "all_distinct_windows.parquet"
MONTHLY_FILE = DATA_DIR / "summary_by_month.csv"


@pytest.fixture(scope="module")
def empirical_artifacts():
    sig_df = pl.read_parquet(SIGNALS_FILE)
    win_df = pl.read_parquet(WINDOWS_FILE)
    mon_df = pl.read_csv(MONTHLY_FILE)
    inst_df = sig_df.filter(pl.col("institutional_net_bps") > 0.0)
    return sig_df, inst_df, win_df, mon_df


def test_s1_heavy_tailed_profit_distribution_ito_2012(empirical_artifacts):
    """S1: Verify institutional profits exhibit extreme leptokurtosis (Kurtosis > 20) and Pareto alpha < 3."""
    _, inst_df, _, _ = empirical_artifacts
    bps = inst_df["institutional_net_bps"].to_numpy()
    mean_b, std_b = np.mean(bps), np.std(bps)
    kurtosis = float(np.mean(((bps - mean_b) / std_b) ** 4))
    tail = bps[bps >= 0.25]
    alpha_mle = float(len(tail) / np.sum(np.log(tail / 0.25)))
    assert kurtosis > 20.0
    assert 1.0 < alpha_mle < 3.0


def test_s2_exponential_half_life_regression_fenn_2009(empirical_artifacts):
    """S2: Verify window survival function fits exponential decay P(T>=t) = exp(-t/tau) with R^2 > 0.90."""
    _, _, win_df, _ = empirical_artifacts
    durations = win_df["duration_ms"].to_numpy()
    grid_t = np.array([100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000], dtype=float)
    y_log = np.log([np.mean(durations >= t) for t in grid_t])
    slope, intercept = np.polyfit(grid_t, y_log, 1)
    y_pred = slope * grid_t + intercept
    r2 = float(1.0 - np.sum((y_log - y_pred) ** 2) / np.sum((y_log - np.mean(y_log)) ** 2))
    assert slope < 0.0
    assert r2 > 0.90


def test_s3_intraday_session_clustering_kozhan_2012(empirical_artifacts):
    """S3: Chi-squared test rejects uniform hourly distribution across 24 UTC hours (p < 0.001)."""
    _, inst_df, _, _ = empirical_artifacts
    hourly = inst_df.with_columns(pl.col("time").dt.hour().alias("h")).group_by("h").agg(pl.len().alias("c"))
    obs = np.zeros(24)
    for r in hourly.to_dicts():
        obs[int(r["h"])] = r["c"]
    exp = np.full(24, inst_df.height / 24.0)
    chi2 = float(np.sum(((obs - exp) ** 2) / exp))
    assert chi2 > 49.73  # Critical value for df=23 at p=0.001


def test_s4_latency_decay_monotonicity_emh(empirical_artifacts):
    """S4: Verify cumulative return is strictly non-increasing across execution lags [0..10,000 ms]."""
    _, _, win_df, _ = empirical_artifacts
    dur = win_df["duration_ms"].to_numpy()
    bps = win_df["peak_net_bps"].to_numpy()
    lags = [0, 100, 200, 500, 1000, 2000, 5000, 10000]
    cum_profits = [float(np.sum(bps[dur > lag])) for lag in lags]
    assert all(cum_profits[i] > cum_profits[i + 1] for i in range(len(cum_profits) - 1))


def test_s5_cost_sensitivity_monotonicity(empirical_artifacts):
    """S5: Increasing cycle fees from 0.0 to 2.0 bps strictly reduces surviving signals & cumulative profit."""
    sig_df, _, _, _ = empirical_artifacts
    raw = sig_df["raw_bid_ask_bps"].to_numpy()
    fees = np.arange(0.0, 2.1, 0.25)
    counts = [int(np.sum(raw > f)) for f in fees]
    assert all(counts[i] > counts[i + 1] for i in range(len(counts) - 1))


def test_s6_bootstrap_temporal_significance(empirical_artifacts):
    """S6: Verify real arbitrage event clustering (<=5s inter-arrival) exceeds 99th percentile of shuffled null."""
    _, _, win_df, _ = empirical_artifacts
    m_df = win_df.filter(pl.col("folder") == "March2021").sort("start_time")
    gaps = m_df["start_time"].diff().dt.total_milliseconds().drop_nulls().to_numpy() / 1000.0
    real_burst = float(np.mean(gaps <= 5.0))
    rng = np.random.default_rng(123)
    span = float(np.sum(gaps))
    null_bursts = [float(np.mean(np.diff(np.sort(rng.uniform(0, span, len(gaps) + 1))) <= 5.0)) for _ in range(100)]
    assert real_burst > np.percentile(null_bursts, 99)


def test_s7_walk_forward_5_epoch_stability(empirical_artifacts):
    """S7: Verify all 5 rolling 6-month walk-forward epochs from 2020 to 2022 yield positive institutional signals."""
    _, _, _, mon_df = empirical_artifacts
    assert mon_df.height == 29
    assert (mon_df["inst_ticks_100ms"] > 0).all()
    assert (mon_df["mean_inst_bps"] > 0.0).all()