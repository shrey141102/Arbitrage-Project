"""
MASTER SCRIPT - Microstructure-aware FX graph arbitrage (capstone)
===================================================================
One file, run top to bottom:  python master.py   (or paste sections into notebook cells)

  1. CONFIG
  2. DATA ACQUISITION   download Dukascopy tick bid/ask (cached), save snapshot parquet
  3. CLEANING           1-second grid, staleness limit, keep only seconds where ALL pairs are fresh
  4. GRAPH + ALGORITHM  bid/ask directed edges, hop-limited Bellman-Ford with parent pointers
  5. BACKTEST           signal at t, execute at t+lag, cooldown, costs, in-sample / out-of-sample
  6. REPORT             metrics table, latency-decay curve, cost sensitivity

Requirements:  pip install pandas numpy requests pyarrow matplotlib
If data/fx_ticks.parquet exists the script uses it and never touches the network.
"""
import os
import lzma
import struct
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

# =====================================================================
# 1. CONFIG
# =====================================================================
DATA_DIR = os.environ.get("DATA_DIR", "data")
RAW_DIR = f"{DATA_DIR}/raw"                    # cached .bi5 files (part of your snapshot)
SNAPSHOT = f"{DATA_DIR}/fx_ticks.parquet"      # cleaned tick snapshot (submit this)
REPORTS_DIR = os.environ.get("REPORTS_DIR", "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
TABLES_DIR = os.path.join(REPORTS_DIR, "tables")

CCYS = ["GBP", "EUR", "USD", "JPY", "CHF", "AUD"]
PAIRS = {  # Dukascopy symbol: (base, quote)  -> 6 currencies, 15 pairs
    "EURGBP": ("EUR", "GBP"), "EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"), "EURJPY": ("EUR", "JPY"), "GBPJPY": ("GBP", "JPY"),
    "USDCHF": ("USD", "CHF"), "EURCHF": ("EUR", "CHF"), "GBPCHF": ("GBP", "CHF"),
    "CHFJPY": ("CHF", "JPY"), "AUDUSD": ("AUD", "USD"), "EURAUD": ("EUR", "AUD"),
    "GBPAUD": ("GBP", "AUD"), "AUDJPY": ("AUD", "JPY"), "AUDCHF": ("AUD", "CHF"),
}

START_DAY = datetime(2026, 10, 5)              # weekdays between these dates are downloaded
END_DAY = datetime(2026, 10, 16)
HOURS = [12, 13, 14, 15]                       # 12:00-16:00 UTC (London/New York overlap)

STALE_S = 5            # carry a quote forward at most this many seconds
KS = (3, 4)            # cycle lengths to search (3 = triangle). k=5 adds non-simple walks, see notes
TRAIN_FRAC = 0.7       # first 70% of days = in-sample, rest = out-of-sample
NOTIONAL = 1_000_000   # trade size q (in the start currency)
FEE_BPS = 0.0          # extra commission per leg in bps (spread is already inside bid/ask)
SLIP_BPS_PER_M = 0.1   # ASSUMPTION: slippage per leg in bps per 1,000,000 notional
THETAS_BPS = [0.0, 0.25, 0.5, 1.0]   # profit thresholds tried in-sample (this is your "variations tried")
LAG_S = 1              # main latency: execute 1 second after the signal
LAGS_TEST = [0, 1, 2, 5, 10]         # latency-decay curve (0 = instant, unrealistic reference)
COOLDOWN_S = 5         # after a trade attempt, wait this long before the next one


# =====================================================================
# 2. DATA ACQUISITION
# =====================================================================
HEADERS = {"User-Agent": "Mozilla/5.0"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)
PAUSE_S = 1.0          # pause between requests (be gentle: the server throttles fast clients)
FAILED = []            # (pair, hour) that could not be downloaded this run -> NOT cached, retried next run


def fetch_hour(pair, dt, retries=4):
    """Download + decode one hour of Dukascopy ticks (cached on disk). dt is naive UTC.
    Only a real answer (HTTP 200 or 404) is cached. Network failures are never cached."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = f"{RAW_DIR}/{pair}_{dt:%Y%m%d_%H}.bi5"
    cols = ["time", "bid", "ask", "bid_vol", "ask_vol"]
    if os.path.exists(path):
        content = open(path, "rb").read()
    else:
        url = (f"https://datafeed.dukascopy.com/datafeed/{pair}/"
               f"{dt.year}/{dt.month - 1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5")  # month is 0-indexed
        content, answered = b"", False
        for attempt in range(retries):
            try:
                r = SESSION.get(url, timeout=(10, 60))
                if r.status_code == 200:
                    content, answered = r.content, True
                    break
                if r.status_code == 404:
                    answered = True                      # genuinely no data for this hour
                    break
            except requests.exceptions.RequestException as e:
                print(f"  {pair} {dt}: {type(e).__name__}, retry {attempt + 1}")
            time.sleep(5 * (attempt + 1))                # back off: 5s, 10s, 15s, 20s
        if not answered:
            FAILED.append((pair, dt))
            return pd.DataFrame(columns=cols)            # do NOT cache a failure
        open(path, "wb").write(content)                  # empty file = server said "no ticks"
    if not content:
        return pd.DataFrame(columns=cols)
    raw = lzma.decompress(content)
    scale = 1e3 if pair.endswith("JPY") else 1e5
    rows = []
    for i in range(0, len(raw), 20):                     # 20 bytes per tick
        ms, ask, bid, ask_vol, bid_vol = struct.unpack(">IIIff", raw[i:i + 20])
        rows.append((dt + timedelta(milliseconds=ms), bid / scale, ask / scale, bid_vol, ask_vol))
    return pd.DataFrame(rows, columns=cols)


def load_or_download():
    if os.path.exists(SNAPSHOT):
        print(f"Using snapshot {SNAPSHOT}")
        return pd.read_parquet(SNAPSHOT)
    days = [START_DAY + timedelta(days=i) for i in range((END_DAY - START_DAY).days + 1)]
    days = [d for d in days if d.weekday() < 5]
    print(f"Downloading {len(PAIRS)} pairs x {len(days)} days x {len(HOURS)} hours "
          f"= {len(PAIRS) * len(days) * len(HOURS)} requests (cached, resumable)")
    frames, streak = [], 0
    for pair in PAIRS:
        for day in days:
            for h in HOURS:
                n_failed = len(FAILED)
                df = fetch_hour(pair, day.replace(hour=h))
                if len(FAILED) > n_failed:               # this request failed
                    streak += 1
                    if streak >= 3:
                        print("  server seems to be throttling us - cooling down for 120 s")
                        time.sleep(120)
                        streak = 0
                else:
                    streak = 0
                    if len(df):
                        df["pair"] = pair
                        frames.append(df)
                time.sleep(PAUSE_S)
        print(f"  done {pair}  (failed hours so far: {len(FAILED)})")
    if FAILED:
        print(f"\n{len(FAILED)} pair-hours failed, so NO snapshot was saved (it would be incomplete).")
        print("Just run the script again: everything already downloaded is cached, only these are retried.")
        print("First few:", FAILED[:5])
        raise SystemExit(1)
    ticks = pd.concat(frames, ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    ticks.to_parquet(SNAPSHOT)
    print(f"Saved {len(ticks):,} ticks to {SNAPSHOT}")
    return ticks


def verify_against_export(ticks, csv_path, pair="EURGBP"):
    """Compare snapshot 1-minute BID bars with a Dukascopy 1-minute export (CSV from the web dialog)."""
    exp = pd.read_csv(csv_path)
    exp.columns = ["time", "open", "high", "low", "close", "vol"]
    exp["time"] = pd.to_datetime(exp["time"]).dt.tz_localize(None)
    exp = exp.set_index("time")
    g = ticks[ticks.pair == pair].set_index("time").sort_index()
    bars = g["bid"].resample("1min").ohlc().dropna()
    cmp_ = bars.join(exp[["open", "high", "low", "close"]], rsuffix="_exp", how="inner")
    print(f"VERIFY {pair}: {len(cmp_)} overlapping minutes")
    for c in ["open", "high", "low", "close"]:
        d = (cmp_[c] - cmp_[c + "_exp"]).abs()
        print(f"  {c:5s} max abs diff {d.max():.6f} | exact {int((d < 1e-9).sum())}/{len(d)}")


# =====================================================================
# 3. CLEANING  (1-second grid, staleness limit, all pairs must be fresh)
# =====================================================================
def build_grid(ticks, stale_s):
    """Last quote in (s-1, s] is labelled s, so the value at second s uses ONLY information up to s."""
    bids, asks = {}, {}
    for pair, g in ticks.groupby("pair"):
        g = g.set_index("time").sort_index()
        q = g[["bid", "ask"]].resample("1s", closed="right", label="right").last()
        q = q.ffill(limit=stale_s)               # stale quotes older than stale_s become NaN
        bids[pair], asks[pair] = q["bid"], q["ask"]
    bid, ask = pd.concat(bids, axis=1, sort=True), pd.concat(asks, axis=1, sort=True)
    ok = bid.notna().all(axis=1) & ask.notna().all(axis=1)
    return bid[ok], ask[ok]


# =====================================================================
# 4. GRAPH + ALGORITHM
# =====================================================================
def build_weights(bid, ask, fee_bps, slip_bps):
    """W[t,u,v] = -ln(rate u->v after costs). Selling base->quote earns the BID; quote->base pays the ASK."""
    T, n = len(bid), len(CCYS)
    idx = {c: i for i, c in enumerate(CCYS)}
    W = np.full((T, n, n), np.inf)
    cost = (1 - fee_bps / 1e4) * (1 - slip_bps / 1e4)
    for p, (b, q) in PAIRS.items():
        i, j = idx[b], idx[q]
        W[:, i, j] = -np.log(bid[p].to_numpy() * cost)
        W[:, j, i] = -np.log(cost / ask[p].to_numpy())
    return W


def best_cycles(W, ks, chunk=20000):
    """
    Hop-limited Bellman-Ford (min-plus DP): D_k[t,u,v] = cheapest walk of EXACTLY k edges u->v.
    A cycle of length k exists at time t if D_k[t,s,s] < 0. Parent pointers let us rebuild the loop.
    Complexity per time step: O(K * n^3) (here n=6), total O(T * K * n^3).
    Returns best (most negative) cycle per time step: weight, k, start node, plus parent pointers.
    """
    T, n, _ = W.shape
    kmax = max(ks)
    best_w = np.full(T, np.inf)
    best_k = np.zeros(T, dtype=int)
    best_s = np.zeros(T, dtype=int)
    par = {k: np.zeros((T, n, n), dtype=np.int8) for k in range(2, kmax + 1)}
    ar = np.arange(n)
    for a in range(0, T, chunk):
        Wc = W[a:a + chunk]
        D = Wc
        bw, bk, bs = best_w[a:a + chunk], best_k[a:a + chunk], best_s[a:a + chunk]  # views
        for k in range(1, kmax + 1):
            if k >= 2:
                tmp = D[:, :, :, None] + Wc[:, None, :, :]        # [t, u, w, v]
                par[k][a:a + chunk] = tmp.argmin(axis=2)          # predecessor w of v
                D = tmp.min(axis=2)
            if k in ks:
                diag = D[:, ar, ar]
                s = diag.argmin(axis=1)
                w = diag[np.arange(len(diag)), s]
                better = w < bw
                bw[better], bk[better], bs[better] = w[better], k, s[better]
    return best_w, best_k, best_s, par


def reconstruct(par, t, s, k):
    """Follow parent pointers back from (s, hop k) to get the node sequence s -> ... -> s."""
    seq, cur = [s], s
    for j in range(k, 1, -1):
        cur = int(par[j][t, s, cur])
        seq.append(cur)
    seq.append(s)                                 # hop 1 always starts at s
    return seq[::-1]


# =====================================================================
# 5. BACKTEST
# =====================================================================
def run_strategy(W, times, det, theta_bps, lag, row_mask):
    """Signal at row t (info up to t). Execution at the quotes of t+lag seconds. No peeking."""
    best_w, best_k, best_s, par = det
    thr = -np.log1p(theta_bps / 1e4)
    sig = np.where((best_w < thr) & row_mask)[0]
    exec_pos = times.get_indexer(times[sig] + pd.Timedelta(seconds=lag))   # -1 if quote missing/stale
    trades, last, missed = [], None, 0
    for t, e in zip(sig, exec_pos):
        if last is not None and (times[t] - last).total_seconds() < COOLDOWN_S:
            continue
        last = times[t]
        if e < 0:
            missed += 1
            continue
        seq = reconstruct(par, t, int(best_s[t]), int(best_k[t]))
        if len(set(seq[:-1])) != len(seq) - 1:    # skip non-simple loops
            continue
        w_exec = sum(W[e, u, v] for u, v in zip(seq[:-1], seq[1:]))
        trades.append((times[t], ">".join(CCYS[i] for i in seq), int(best_k[t]),
                       float(np.expm1(-best_w[t])), float(np.expm1(-w_exec))))
    tr = pd.DataFrame(trades, columns=["time", "cycle", "k", "profit_signal", "profit_exec"])
    return tr, len(sig), missed


def metrics(trades, days):
    """Daily return = sum of trade returns on the notional (one trade at a time). Sharpe annualised x sqrt(252)."""
    daily = pd.Series(0.0, index=pd.DatetimeIndex(days))
    if len(trades):
        s = trades.groupby(trades["time"].dt.normalize())["profit_exec"].sum()
        daily.loc[s.index] = s.values
    cum = daily.cumsum()
    sd = daily.std()
    sharpe = daily.mean() / sd * np.sqrt(252) if sd and sd > 0 else np.nan
    mdd = float((cum - cum.cummax()).min()) if len(cum) else 0.0
    return {"trades": len(trades), "cum_return": float(cum.iloc[-1]) if len(cum) else 0.0,
            "sharpe": float(sharpe), "max_drawdown": mdd,
            "win_rate": float((trades.profit_exec > 0).mean()) if len(trades) else np.nan,
            "avg_trade_bps": float(trades.profit_exec.mean() * 1e4) if len(trades) else np.nan}


# =====================================================================
# 6. REPORT
# =====================================================================
def main():
    ticks = load_or_download()
    ticks["time"] = pd.to_datetime(ticks["time"])
    if os.environ.get("VERIFY_CSV"):
        verify_against_export(ticks, os.environ["VERIFY_CSV"])

    bid, ask = build_grid(ticks, STALE_S)
    times = bid.index
    print(f"Grid: {len(times):,} usable seconds (all {len(PAIRS)} pairs fresh) "
          f"from {times.min()} to {times.max()}")

    days = sorted(times.normalize().unique())
    n_train = max(1, int(round(len(days) * TRAIN_FRAC)))
    train_days, test_days = days[:n_train], days[n_train:]
    in_mask = times.normalize().isin(train_days)
    out_mask = times.normalize().isin(test_days)
    print(f"In-sample days: {len(train_days)} | out-of-sample days: {len(test_days)}")

    slip = SLIP_BPS_PER_M * NOTIONAL / 1e6
    W = build_weights(bid, ask, FEE_BPS, slip)
    t0 = time.time()
    det = best_cycles(W, KS)
    print(f"Cycle search took {time.time() - t0:.1f}s  [O(T*K*n^3), T={len(times):,}, n={len(CCYS)}, K={max(KS)}]")

    # --- tune threshold IN-SAMPLE only
    scores = {}
    for th in THETAS_BPS:
        tr, _, _ = run_strategy(W, times, det, th, LAG_S, in_mask)
        scores[th] = metrics(tr, train_days)["cum_return"]
    best_theta = max(scores, key=lambda k: (scores[k], -k))
    print(f"\nThresholds tried in-sample: {len(THETAS_BPS)} -> {scores}\nChosen theta = {best_theta} bps")

    # --- main results table
    rows = {}
    for name, mask, dd in [("in-sample", in_mask, train_days), ("out-of-sample", out_mask, test_days)]:
        tr, nsig, missed = run_strategy(W, times, det, best_theta, LAG_S, mask)
        m = metrics(tr, dd)
        m.update({"signals": nsig, "missed_exec": missed})
        rows[name] = m
        if name == "out-of-sample":
            test_trades = tr
    res = pd.DataFrame(rows).T
    print("\n=== KEY RESULTS (lag = %ds, notional = %s) ===" % (LAG_S, f"{NOTIONAL:,}"))
    print(res.to_string(float_format=lambda x: f"{x:.6f}"))
    print("(returns are fractions of notional; with very few days Sharpe is NOT statistically meaningful)")
    if len(test_trades):
        print("\nTop cycles out-of-sample:\n", test_trades.cycle.value_counts().head(5).to_string())

    # --- latency decay curve (out-of-sample)
    decay = []
    for lag in LAGS_TEST:
        tr, nsig, missed = run_strategy(W, times, det, best_theta, lag, out_mask)
        m = metrics(tr, test_days)
        decay.append({"lag_s": lag, "signals": nsig, "trades": m["trades"],
                      "avg_trade_bps": m["avg_trade_bps"], "cum_return": m["cum_return"]})
    decay = pd.DataFrame(decay)
    print("\n=== LATENCY DECAY (out-of-sample) ===\n", decay.to_string(index=False))

    # --- cost sensitivity (out-of-sample): how much extra cost kills it?
    sens = []
    for fee in [0.0, 0.25, 0.5, 1.0]:
        Wf = build_weights(bid, ask, fee, slip)
        detf = best_cycles(Wf, KS)
        tr, _, _ = run_strategy(Wf, times, detf, best_theta, LAG_S, out_mask)
        m = metrics(tr, test_days)
        sens.append({"extra_fee_bps_per_leg": fee, "trades": m["trades"], "cum_return": m["cum_return"]})
    sens = pd.DataFrame(sens)
    print("\n=== COST SENSITIVITY (out-of-sample) ===\n", sens.to_string(index=False))

    # --- save outputs
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    res.to_csv(os.path.join(TABLES_DIR, "key_results.csv"))
    decay.to_csv(os.path.join(TABLES_DIR, "latency_decay.csv"), index=False)
    sens.to_csv(os.path.join(TABLES_DIR, "cost_sensitivity.csv"), index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(decay.lag_s, decay.avg_trade_bps, marker="o")
        plt.axhline(0, color="k", lw=0.5)
        plt.xlabel("execution delay (s)")
        plt.ylabel("avg net profit per trade (bps)")
        plt.title("Arbitrage decay with latency (out-of-sample)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "latency_decay.png"), dpi=150)
    except Exception as e:
        print("plot skipped:", e)
    print(f"\nSaved CSVs and plot to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()