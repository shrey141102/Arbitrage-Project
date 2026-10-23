import time
import math
from pathlib import Path
from itertools import permutations
import polars as pl
import numpy as np

# =====================================================================
# 1. CONFIGURATION & UNIVERSE DEFINITION
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data/kaggle"

# 6 Major Currencies -> Up to 15 Cross Pairs -> 30 Directed Edges
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD"]
TARGET_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD",
    "GBPJPY", "GBPCHF", "GBPAUD",
    "CHFJPY", "AUDJPY", "AUDCHF", "AUDNZD"
]


# =====================================================================
# 2. POLARS LAZY CSV LOADER & DYNAMIC ASOF ALIGNMENT PIPELINE
# =====================================================================
def discover_kaggle_csvs(search_dir: Path) -> dict[str, Path]:
    """Finds matching CSV files for our target pairs inside data/ or subfolders."""
    found = {}
    all_csvs = list(search_dir.rglob("*.csv"))
    for pair in TARGET_PAIRS:
        for csv_path in all_csvs:
            # Match pair name ignoring slashes or case (e.g. 'AUD/JPY' matches 'AUDJPY')
            clean_name = csv_path.name.upper().replace("/", "").replace("_", "")
            if pair.upper() in clean_name:
                found[pair] = csv_path
                break
    return found


def load_and_clean_pair_polars(pair: str, csv_path: Path) -> pl.DataFrame:
    """
    Robustly loads Kaggle tick CSVs matching your exact sample schema:
    - column_0: Pair name (e.g., 'AUD/JPY')
    - column_1: Timestamp (e.g., '20200401 00:00:00.035')
    - column_2: Bid price
    - column_3: Ask price
    """
    lf = pl.scan_csv(csv_path, has_header=False, try_parse_dates=False)
    schema_names = lf.collect_schema().names()
    
    time_col = schema_names[1] if len(schema_names) > 1 else schema_names[0]
    bid_col = schema_names[2] if len(schema_names) > 2 else schema_names[1]
    ask_col = schema_names[3] if len(schema_names) > 3 else schema_names[2]
    
    has_vol = len(schema_names) >= 6
    bid_vol_col = schema_names[4] if has_vol else None
    ask_vol_col = schema_names[5] if has_vol else None

    time_expr = (
        pl.col(time_col)
        .str.slice(0, 23)
        .str.to_datetime("%Y%m%d %H:%M:%S%.f", strict=False)
        .alias("time")
    )

    select_exprs = [
        time_expr,
        pl.col(bid_col).cast(pl.Float64, strict=False).alias(f"{pair}_bid"),
        pl.col(ask_col).cast(pl.Float64, strict=False).alias(f"{pair}_ask"),
    ]

    if bid_vol_col and ask_vol_col:
        select_exprs.extend([
            pl.col(bid_vol_col).cast(pl.Float64, strict=False).alias(f"{pair}_bid_vol"),
            pl.col(ask_vol_col).cast(pl.Float64, strict=False).alias(f"{pair}_ask_vol"),
        ])
    else:
        select_exprs.extend([
            pl.lit(1.0).alias(f"{pair}_bid_vol"),
            pl.lit(1.0).alias(f"{pair}_ask_vol"),
        ])

    df = (
        lf.select(select_exprs)
        .drop_nulls(subset=["time", f"{pair}_bid", f"{pair}_ask"])
        .sort("time")
        .unique(subset=["time"], keep="last", maintain_order=True)
        .collect()
    )
    return df


def build_polars_synchronized_grid(freq: str = "1s") -> tuple[pl.DataFrame, list[str]]:
    cache_path = DATA_DIR / "polars_aligned_april2020_dynamic.parquet"
    csv_map = discover_kaggle_csvs(DATA_DIR)

    if not csv_map:
        raise FileNotFoundError(f"No Kaggle CSVs found in {DATA_DIR}. Please check your folder structure.")

    print(f"Discovered {len(csv_map)} Kaggle FX pairs: {list(csv_map.keys())}")
    t0 = time.perf_counter()

    loaded_pairs = {}
    min_times, max_times = [], []

    for pair, path in csv_map.items():
        pair_df = load_and_clean_pair_polars(pair, path)
        if pair_df.height > 0:
            loaded_pairs[pair] = pair_df
            min_times.append(pair_df["time"].min())
            max_times.append(pair_df["time"].max())
            print(f"  -> [{pair:6s}] Loaded {pair_df.height:>8,} ticks from {path.name}")
        else:
            print(f"  -> [{pair:6s}] 0 ticks found.")

    if not loaded_pairs:
        raise ValueError("All loaded pair DataFrames were empty.")

    # Dynamically set start and end to the overlapping active window of the day (e.g., first 2 hours of April 1, 2020)
    from datetime import timedelta
    
    # Dynamically set start and end using standard Python datetime & timedelta
    start_dt_obj = max(min_times)
    end_dt_obj = start_dt_obj + timedelta(hours=2)
    
    start_dt = start_dt_obj.strftime("%Y-%m-%d %H:00:00")
    end_dt = end_dt_obj.strftime("%Y-%m-%d %H:00:00")
    print(f"\nAuto-Detected Active Window: {start_dt} to {end_dt}")

    master_grid = pl.DataFrame({
        "time": pl.datetime_range(
            start=pl.lit(start_dt).str.to_datetime(),
            end=pl.lit(end_dt).str.to_datetime(),
            interval=freq,
            eager=True
        )
    }).sort("time")

    active_pairs = []
    for pair, pair_df in loaded_pairs.items():
        # Filter pair df to window
        subset = pair_df.filter(
            (pl.col("time") >= pl.lit(start_dt).str.to_datetime()) & 
            (pl.col("time") <= pl.lit(end_dt).str.to_datetime())
        )
        if subset.height > 0:
            master_grid = master_grid.join_asof(
                subset,
                on="time",
                strategy="backward",
                tolerance="60s"
            )
            active_pairs.append(pair)

    master_grid = master_grid.drop_nulls()
    master_grid.write_parquet(cache_path)
    elapsed = time.perf_counter() - t0
    print(f"Polars Dynamic Synchronization Complete in {elapsed:.3f}s | Grid Shape: {master_grid.shape}\n")
    return master_grid, active_pairs


# =====================================================================
# 3. EXACT MULTI-HOP BELLMAN-FORD ENGINE
# =====================================================================
class MultiHopArbitrageEngine:
    def __init__(self, currencies: list[str], active_pairs: list[str], fee_bps: float = 0.1, trade_size_m: float = 0.5):
        self.currencies = currencies
        self.active_pairs = active_pairs
        self.V = len(currencies)
        self.c_to_idx = {c: i for i, c in enumerate(currencies)}
        self.idx_to_c = {i: c for i, c in enumerate(currencies)}
        self.fee = fee_bps * 1e-4
        self.trade_size_m = trade_size_m

        self.pair_meta = [
            (pair, self.c_to_idx[pair[:3]], self.c_to_idx[pair[3:]])
            for pair in active_pairs
            if pair[:3] in self.c_to_idx and pair[3:] in self.c_to_idx
        ]

    def build_graph(self, row_dict: dict, mode: str = "institutional"):
        edges = []
        rate_lookup = {}

        for pair, u, v in self.pair_meta:
            bid = row_dict[f"{pair}_bid"]
            ask = row_dict[f"{pair}_ask"]
            bid_vol = row_dict[f"{pair}_bid_vol"]
            ask_vol = row_dict[f"{pair}_ask_vol"]

            if bid <= 0.0 or ask <= 0.0 or ask < bid:
                continue

            if mode == "naive_mid":
                mid = 0.5 * (bid + ask)
                r_uv, r_vu = mid, 1.0 / mid
                edges.append((u, v, -math.log(r_uv), r_uv))
                edges.append((v, u, -math.log(r_vu), r_vu))
                rate_lookup[(u, v)] = r_uv
                rate_lookup[(v, u)] = r_vu

            elif mode == "raw_bid_ask":
                r_uv, r_vu = bid, 1.0 / ask
                edges.append((u, v, -math.log(r_uv), r_uv))
                edges.append((v, u, -math.log(r_vu), r_vu))
                rate_lookup[(u, v)] = r_uv
                rate_lookup[(v, u)] = r_vu

            elif mode == "institutional":
                if bid_vol >= self.trade_size_m * 0.1:
                    slip_uv = 0.00005 * max(0.0, (self.trade_size_m / max(bid_vol, 0.01)) - 1.0)
                    r_uv = bid * (1.0 - self.fee) * (1.0 - slip_uv)
                    edges.append((u, v, -math.log(r_uv), r_uv))
                    rate_lookup[(u, v)] = r_uv

                if ask_vol >= self.trade_size_m * 0.1:
                    slip_vu = 0.00005 * max(0.0, (self.trade_size_m / max(ask_vol, 0.01)) - 1.0)
                    r_vu = (1.0 / ask) * (1.0 - self.fee) * (1.0 - slip_vu)
                    edges.append((v, u, -math.log(r_vu), r_vu))
                    rate_lookup[(v, u)] = r_vu

        return edges, rate_lookup

    def detect_negative_cycles(self, edges: list, rate_lookup: dict) -> list[tuple[list[int], float]]:
        dist = [0.0] * self.V
        parent = [-1] * self.V
        eps = 1e-12

        # V - 1 relaxation passes
        for _ in range(self.V - 1):
            changed = False
            for u, v, w, _ in edges:
                if dist[u] + w < dist[v] - eps:
                    dist[v] = dist[u] + w
                    parent[v] = u
                    changed = True
            if not changed:
                return []

        # V-th detection pass: freeze parent snapshot so detection NEVER mutates parent pointers in-place
        frozen_parent = parent.copy()
        candidates = []
        for u, v, w, _ in edges:
            if dist[u] + w < dist[v] - eps:
                candidates.append((u, v))

        verified_cycles = []
        seen_canonical = set()

        for u_trig, v_trig in candidates:
            # Use an isolated copy of frozen_parent for this specific candidate edge
            local_parent = frozen_parent.copy()
            local_parent[v_trig] = u_trig

            curr = v_trig
            for _ in range(self.V):
                curr = local_parent[curr]
                if curr == -1:
                    break
            if curr == -1:
                continue

            path = []
            visited_pos = {}
            node = curr
            while node != -1 and node not in visited_pos:
                visited_pos[node] = len(path)
                path.append(node)
                node = local_parent[node]

            if node != -1 and node in visited_pos:
                cycle_nodes = path[visited_pos[node]:][::-1]
                if len(cycle_nodes) >= 3:
                    min_pos = cycle_nodes.index(min(cycle_nodes))
                    canonical = tuple(cycle_nodes[min_pos:] + cycle_nodes[:min_pos])

                    if canonical not in seen_canonical:
                        full_loop = list(canonical) + [canonical[0]]
                        bps = self.compute_cycle_bps(full_loop, rate_lookup)
                        if bps is not None and bps > 1e-6:
                            seen_canonical.add(canonical)
                            verified_cycles.append((full_loop, bps))

        return verified_cycles

    @staticmethod
    def compute_cycle_bps(cycle: list[int], rate_lookup: dict) -> float | None:
        prod = 1.0
        for i in range(len(cycle) - 1):
            edge = (cycle[i], cycle[i + 1])
            if edge not in rate_lookup:
                return None
            prod *= rate_lookup[edge]
        return (prod - 1.0) * 1e4


# =====================================================================
# 4. BACKTEST EXECUTION
# =====================================================================
def run_event_driven_backtest():
    csv_map = discover_kaggle_csvs(DATA_DIR)
    loaded_dfs = []
    
    # Load all pairs with precise timestamps
    for pair, path in csv_map.items():
        df = load_and_clean_pair_polars(pair, path)
        if df.height > 0:
            loaded_dfs.append(df)
            
    print(f"\nLoaded {len(loaded_dfs)} active pair streams. Running Event-Driven Tick Engine...")
    
    # Combine and sort all ticks chronologically across all pairs (true event-driven feed)
    # This captures sub-millisecond asynchronous quote lags between dealers
    master_ticks = pl.concat(loaded_dfs, how="diagonal_relaxed").sort("time")
    
    # We slice the first 100,000 raw tick events to inspect sub-second discrepancies
    subset_ticks = master_ticks.head(100000).to_dicts()
    
    active_pairs = list(csv_map.keys())
    active_currencies = sorted(list({p[:3] for p in active_pairs} | {p[3:] for p in active_pairs}))
    engine = MultiHopArbitrageEngine(active_currencies, active_pairs, fee_bps=0.0, trade_size_m=0.1)
    
    signals = []
    # Maintain a rolling state book of the latest price for each pair
    latest_book = {pair: {"bid": 0.0, "ask": 0.0, "bid_vol": 1.0, "ask_vol": 1.0} for pair in active_pairs}
    
    for row in subset_ticks:
        # Identify which pair updated in this tick
        for pair in active_pairs:
            if f"{pair}_bid" in row and row[f"{pair}_bid"] is not None:
                latest_book[pair] = {
                    "bid": row[f"{pair}_bid"],
                    "ask": row[f"{pair}_ask"],
                    "bid_vol": row.get(f"{pair}_bid_vol", 1.0),
                    "ask_vol": row.get(f"{pair}_ask_vol", 1.0),
                }
                
        # Flatten book into dict for graph builder
        flat_row = {}
        valid_book = True
        for pair, vals in latest_book.items():
            if vals["bid"] == 0.0:
                valid_book = False
                break
            flat_row[f"{pair}_bid"] = vals["bid"]
            flat_row[f"{pair}_ask"] = vals["ask"]
            flat_row[f"{pair}_bid_vol"] = vals["bid_vol"]
            flat_row[f"{pair}_ask_vol"] = vals["ask_vol"]
            
        if not valid_book:
            continue
            
        # Check graph instantly on this microsecond tick update (Raw Bid/Ask mode, zero fees)
        edges, lookup = engine.build_graph(flat_row, mode="raw_bid_ask")
        cycles = engine.detect_negative_cycles(edges, lookup)
        
        for cycle, bps in cycles:
            if bps > 0.1:  # Filter for real positive microsecond anomalies
                signals.append({
                    "time": row["time"],
                    "profit_bps": bps,
                    "cycle": " -> ".join([engine.idx_to_c[i] for i in cycle])
                })
                
    sig_df = pl.DataFrame(signals)
    print(f"\n================ EVENT-DRIVEN TICK RESULTS ================")
    print(f"Total Ticks Evaluated : 100,000")
    print(f"Arbitrage Signals Found : {len(sig_df):,}")
    if len(sig_df) > 0:
        print(sig_df.sort("profit_bps", ascending=False).head(10))
    print("===========================================================\n")

if __name__ == "__main__":
    run_event_driven_backtest()