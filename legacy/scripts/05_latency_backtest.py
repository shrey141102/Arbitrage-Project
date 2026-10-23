import numpy as np
import pandas as pd
from pathlib import Path
from itertools import permutations

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GRID_FILE = DATA_DIR / "aligned_grid_20261014_13.parquet"

CURRENCIES = ["EUR", "GBP", "USD"]
PAIRS = ["EURUSD", "GBPUSD", "EURGBP"]

class ExecutionEngine:
    def __init__(self, currencies, pairs):
        self.currencies = currencies
        self.pairs = pairs
        self.c_to_idx = {c: i for i, c in enumerate(currencies)}
        self.idx_to_c = {i: c for i, c in enumerate(currencies)}
        
    def get_mid_price_edges(self, row):
        """Generates the naive theoretical signals at time t."""
        edges = {}
        for pair in self.pairs:
            u, v = self.c_to_idx[pair[:3]], self.c_to_idx[pair[3:]]
            mid = (row[f"{pair}_bid"] + row[f"{pair}_ask"]) / 2.0
            edges[(u, v)] = mid
            edges[(v, u)] = 1.0 / mid
        return edges
        
    def get_institutional_edges(self, row, fee_bps=0.1, trade_size_m=0.5):
        """Generates the harsh reality of the order book at time t+1."""
        edges = {}
        fee = fee_bps * 1e-4
        for pair in self.pairs:
            u, v = self.c_to_idx[pair[:3]], self.c_to_idx[pair[3:]]
            bid, ask = row[f"{pair}_bid"], row[f"{pair}_ask"]
            bid_vol, ask_vol = row[f"{pair}_bid_vol"], row[f"{pair}_ask_vol"]
            
            # Base -> Quote
            if bid_vol >= trade_size_m * 0.1:
                slip = 0.00005 * max(0.0, (trade_size_m / max(bid_vol, 0.01)) - 1.0)
                edges[(u, v)] = bid * (1.0 - fee) * (1.0 - slip)
                
            # Quote -> Base
            if ask_vol >= trade_size_m * 0.1:
                slip = 0.00005 * max(0.0, (trade_size_m / max(ask_vol, 0.01)) - 1.0)
                edges[(v, u)] = (1.0 / ask) * (1.0 - fee) * (1.0 - slip)
        return edges

def find_best_naive_cycle(edges_dict, num_currencies):
    """Search every simple three-currency cycle for the best mid-price yield."""
    best_cycle = None
    best_yield = 1.0

    for u, v, w in permutations(range(num_currencies), 3):
        if (u, v) not in edges_dict or (v, w) not in edges_dict or (w, u) not in edges_dict:
            continue
        cycle_yield = (
            edges_dict[(u, v)]
            * edges_dict[(v, w)]
            * edges_dict[(w, u)]
        )
        if cycle_yield > best_yield:
            best_cycle = [u, v, w, u]
            best_yield = cycle_yield

    return best_cycle, best_yield

def run_latency_simulation():
    df = pd.read_parquet(GRID_FILE)
    engine = ExecutionEngine(CURRENCIES, PAIRS)
    
    records = []
    
    # Iterate up to the second-to-last row so we can look 1 second ahead
    for t in range(len(df) - 1):
        row_t = df.iloc[t]
        row_t1 = df.iloc[t + 1]
        
        # 1. Look for a theoretical signal at time t
        mid_edges_t = engine.get_mid_price_edges(row_t)
        cycle, theoretical_yield = find_best_naive_cycle(mid_edges_t, len(CURRENCIES))
        
        if cycle:
            expected_profit_bps = (theoretical_yield - 1.0) * 10000
            
            # 2. Attempt to execute that exact cycle at time t+1 using the real order book
            inst_edges_t1 = engine.get_institutional_edges(row_t1)
            
            realized_yield = 1.0
            failed_execution = False
            filled_edges = []
            for i in range(len(cycle) - 1):
                edge = (cycle[i], cycle[i+1])
                if edge in inst_edges_t1:
                    realized_yield *= inst_edges_t1[edge]
                    filled_edges.append(edge)
                else:
                    failed_execution = True
                    break
            
            if failed_execution:
                unwind_factor = realized_yield
                for u, v in reversed(filled_edges):
                    base, quote = engine.idx_to_c[u], engine.idx_to_c[v]
                    reverse_rate = None
                    for pair in engine.pairs:
                        pair_base, pair_quote = pair[:3], pair[3:]
                        if (pair_base, pair_quote) == (base, quote):
                            reverse_rate = row_t1[f"{pair}_ask"]
                            reverse_rate = 1.0 / reverse_rate
                            break
                        if (pair_base, pair_quote) == (quote, base):
                            reverse_rate = row_t1[f"{pair}_bid"]
                            break
                    if reverse_rate is None:
                        unwind_factor = 0.0
                        break
                    unwind_factor *= reverse_rate
                realized_bps = (unwind_factor - 1.0) * 10000
            else:
                realized_bps = (realized_yield - 1.0) * 10000
                
            records.append({
                "time": row_t["time"],
                "expected_bps": expected_profit_bps,
                "realized_bps_1s_delay": realized_bps
            })
            
    results = pd.DataFrame(records)
    
    print("\n========= LATENCY & LOOK-AHEAD BIAS PENALTY =========")
    print(f"Total Naive Signals Attempted : {len(results):,}")
    print(f"Average Expected Profit (t)   : {results['expected_bps'].mean():.3f} bps")
    print(f"Average Realized Return (t+1) : {results['realized_bps_1s_delay'].mean():.3f} bps")
    print(f"Win Rate after 1s delay       : {(results['realized_bps_1s_delay'] > 0).mean() * 100:.1f}%")
    print(f"Worst Case Execution Loss     : {results['realized_bps_1s_delay'].min():.3f} bps")
    print("=====================================================")

if __name__ == "__main__":
    run_latency_simulation()