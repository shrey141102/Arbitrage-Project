import numpy as np
import pandas as pd
from pathlib import Path

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

def find_best_naive_cycle(edges_dict):
    """Hardcoded check for the 2 possible triangles to find the highest theoretical mid-price yield."""
    # Cycle 1: EUR(0) -> GBP(1) -> USD(2) -> EUR(0)
    c1 = edges_dict[(0,1)] * edges_dict[(1,2)] * edges_dict[(2,0)]
    # Cycle 2: EUR(0) -> USD(2) -> GBP(1) -> EUR(0)
    c2 = edges_dict[(0,2)] * edges_dict[(2,1)] * edges_dict[(1,0)]
    
    if c1 > 1.0 and c1 > c2: return [0, 1, 2, 0], c1
    if c2 > 1.0: return [0, 2, 1, 0], c2
    return None, 0.0

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
        cycle, theoretical_yield = find_best_naive_cycle(mid_edges_t)
        
        if cycle:
            expected_profit_bps = (theoretical_yield - 1.0) * 10000
            
            # 2. Attempt to execute that exact cycle at time t+1 using the real order book
            inst_edges_t1 = engine.get_institutional_edges(row_t1)
            
            realized_yield = 1.0
            failed_execution = False
            for i in range(len(cycle) - 1):
                edge = (cycle[i], cycle[i+1])
                if edge in inst_edges_t1:
                    realized_yield *= inst_edges_t1[edge]
                else:
                    failed_execution = True
                    break
            
            if failed_execution:
                realized_bps = -5.0 # Severe penalty for a broken leg (inventory risk)
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