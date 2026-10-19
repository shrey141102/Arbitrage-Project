import numpy as np
import pandas as pd
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GRID_FILE = DATA_DIR / "aligned_grid_20261014_13.parquet"

CURRENCIES = ["EUR", "GBP", "USD"]
PAIRS = ["EURUSD", "GBPUSD", "EURGBP"]

class FXArbitrageEngine:
    def __init__(self, currencies, pairs, fee_bps=0.1, trade_size_m=0.5):
        """
        fee_bps: Institutional exchange commission in basis points
        trade_size_m: Notional trade size in millions (Dukascopy volume is in millions)
        """
        self.currencies = currencies
        self.pairs = pairs
        self.V = len(currencies)
        self.c_to_idx = {c: i for i, c in enumerate(currencies)}
        self.idx_to_c = {i: c for i, c in enumerate(currencies)}
        self.fee = fee_bps * 1e-4
        self.trade_size_m = trade_size_m

    def build_edges(self, row):
        """Builds depth-aware directed edges strictly from Bid/Ask."""
        edges = []
        for pair in self.pairs:
            base, quote = pair[:3], pair[3:]
            u, v = self.c_to_idx[base], self.c_to_idx[quote]
            
            bid, ask = row[f"{pair}_bid"], row[f"{pair}_ask"]
            bid_vol, ask_vol = row[f"{pair}_bid_vol"], row[f"{pair}_ask_vol"]
            
            # 1. Base -> Quote (Sell Base, Buy Quote) -> Hits the BID
            if bid_vol >= self.trade_size_m * 0.1:  # Prune if book is empty
                # Linear slippage if our size exceeds top-of-book depth
                slip_uv = 0.00005 * max(0.0, (self.trade_size_m / max(bid_vol, 0.01)) - 1.0)
                eff_rate_uv = bid * (1.0 - self.fee) * (1.0 - slip_uv)
                edges.append((u, v, -np.log(eff_rate_uv), eff_rate_uv))

            # 2. Quote -> Base (Sell Quote, Buy Base) -> Hits 1 / ASK
            if ask_vol >= self.trade_size_m * 0.1:
                slip_vu = 0.00005 * max(0.0, (self.trade_size_m / max(ask_vol, 0.01)) - 1.0)
                eff_rate_vu = (1.0 / ask) * (1.0 - self.fee) * (1.0 - slip_vu)
                edges.append((v, u, -np.log(eff_rate_vu), eff_rate_vu))
                
        return edges

    def bellman_ford_cycles(self, edges):
        """O(V * E) Bellman-Ford algorithm to find negative weight cycles."""
        dist = [0.0] * self.V
        parent = [-1] * self.V
        
        # Relax edges V-1 times
        for _ in range(self.V - 1):
            updated = False
            for u, v, w, _ in edges:
                if dist[u] + w < dist[v] - 1e-12:
                    dist[v] = dist[u] + w
                    parent[v] = u
                    updated = True
            if not updated:
                break

        # V-th iteration: Detect negative cycles & reconstruct the path
        cycles_found = []
        seen_cycles = set()
        
        for u, v, w, _ in edges:
            if dist[u] + w < dist[v] - 1e-12:
                curr = v
                for _ in range(self.V):  # Walk back to guarantee we are inside the cycle
                    curr = parent[curr]
                
                cycle = []
                node = curr
                while True:
                    cycle.append(node)
                    node = parent[node]
                    if node == curr or node in cycle or node == -1:
                        break
                
                if node == curr and len(cycle) >= 3:
                    cycle.reverse()
                    min_idx = cycle.index(min(cycle))
                    canonical = tuple(cycle[min_idx:] + cycle[:min_idx]) # Standardize orientation
                    if canonical not in seen_cycles:
                        seen_cycles.add(canonical)
                        cycles_found.append(list(canonical) + [canonical[0]])
                        
        return cycles_found

def evaluate_cycle_return(cycle, edges_dict):
    """Calculates exact multiplicative net return (in bps) of a cycle."""
    prod = 1.0
    for i in range(len(cycle) - 1):
        u, v = cycle[i], cycle[i + 1]
        if (u, v) not in edges_dict:
            return None # Leg risk: Edge was pruned due to low volume
        prod *= edges_dict[(u, v)]
    return (prod - 1.0) * 10000  # Convert to Basis Points (bps)

def run_engine():
    if not GRID_FILE.exists():
        print(f"Error: Could not find {GRID_FILE}")
        return
        
    df = pd.read_parquet(GRID_FILE)
    print(f"Loaded {len(df)} ticks. Simulating Trade Size: $500k, Fee: 0.1 bps per leg.")
    
    # Initialize Engine
    engine = FXArbitrageEngine(CURRENCIES, PAIRS, fee_bps=0.1, trade_size_m=0.5)
    
    records = []
    start_time = time.time()
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # 1. Build depth-aware graph for this specific second
        edges = engine.build_edges(row)
        edges_dict = {(u, v): rate for u, v, _, rate in edges}
        
        # 2. Run Bellman-Ford
        cycles = engine.bellman_ford_cycles(edges)
        
        # 3. Evaluate any found cycles
        for cycle in cycles:
            ret_bps = evaluate_cycle_return(cycle, edges_dict)
            if ret_bps is not None and ret_bps > 0:
                cycle_str = " -> ".join([engine.idx_to_c[idx] for idx in cycle])
                records.append({
                    "time": row["time"],
                    "cycle": cycle_str,
                    "profit_bps": ret_bps
                })
                
    elapsed = time.time() - start_time
    print(f"\nScan complete in {elapsed:.2f} seconds.")
    
    results = pd.DataFrame(records)
    if not results.empty:
        print(f"Found {len(results)} profitable arbitrage opportunities!")
        print(results.sort_values("profit_bps", ascending=False).head(10))
    else:
        print("No profitable cycles found after accounting for Bid/Ask spread, slippage, and fees.")
        print("This confirms the Efficient Market Hypothesis for this specific hour.")

if __name__ == "__main__":
    run_engine()