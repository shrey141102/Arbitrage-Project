import numpy as np
import pandas as pd
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GRID_FILE = DATA_DIR / "aligned_grid_20261014_13.parquet"

CURRENCIES = ["EUR", "GBP", "USD"]
PAIRS = ["EURUSD", "GBPUSD", "EURGBP"]

class ComparativeFXEngine:
    def __init__(self, currencies, pairs, trade_size_m=0.5):
        self.currencies = currencies
        self.pairs = pairs
        self.V = len(currencies)
        self.c_to_idx = {c: i for i, c in enumerate(currencies)}
        self.idx_to_c = {i: c for i, c in enumerate(currencies)}
        self.trade_size_m = trade_size_m

    def build_edges(self, row, mode="institutional"):
        """Builds graph edges based on the selected realism mode."""
        edges = []
        for pair in self.pairs:
            u, v = self.c_to_idx[pair[:3]], self.c_to_idx[pair[3:]]
            bid, ask = row[f"{pair}_bid"], row[f"{pair}_ask"]
            bid_vol, ask_vol = row[f"{pair}_bid_vol"], row[f"{pair}_ask_vol"]
            
            if mode == "naive_mid":
                # The generic student approach: Mid-prices, infinite liquidity, zero fees
                mid = (bid + ask) / 2.0
                edges.append((u, v, -np.log(mid), mid))
                edges.append((v, u, -np.log(1.0 / mid), 1.0 / mid))
                continue

            fee = 0.00001 if mode == "institutional" else 0.0  # 0.1 bps fee for institutional
            
            # Base -> Quote (Hits BID)
            if bid_vol >= (self.trade_size_m * 0.1 if mode == "institutional" else 0):
                slip = 0.00005 * max(0.0, (self.trade_size_m / max(bid_vol, 0.01)) - 1.0) if mode == "institutional" else 0.0
                rate = bid * (1.0 - fee) * (1.0 - slip)
                edges.append((u, v, -np.log(rate), rate))

            # Quote -> Base (Hits ASK)
            if ask_vol >= (self.trade_size_m * 0.1 if mode == "institutional" else 0):
                slip = 0.00005 * max(0.0, (self.trade_size_m / max(ask_vol, 0.01)) - 1.0) if mode == "institutional" else 0.0
                rate = (1.0 / ask) * (1.0 - fee) * (1.0 - slip)
                edges.append((v, u, -np.log(rate), rate))
                
        return edges

    def bellman_ford_cycles(self, edges):
        dist, parent = [0.0] * self.V, [-1] * self.V
        
        for _ in range(self.V - 1):
            updated = False
            for u, v, w, _ in edges:
                if dist[u] + w < dist[v] - 1e-12:
                    dist[v], parent[v], updated = dist[u] + w, u, True
            if not updated: break

        cycles_found, seen = [], set()
        for u, v, w, _ in edges:
            if dist[u] + w < dist[v] - 1e-12:
                curr = v
                for _ in range(self.V): curr = parent[curr]
                cycle, node = [], curr
                while True:
                    cycle.append(node)
                    node = parent[node]
                    if node == curr or node in cycle or node == -1: break
                if node == curr and len(cycle) >= 3:
                    cycle.reverse()
                    min_idx = cycle.index(min(cycle))
                    canonical = tuple(cycle[min_idx:] + cycle[:min_idx])
                    if canonical not in seen:
                        seen.add(canonical)
                        cycles_found.append(list(canonical) + [canonical[0]])
        return cycles_found

def evaluate_cycle(cycle, edges_dict):
    prod = 1.0
    for i in range(len(cycle) - 1):
        if (cycle[i], cycle[i+1]) not in edges_dict: return None
        prod *= edges_dict[(cycle[i], cycle[i+1])]
    return (prod - 1.0) * 10000

def run_comparison():
    df = pd.read_parquet(GRID_FILE)
    engine = ComparativeFXEngine(CURRENCIES, PAIRS, trade_size_m=0.5)
    
    modes = {
        "naive_mid": "1. Naive Mid-Price (Illusion)",
        "bid_ask_raw": "2. Raw Bid/Ask (Zero Fees)",
        "institutional": "3. Institutional (Fees + Slippage)"
    }
    
    results = {m: 0 for m in modes.keys()}
    avg_profits = {m: [] for m in modes.keys()}
    
    print("Running Microstructure Comparison Benchmark...")
    
    for i in range(len(df)):
        row = df.iloc[i]
        for mode in modes.keys():
            edges = engine.build_edges(row, mode=mode)
            edges_dict = {(u, v): r for u, v, _, r in edges}
            cycles = engine.bellman_ford_cycles(edges)
            
            for cycle in cycles:
                prof = evaluate_cycle(cycle, edges_dict)
                if prof is not None and prof > 0:
                    results[mode] += 1
                    avg_profits[mode].append(prof)

    print("\n========= ARBITRAGE SIGNAL COUNT BY MICROSTRUCTURE =========")
    for mode, desc in modes.items():
        count = results[mode]
        mean_prof = np.mean(avg_profits[mode]) if count > 0 else 0.0
        print(f"{desc:<38} : {count:>5} signals | Avg Profit: {mean_prof:.2f} bps")
    print("============================================================")

if __name__ == "__main__":
    run_comparison()