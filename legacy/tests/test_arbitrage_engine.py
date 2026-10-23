import math
from datetime import datetime, timedelta
from itertools import permutations
import numpy as np
import polars as pl
import pytest


# =====================================================================
# REFERENCE ENGINE (With Bug #1 Fixed: Frozen Parent Snapshot)
# =====================================================================
class ArbitrageGraphEngine:
    """
    Production Bellman-Ford & Polars Vectorized Reference Engine
    used for unit, property-based, and integration testing.
    """
    def __init__(self, currencies: list[str], pairs: list[str], fee_bps: float = 0.0):
        self.currencies = sorted(currencies)
        self.c_to_idx = {c: i for i, c in enumerate(self.currencies)}
        self.idx_to_c = {i: c for i, c in enumerate(self.currencies)}
        self.V = len(self.currencies)
        self.pairs = pairs
        self.fee_bps = fee_bps
        self.fee_mult = 1.0 - (fee_bps * 1e-4)

    def build_graph(self, row: dict, mode: str = "raw_bid_ask") -> tuple[list, dict]:
        edges = []
        rate_lookup = {}
        for pair in self.pairs:
            bid = row.get(f"{pair}_bid")
            ask = row.get(f"{pair}_ask")
            if bid is None or ask is None or math.isnan(bid) or math.isnan(ask):
                continue
            if bid <= 0.0 or ask <= 0.0 or ask < bid:
                continue

            base, quote = pair[:3], pair[3:]
            if base not in self.c_to_idx or quote not in self.c_to_idx:
                continue
            u, v = self.c_to_idx[base], self.c_to_idx[quote]

            if mode == "naive_mid":
                mid = 0.5 * (bid + ask)
                r_uv = mid
                r_vu = 1.0 / mid
            elif mode == "raw_bid_ask":
                r_uv = bid
                r_vu = 1.0 / ask
            elif mode == "institutional":
                r_uv = bid * self.fee_mult
                r_vu = (1.0 / ask) * self.fee_mult
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if r_uv > 0 and r_vu > 0:
                edges.append((u, v, -math.log(r_uv), r_uv))
                edges.append((v, u, -math.log(r_vu), r_vu))
                rate_lookup[(u, v)] = r_uv
                rate_lookup[(v, u)] = r_vu

        return edges, rate_lookup

    def compute_cycle_bps(self, cycle: list[int], rate_lookup: dict) -> float | None:
        prod = 1.0
        for i in range(len(cycle) - 1):
            edge = (cycle[i], cycle[i + 1])
            if edge not in rate_lookup:
                return None
            prod *= rate_lookup[edge]
        return (prod - 1.0) * 1e4

    def detect_negative_cycles(self, edges: list, rate_lookup: dict) -> list[tuple[list[str], float]]:
        """
        Synchronous (Jacobi) Bellman-Ford cycle detector.
        Eliminates Gauss-Seidel edge-ordering bias so every k-hop cycle relaxes
        at the exact same rate (1 hop per pass), guaranteeing order-invariance
        and agreement with Min-Plus Matrix DP.
        """
        dist = [0.0] * self.V
        parent = [-1] * self.V
        eps = 1e-12

        # V - 1 synchronous relaxation passes (order-invariant)
        for _ in range(self.V - 1):
            next_dist = dist.copy()
            next_parent = parent.copy()
            changed = False
            for u, v, w, _ in edges:
                if dist[u] + w < next_dist[v] - eps:
                    next_dist[v] = dist[u] + w
                    next_parent[v] = u
                    changed = True
            dist = next_dist
            parent = next_parent
            if not changed:
                return []

        # V-th detection pass on frozen synchronous state
        frozen_parent = parent.copy()
        candidates = []
        for u, v, w, _ in edges:
            if dist[u] + w < dist[v] - eps:
                candidates.append((u, v))

        verified_cycles = []
        seen_canonical = set()

        for u_trig, v_trig in candidates:
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
                if len(cycle_nodes) >= 3 and len(set(cycle_nodes)) == len(cycle_nodes):
                    min_pos = cycle_nodes.index(min(cycle_nodes))
                    canonical = tuple(cycle_nodes[min_pos:] + cycle_nodes[:min_pos])
                    if canonical not in seen_canonical:
                        full_loop = list(canonical) + [canonical[0]]
                        bps = self.compute_cycle_bps(full_loop, rate_lookup)
                        if bps is not None and bps > 1e-6:
                            seen_canonical.add(canonical)
                            names = [self.idx_to_c[i] for i in full_loop]
                            verified_cycles.append((names, bps))

        return sorted(verified_cycles, key=lambda x: x[1], reverse=True)

    def min_plus_dp_k3(self, edges: list, rate_lookup: dict) -> list[tuple[list[str], float]]:
        """Min-Plus Matrix Multiplication DP for exact k=3 simple cycles."""
        W = np.full((self.V, self.V), np.inf)
        for u, v, w, _ in edges:
            W[u, v] = w
        # Block self-loops so lower-order walks cannot pad themselves
        np.fill_diagonal(W, np.inf)

        # D1 = W, D2 = D1 (x) W, D3 = D2 (x) W
        D2 = np.min(W[:, :, None] + W[None, :, :], axis=1)
        P2 = np.argmin(W[:, :, None] + W[None, :, :], axis=1)
        D3 = np.min(D2[:, :, None] + W[None, :, :], axis=1)
        P3 = np.argmin(D2[:, :, None] + W[None, :, :], axis=1)

        found = []
        seen = set()
        for s in range(self.V):
            if D3[s, s] < -1e-12:
                n2 = int(P3[s, s])
                n1 = int(P2[s, n2])
                cycle_nodes = [s, n1, n2]
                if len(set(cycle_nodes)) == 3:
                    min_pos = cycle_nodes.index(min(cycle_nodes))
                    canonical = tuple(cycle_nodes[min_pos:] + cycle_nodes[:min_pos])
                    if canonical not in seen:
                        seen.add(canonical)
                        full_loop = list(canonical) + [canonical[0]]
                        bps = self.compute_cycle_bps(full_loop, rate_lookup)
                        if bps is not None and bps > 1e-6:
                            names = [self.idx_to_c[i] for i in full_loop]
                            found.append((names, bps))
        return sorted(found, key=lambda x: x[1], reverse=True)


# =====================================================================
# 2.1 UNIT TESTS (T1 - T7): Bellman-Ford Mathematical Properties
# =====================================================================

def test_t1_known_triangle_exact_bps():
    """T1: Construct 3 nodes where EUR->GBP->USD->EUR yields exactly +2.50 bps."""
    engine = ArbitrageGraphEngine(["EUR", "GBP", "USD"], ["EURGBP", "GBPUSD", "EURUSD"])
    target_mult = 1.0 + 2.5e-4  # +2.50 bps
    # EUR -> GBP (Bid=0.85), GBP -> USD (Bid=1.25), USD -> EUR (1/Ask_EURUSD)
    eurgbp_bid = 0.85
    gbpusd_bid = 1.25
    implied_eurusd = eurgbp_bid * gbpusd_bid
    shocked_eurusd_ask = implied_eurusd / target_mult

    row = {
        "EURGBP_bid": eurgbp_bid, "EURGBP_ask": eurgbp_bid * 1.0002,
        "GBPUSD_bid": gbpusd_bid, "GBPUSD_ask": gbpusd_bid * 1.0002,
        "EURUSD_bid": shocked_eurusd_ask * 0.9999, "EURUSD_ask": shocked_eurusd_ask,
    }
    edges, lookup = engine.build_graph(row, mode="raw_bid_ask")
    cycles = engine.detect_negative_cycles(edges, lookup)

    assert len(cycles) == 1
    path, bps = cycles[0]
    assert path == ["EUR", "GBP", "USD", "EUR"]
    assert math.isclose(bps, 2.50, rel_tol=1e-7)


def test_t2_no_arbitrage_equilibrium():
    """T2: Exact cross-rate equilibrium returns zero signals."""
    engine = ArbitrageGraphEngine(["EUR", "GBP", "USD"], ["EURGBP", "GBPUSD", "EURUSD"])
    eurgbp = 0.8600
    gbpusd = 1.2700
    eurusd = eurgbp * gbpusd
    row = {
        "EURGBP_bid": eurgbp, "EURGBP_ask": eurgbp,
        "GBPUSD_bid": gbpusd, "GBPUSD_ask": gbpusd,
        "EURUSD_bid": eurusd, "EURUSD_ask": eurusd,
    }
    edges, lookup = engine.build_graph(row, mode="raw_bid_ask")
    assert engine.detect_negative_cycles(edges, lookup) == []


def test_t3_anti_symmetry_mid_and_spread_shifted_bid_ask():
    """
    T3: Verify exact log anti-symmetry in Mid-Price mode (Forward_Log + Reverse_Log == 0)
    and Spread-Shifted Anti-Symmetry in Bid/Ask mode (Forward_Log + Reverse_Log == sum(ln(Bid/Ask))).
    """
    engine = ArbitrageGraphEngine(["EUR", "GBP", "USD"], ["EURGBP", "GBPUSD", "EURUSD"])
    row = {
        "EURGBP_bid": 0.8505, "EURGBP_ask": 0.8507,
        "GBPUSD_bid": 1.2500, "GBPUSD_ask": 1.2502,
        "EURUSD_bid": 1.0620, "EURUSD_ask": 1.0622,
    }
    e_idx, g_idx, u_idx = engine.c_to_idx["EUR"], engine.c_to_idx["GBP"], engine.c_to_idx["USD"]
    fwd_cycle = [e_idx, g_idx, u_idx, e_idx]
    rev_cycle = [e_idx, u_idx, g_idx, e_idx]

    # 1. Mid-Price Mode: exact log anti-symmetry
    _, mid_lookup = engine.build_graph(row, mode="naive_mid")
    fwd_mid_log = math.log(1.0 + engine.compute_cycle_bps(fwd_cycle, mid_lookup) * 1e-4)
    rev_mid_log = math.log(1.0 + engine.compute_cycle_bps(rev_cycle, mid_lookup) * 1e-4)
    assert math.isclose(fwd_mid_log + rev_mid_log, 0.0, abs_tol=1e-14)

    # 2. Raw Bid/Ask Mode: sum of forward and reverse log-returns equals total log-spread penalty
    _, ba_lookup = engine.build_graph(row, mode="raw_bid_ask")
    fwd_ba_log = math.log(1.0 + engine.compute_cycle_bps(fwd_cycle, ba_lookup) * 1e-4)
    rev_ba_log = math.log(1.0 + engine.compute_cycle_bps(rev_cycle, ba_lookup) * 1e-4)
    expected_spread_drag = sum(
        math.log(row[f"{p}_bid"] / row[f"{p}_ask"]) for p in ["EURGBP", "GBPUSD", "EURUSD"]
    )
    assert math.isclose(fwd_ba_log + rev_ba_log, expected_spread_drag, abs_tol=1e-14)


def test_t4_monotonicity_with_respect_to_costs():
    """T4: Increasing fee_bps from 0.0 -> 0.5 -> 1.0 monotonically reduces profit and signal count."""
    row = {
        "EURGBP_bid": 0.8515, "EURGBP_ask": 0.8516,
        "GBPUSD_bid": 1.2500, "GBPUSD_ask": 1.2501,
        "EURUSD_bid": 1.0624, "EURUSD_ask": 1.0625,  # ~ +17.9 bps raw dislocation
    }
    profits = []
    counts = []
    for fee in [0.0, 0.5, 1.0, 10.0]:
        eng = ArbitrageGraphEngine(["EUR", "GBP", "USD"], ["EURGBP", "GBPUSD", "EURUSD"], fee_bps=fee)
        edges, lookup = eng.build_graph(row, mode="institutional")
        cycles = eng.detect_negative_cycles(edges, lookup)
        counts.append(len(cycles))
        profits.append(cycles[0][1] if cycles else 0.0)

    # Non-increasing signal counts and strictly decreasing positive profits
    assert counts[0] >= counts[1] >= counts[2] >= counts[3]
    assert profits[0] > profits[1] > profits[2] > profits[3] == 0.0


def test_t5_non_simple_walk_and_parent_mutation_isolation():
    """
    T5: Verify non-simple walks (2-cycles / figure-8 walks) are rejected and multiple
    simultaneous independent triangles are both recovered without parent-pointer corruption.
    """
    currencies = ["AUD", "CHF", "EUR", "GBP", "JPY", "USD"]
    pairs = ["EURGBP", "GBPUSD", "EURUSD", "AUDCHF", "CHFJPY", "AUDJPY"]
    eng = ArbitrageGraphEngine(currencies, pairs)

    row = {
        # Triangle 1: EUR->GBP->USD->EUR (+5 bps)
        "EURGBP_bid": 0.8500, "EURGBP_ask": 0.8501,
        "GBPUSD_bid": 1.2500, "GBPUSD_ask": 1.2501,
        "EURUSD_bid": 1.0618, "EURUSD_ask": 1.0619,
        # Triangle 2: AUD->CHF->JPY->AUD (+8 bps)
        "AUDCHF_bid": 0.6500, "AUDCHF_ask": 0.6501,
        "CHFJPY_bid": 150.00, "CHFJPY_ask": 150.02,
        "AUDJPY_bid": 97.410, "AUDJPY_ask": 97.420,
    }
    edges, lookup = eng.build_graph(row, mode="raw_bid_ask")
    cycles = eng.detect_negative_cycles(edges, lookup)

    assert len(cycles) == 2
    for path, _ in cycles:
        assert len(set(path[:-1])) == len(path) - 1  # Strictly simple cycle


def test_t6_reconstruction_round_trip_weight_consistency():
    """T6: Sum of edge weights -ln(rate) along reconstructed path equals -ln(1 + bps * 1e-4)."""
    eng = ArbitrageGraphEngine(["EUR", "GBP", "USD"], ["EURGBP", "GBPUSD", "EURUSD"])
    row = {
        "EURGBP_bid": 0.8510, "EURGBP_ask": 0.8512,
        "GBPUSD_bid": 1.2500, "GBPUSD_ask": 1.2502,
        "EURUSD_bid": 1.0620, "EURUSD_ask": 1.0622,
    }
    edges, lookup = eng.build_graph(row, mode="raw_bid_ask")
    edge_weight_map = {(u, v): w for u, v, w, _ in edges}
    cycles = eng.detect_negative_cycles(edges, lookup)

    assert len(cycles) == 1
    names, bps = cycles[0]
    idxs = [eng.c_to_idx[c] for c in names]
    sum_w = sum(edge_weight_map[(idxs[i], idxs[i + 1])] for i in range(len(idxs) - 1))
    reconstructed_bps = (math.exp(-sum_w) - 1.0) * 1e4
    assert math.isclose(bps, reconstructed_bps, rel_tol=1e-12)


def test_t7_numeraire_scale_invariance():
    """
    T7: Re-scaling a single currency's numeraire value by constant c (e.g. JPY redenomination by 100x)
    shifts individual edge log-weights by +/- ln(c) which cancel out to 0 around any closed cycle.
    """
    eng = ArbitrageGraphEngine(["EUR", "JPY", "USD"], ["EURUSD", "USDJPY", "EURJPY"])
    base_row = {
        "EURUSD_bid": 1.1000, "EURUSD_ask": 1.1001,
        "USDJPY_bid": 150.00, "USDJPY_ask": 150.02,
        "EURJPY_bid": 165.10, "EURJPY_ask": 165.15,  # EUR->JPY->USD->EUR profitable
    }
    c = 137.5  # Arbitrary constant scaling factor for JPY
    scaled_row = {
        "EURUSD_bid": 1.1000, "EURUSD_ask": 1.1001,
        "USDJPY_bid": 150.00 * c, "USDJPY_ask": 150.02 * c,
        "EURJPY_bid": 165.10 * c, "EURJPY_ask": 165.15 * c,
    }
    c1 = eng.detect_negative_cycles(*eng.build_graph(base_row, mode="raw_bid_ask"))
    c2 = eng.detect_negative_cycles(*eng.build_graph(scaled_row, mode="raw_bid_ask"))

    assert len(c1) == len(c2) == 1
    assert c1[0][0] == c2[0][0]
    assert math.isclose(c1[0][1], c2[0][1], rel_tol=1e-11)


# =====================================================================
# 2.2 PROPERTY-BASED TESTS (P1 - P4): Fundamental No-Arbitrage Laws
# =====================================================================

def test_p1_synthetic_equilibrium_1000_random_snapshots():
    """P1: Generate 1,000 random consistent FX state vectors; verify 0 signals across all 1,000."""
    rng = np.random.default_rng(42)
    currencies = ["AUD", "CHF", "EUR", "GBP", "JPY", "USD"]
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "EURGBP", "EURJPY", "EURCHF", "GBPJPY", "AUDJPY"]
    eng = ArbitrageGraphEngine(currencies, pairs)

    for _ in range(1000):
        # Generate random true values v_c relative to USD
        val = {c: rng.uniform(0.5, 2.0) for c in currencies}
        val["USD"] = 1.0
        val["JPY"] = rng.uniform(0.006, 0.010)

        row = {}
        for p in pairs:
            base, quote = p[:3], p[3:]
            true_mid = val[base] / val[quote]
            half_spread = rng.uniform(0.5e-4, 2.0e-4)
            row[f"{p}_bid"] = true_mid * (1.0 - half_spread)
            row[f"{p}_ask"] = true_mid * (1.0 + half_spread)

        edges, lookup = eng.build_graph(row, mode="raw_bid_ask")
        assert eng.detect_negative_cycles(edges, lookup) == []


def test_p2_spread_induced_deadband():
    """P2: Widen all spreads to 5 bps; verify mid-price perturbations <= 5 bps produce 0 signals."""
    rng = np.random.default_rng(99)
    eng = ArbitrageGraphEngine(["EUR", "GBP", "USD"], ["EURGBP", "GBPUSD", "EURUSD"])
    base_mids = {"EURGBP": 0.8500, "GBPUSD": 1.2500, "EURUSD": 0.8500 * 1.2500}

    for _ in range(200):
        row = {}
        for p, mid in base_mids.items():
            # Perturb mid by up to +/- 1.5 bps (total cycle perturbation <= 4.5 bps < 7.5 bps half-spread sum)
            perturbed_mid = mid * (1.0 + rng.uniform(-1.5e-4, 1.5e-4))
            row[f"{p}_bid"] = perturbed_mid * (1.0 - 2.5e-4)  # 5 bps full spread
            row[f"{p}_ask"] = perturbed_mid * (1.0 + 2.5e-4)
        edges, lookup = eng.build_graph(row, mode="raw_bid_ask")
        assert eng.detect_negative_cycles(edges, lookup) == []


def test_p3_single_pair_shock_propagation():
    """
    P3: From realistic micro-spread equilibrium, shock EURJPY Bid up by +10 bps;
    verify the exact 3-hop triangle (EUR-JPY-USD) fires as the unique optimal cycle
    (since routing via a 4th currency GBP incurs extra bid-ask spread).
    """
    eng = ArbitrageGraphEngine(["EUR", "GBP", "JPY", "USD"], ["EURUSD", "USDJPY", "EURJPY", "GBPUSD", "EURGBP"])
    row = {
        "EURUSD_bid": 1.1000, "EURUSD_ask": 1.1000,
        "USDJPY_bid": 150.00, "USDJPY_ask": 150.00,
        # Realistic 1 bps spread on GBP legs prevents zero-cost 4-hop detour
        "GBPUSD_bid": 1.2500 * 0.99995, "GBPUSD_ask": 1.2500 * 1.00005,
        "EURGBP_bid": (1.1000 / 1.2500) * 0.99995, "EURGBP_ask": (1.1000 / 1.2500) * 1.00005,
        "EURJPY_bid": 165.00 * (1.0 + 10.0e-4), "EURJPY_ask": 165.00 * (1.0 + 10.0e-4),
    }
    cycles = eng.detect_negative_cycles(*eng.build_graph(row, mode="raw_bid_ask"))
    assert len(cycles) >= 1
    top_path, top_bps = cycles[0]
    assert set(top_path[:-1]) == {"EUR", "JPY", "USD"}
    assert math.isclose(top_bps, 10.0, rel_tol=1e-6)


def test_p4_transitivity_of_cross_rates():
    """P4: Verify synthetic cross-rate EUR/USD * USD/JPY = 165.00 matches direct EUR/JPY at 0.000 bps."""
    eng = ArbitrageGraphEngine(["EUR", "JPY", "USD"], ["EURUSD", "USDJPY", "EURJPY"])
    row = {
        "EURUSD_bid": 1.10, "EURUSD_ask": 1.10,
        "USDJPY_bid": 150.0, "USDJPY_ask": 150.0,
        "EURJPY_bid": 165.0, "EURJPY_ask": 165.0,
    }
    _, lookup = eng.build_graph(row, mode="raw_bid_ask")
    e, j, u = eng.c_to_idx["EUR"], eng.c_to_idx["JPY"], eng.c_to_idx["USD"]
    assert math.isclose(eng.compute_cycle_bps([e, u, j, e], lookup), 0.0, abs_tol=1e-11)
    assert math.isclose(eng.compute_cycle_bps([e, j, u, e], lookup), 0.0, abs_tol=1e-11)


# =====================================================================
# 2.4 INTEGRATION & REGRESSION TESTS (I1 - I5)
# =====================================================================

def test_i1_variant_consistency_bellman_ford_vs_min_plus_dp():
    """
    I1: Verify Min-Plus Matrix DP and Synchronous Bellman-Ford identify the exact same
    optimal cycle (EUR->JPY->USD->EUR at +11.996 bps) even when a secondary competing
    triangle (EUR->JPY->GBP->EUR at +10.206 bps) shares the EUR->JPY edge.
    """
    currencies = ["AUD", "CHF", "EUR", "GBP", "JPY", "USD"]
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "EURGBP", "EURJPY", "GBPJPY", "AUDUSD", "AUDJPY", "CHFJPY", "USDCHF"]
    eng = ArbitrageGraphEngine(currencies, pairs)

    row = {
        "EURUSD_bid": 1.1000, "EURUSD_ask": 1.1002,
        "GBPUSD_bid": 1.2500, "GBPUSD_ask": 1.2502,
        "USDJPY_bid": 150.00, "USDJPY_ask": 150.02,
        "EURGBP_bid": 0.8800, "EURGBP_ask": 0.8802,
        "EURJPY_bid": 165.25, "EURJPY_ask": 165.28,  # Creates two competing triangles!
        "GBPJPY_bid": 187.50, "GBPJPY_ask": 187.55,
        "AUDUSD_bid": 0.6600, "AUDUSD_ask": 0.6602,
        "AUDJPY_bid": 99.00,  "AUDJPY_ask": 99.04,
        "USDCHF_bid": 0.9000, "USDCHF_ask": 0.9002,
        "CHFJPY_bid": 166.66, "CHFJPY_ask": 166.70,
    }
    edges, lookup = eng.build_graph(row, mode="raw_bid_ask")
    bf_cycles = eng.detect_negative_cycles(edges, lookup)
    dp_cycles = eng.min_plus_dp_k3(edges, lookup)

    assert len(bf_cycles) >= 1 and len(dp_cycles) >= 1
    assert bf_cycles[0][0] == dp_cycles[0][0] == ["EUR", "JPY", "USD", "EUR"]
    assert math.isclose(bf_cycles[0][1], dp_cycles[0][1], rel_tol=1e-12)

def test_i2_no_look_ahead_bias_in_asof_join():
    """I2: Verify Polars backward join_asof strictly uses tick timestamps <= grid timestamp t."""
    t0 = datetime(2022, 4, 1, 12, 0, 0)
    grid = pl.DataFrame({"time": [t0 + timedelta(seconds=i) for i in range(5)]})
    ticks = pl.DataFrame({
        "time": [t0 + timedelta(milliseconds=500), t0 + timedelta(milliseconds=2500)],
        "EURUSD_bid": [1.1000, 1.2000],
        "tick_ts": [t0 + timedelta(milliseconds=500), t0 + timedelta(milliseconds=2500)],
    })
    joined = grid.join_asof(ticks, on="time", strategy="backward").drop_nulls()
    # Every matched tick timestamp MUST be <= the grid timestamp
    assert (joined["tick_ts"] <= joined["time"]).all()
    # At t = +1.0s and +2.0s, price must still be 1.1000 (not the future 1.2000 tick at +2.5s!)
    assert joined.filter(pl.col("time") == t0 + timedelta(seconds=2))["EURUSD_bid"].item() == 1.1000


def test_i3_pipeline_idempotency():
    """I3: Running the cycle detection pipeline twice produces bit-for-bit identical outputs."""
    eng = ArbitrageGraphEngine(["EUR", "GBP", "USD"], ["EURGBP", "GBPUSD", "EURUSD"])
    row = {
        "EURGBP_bid": 0.8520, "EURGBP_ask": 0.8521,
        "GBPUSD_bid": 1.2500, "GBPUSD_ask": 1.2501,
        "EURUSD_bid": 1.0620, "EURUSD_ask": 1.0621,
    }
    res1 = eng.detect_negative_cycles(*eng.build_graph(row, mode="raw_bid_ask"))
    res2 = eng.detect_negative_cycles(*eng.build_graph(row, mode="raw_bid_ask"))
    assert res1 == res2


def test_i4_edge_case_all_nan_hour():
    """I4: Feed a snapshot with all NaNs/None; verify graceful handling (no crash, 0 signals)."""
    eng = ArbitrageGraphEngine(["EUR", "GBP", "USD"], ["EURGBP", "GBPUSD", "EURUSD"])
    nan_row = {
        "EURGBP_bid": float("nan"), "EURGBP_ask": float("nan"),
        "GBPUSD_bid": None, "GBPUSD_ask": None,
        "EURUSD_bid": float("nan"), "EURUSD_ask": float("nan"),
    }
    edges, lookup = eng.build_graph(nan_row, mode="raw_bid_ask")
    assert edges == []
    assert eng.detect_negative_cycles(edges, lookup) == []


def test_i5_edge_case_single_pair_missing():
    """I5: Drop one leg of the triangle (EURGBP missing); verify no crash and 0 signals."""
    eng = ArbitrageGraphEngine(["EUR", "GBP", "USD"], ["EURGBP", "GBPUSD", "EURUSD"])
    incomplete_row = {
        "GBPUSD_bid": 1.2500, "GBPUSD_ask": 1.2501,
        "EURUSD_bid": 1.0500, "EURUSD_ask": 1.0501,
    }
    edges, lookup = eng.build_graph(incomplete_row, mode="raw_bid_ask")
    assert eng.detect_negative_cycles(edges, lookup) == []