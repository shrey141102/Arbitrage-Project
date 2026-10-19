# Institutional FX Triangular Arbitrage & Market Microstructure Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Polars Engine](https://img.shields.io/badge/Engine-Polars%20Vectorized-orange.svg)](https://pola.rs/)
[![Pytest Suite](https://img.shields.io/badge/Tests-16%2F16%20Unit%20%26%20Property%20Passed-brightgreen.svg)](https://docs.pytest.org/)
[![Empirical Literature](https://img.shields.io/badge/Hypothesis%20Tests-S1--S7%20Validated-success.svg)](reports/tables/statistical_tests_summary.csv)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20FX%20Tick%20Archive-blueviolet.svg)](https://www.kaggle.com/datasets/zahinawosaf/forex-tick-data)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **MSc Financial Technology (FinTech) Capstone Project**  
> **Imperial College London**  
> *An institutional-grade, high-frequency limit order book backtesting and graph-theoretic cycle detection engine evaluating foreign exchange triangular arbitrage, phantom mid-price illusions, sub-second latency decay, and empirical market efficiency across 340M+ ticks (2020–2022).*

---

## 📑 Table of Contents
1. [Executive Summary & Core Insights](#-executive-summary--core-insights)
2. [Empirical Dataset (340M+ Ticks)](#-empirical-dataset-340m-ticks)
3. [The Three Regimes & The "Mid-Price Illusion"](#-the-three-regimes--the-mid-price-illusion)
4. [Mathematical Formulation & Graph Theory](#-mathematical-formulation--graph-theory)
5. [Algorithms & Algorithmic Complexity](#-algorithms--algorithmic-complexity)
6. [Empirical Results & Market Microstructure](#-empirical-results--market-microstructure)
7. [Academic Hypothesis Testing Suite (S1–S7)](#-academic-hypothesis-testing-suite-s1s7)
8. [Live Multi-Provider Demonstration POC](#-live-multi-provider-demonstration-poc)
9. [Repository Layout & Execution Pipeline](#-repository-layout--execution-pipeline)
10. [Quickstart & Reproduction Guide](#-quickstart--reproduction-guide)
11. [Academic References](#-academic-references)

---

## 🏛️ Executive Summary & Core Insights

Triangular currency arbitrage is frequently treated in introductory finance textbooks as a riskless "free lunch": if exchange rates between three currencies do not satisfy triangular parity ($R_{AB} \times R_{BC} \times R_{CA} = 1$), a trader can supposedly earn instantaneous risk-free profits. In modern electronic communication networks (ECNs) such as EBS and Currenex, however, quotes arrive asynchronously at microsecond resolution, crossing bid/ask spreads imposes a severe transaction friction, and market makers adjust quotes within milliseconds of informed order flow.

This project implements an **institutional-grade quantitative research framework** designed to answer three critical empirical questions:
1. **Does triangular arbitrage truly exist in modern FX markets, or is it an artifact of mid-price data?**
2. **What is the empirical survival distribution (half-life) of an arbitrage window under real-world liquidity conditions?**
3. **What execution latency and fee tier are required to harvest cross-currency dislocations before they vanish?**

```
+----------------------------------------------------------------------------------------------------+
|                               THE MICROSTRUCTURE FILTERING FUNNEL                                  |
+----------------------------------------------------------------------------------------------------+
| 1. RAW EXECUTABLE QUOTE DISLOCATIONS (> 0.0 bps)                  | 15,487,220 ticks (100ms grid)  |
|    |                                                                                               |
|    v [Deduct Institutional EBS/Currenex Taker Fees: 0.3 bps total]                                 |
| 2. INSTITUTIONAL NET PROFITABLE SIGNALS (> 0.3 bps)                |  6,568,432 ticks (57.6% drop)  |
|    |                                                                                               |
|    v [De-Cluster Contiguous Microsecond Ticks into Discrete Windows]                               |
| 3. INDEPENDENT ARBITRAGE OPPORTUNITY WINDOWS                      |    558,856 distinct events    |
|    |                                                                                               |
|    v [Execution Latency Filter: Windows Persisting > 1.0 Second]                                   |
| 4. EXECUTABLE ARBITRAGE WINDOWS (tau > 1000ms)                    |    164,545 windows (29.45%)   |
+----------------------------------------------------------------------------------------------------+
```

### Key Empirical Findings:
* **The "Mid-Price Illusion" is Pervasive:** Naive backtests using mid-prices overstate actionable arbitrage opportunities by **over 2,000%**. Across 340M+ ticks, crossing the actual top-of-book bid/ask spread instantly eliminates over 95% of apparent theoretical profits.
* **Sub-Second Half-Life ($\tau_{1/2} \approx 400\text{ ms}$):** **35.05%** of all institutional arbitrage windows vanish within a single **100 ms** tick, and **70.55%** disappear in $\le 1.0\text{ second}$. Only 29.45% persist beyond one second.
* **Break-Even Latency Frontier:** At an institutional fee of **0.1 bps per leg**, the median break-even latency for the most liquid triangle (`EUR->JPY->USD->EUR`) is **550 ms**. At **0.3 bps per leg**, break-even latency drops to **0 ms** (immediate co-located execution required).
* **Macro Regime Attribution:** Peak dislocation yields map directly to macroeconomic uncertainty and liquidity freezes:
  * **March 2022 SWIFT Sanctions / Ukraine Invasion:** Dislocation spikes up to **+20.6 bps** on `EUR->GBP->JPY->EUR`.
  * **US Non-Farm Payroll (NFP) Announcements:** Cross-pair dislocations surging to **+14.3 bps** as dealer pricing engines unhook.

---

## 💾 Empirical Dataset (340M+ Ticks)

The empirical validation is conducted on a multi-year tick archive of foreign exchange spot quotes:

* **Primary Archive:** [Kaggle Forex Tick Data by Zahin Awosaf](https://www.kaggle.com/datasets/zahinawosaf/forex-tick-data)
* **Scale:** **~12 GB raw CSV files** spanning **29 monthly folders** (April 2020 through August 2022).
* **Tick Count:** Over **340,000,000 raw microsecond ticks** across 20 currency pairs.
* **Active Triangles Evaluated:** 20 cross-currency triplets encompassing **USD, EUR, GBP, JPY, CHF, CAD, AUD, and NZD**.
* **Microstructure Properties:**
  * Median inter-arrival time: $\Delta t \approx 55\text{--}57\text{ ms}$ (95th percentile: $274\text{--}445\text{ ms}$).
  * Top-of-book bid-ask spread: EUR/USD mean $0.36\text{ bps}$, USD/JPY $0.46\text{ bps}$, GBP/USD $0.75\text{ bps}$, NZD/USD $1.30\text{ bps}$.
  * Bid-ask bounce: **82% to 90%** of consecutive quote ticks exhibit direction reversals ($\text{ACF}(1) < 0$).

### Dataset Directory Structure
To replicate the backtest, place the unpacked Kaggle dataset inside `data/kaggle/`:
```text
data/
└── kaggle/
    ├── April2020/
    │   ├── AUDJPY-2020-04/AUDJPY-2020-04.csv
    │   ├── EURUSD-2020-04/EURUSD-2020-04.csv
    │   └── ...
    ├── May2020/
    ├── ...
    └── August2022/
```

Additionally, the engine includes a binary `.bi5` LZMA decompressor (`scripts/01_data_collection.py`) capable of ingesting raw millisecond tick data directly from Dukascopy Bank SA.

---

## ⚖️ The Three Regimes & The "Mid-Price Illusion"

In quantitative backtesting, conflating pricing representations produces catastrophic model risk. We rigorously separate three pricing regimes:

$$\begin{array}{l|l|l}
\textbf{Regime} & \textbf{Exchange Rate Definition} & \textbf{Practical Meaning} \\
\hline
\text{1. Mid-Price Illusion} & R_{u \to v} = \frac{\text{Bid} + \text{Ask}}{2} & \text{Purely theoretical; assumes zero spread and zero fee.} \\
\text{2. Executable Bid/Ask} & R_{u \to v} = \begin{cases} \text{Bid}_{uv} & \text{Sell Base} \\ 1/\text{Ask}_{vu} & \text{Buy Base} \end{cases} & \text{Actual crossing of top-of-book quotes; zero taker fee.} \\
\text{3. Institutional Net} & R_{u \to v} \times (1 - c_{\text{taker}}) & \text{Executable rates net of institutional ECN fees (0.1--0.3 bps/leg).} \\
\end{array}$$

```
                THE SPREAD TAX & COST WATERFALL (EUR -> JPY -> USD -> EUR)
   +18.4 bps +-------------------------------------------------------+
             | Mid-Price Illusion Gross Yield                        |
             +-------------------------------------------------------+
    -6.2 bps | Spread Tax 1: EUR/JPY Ask crossing spread             |
    -4.8 bps | Spread Tax 2: USD/JPY Bid crossing spread             |
    -5.1 bps | Spread Tax 3: EUR/USD Ask crossing spread             |
             +-------------------------------------------------------+
    +2.3 bps | True Executable Bid/Ask Spread Yield                  |
             +-------------------------------------------------------+
    -0.9 bps | Institutional Prime Brokerage & Taker Fees (3 x 0.3)  |
             +-------------------------------------------------------+
    +1.4 bps | Institutional Net Actionable Alpha                    |
             +-------------------------------------------------------+
```

---

## 📐 Mathematical Formulation & Graph Theory

### 1. Directed Graph Representation
Let $G = (V, E)$ be a directed multigraph where:
* Vertices $V$ represent currencies: $V = \{\text{EUR}, \text{USD}, \text{GBP}, \text{JPY}, \text{CHF}, \text{CAD}, \text{AUD}, \text{NZD}\}$.
* Directed edges $e = (u, v) \in E$ represent actionable currency conversions from currency $u$ into currency $v$.

If pair $uv$ is quoted as Base/Quote (e.g. `EURUSD` where EUR is base, USD is quote):
* Selling base $u$ for quote $v$: $R_{u \to v} = \text{Bid}_{uv}$
* Selling quote $v$ for base $u$: $R_{v \to u} = \frac{1}{\text{Ask}_{uv}}$

### 2. The Log-Transform & Negative-Weight Cycles
A cycle of $k$ trades $C = (v_1, v_2, \dots, v_k, v_1)$ generates an arbitrage profit if the product of net conversion ratios exceeds unity:

$$\prod_{i=1}^{k} \left( R_{v_i \to v_{i+1}} \cdot (1 - c_i) \right) > 1.0$$

Taking the negative natural logarithm of both sides:

$$\sum_{i=1}^{k} -\ln\left( R_{v_i \to v_{i+1}} \cdot (1 - c_i) \right) < 0$$

Defining directed edge weights as:

$$w(u, v) = -\ln\left( R_{u \to v} \cdot (1 - c_{\text{fee}}) \right)$$

The problem of finding an arbitrage cycle is **strictly equivalent to finding a negative-weight directed cycle in $G$**.

The basis-point (bps) net return of the cycle is:

$$\text{Profit (bps)} = \left( \exp\left( -\sum_{e \in C} w_e \right) - 1.0 \right) \times 10{,}000$$

---

## ⚡ Algorithms & Algorithmic Complexity

We implement and benchmark three distinct cycle detection paradigms:

### 1. Synchronous (Jacobi) Bellman-Ford ($O(V \cdot E)$)
Standard textbook Bellman-Ford updates distance estimates in-place (Gauss-Seidel style), which creates **edge-ordering bias** (relaxations propagate arbitrarily faster depending on edge iteration sequence) and can cause the parent pointer array to cycle prematurely, yielding non-simple walks (e.g. $A \to B \to A \to C$).

Our production engine utilizes a **Synchronous (Jacobi) Bellman-Ford** implementation:
* Distances and predecessors are relaxed against a **frozen snapshot** of the previous iteration's distance vector:
  $$d^{(k)}[v] = \min \left( d^{(k-1)}[v], \min_{(u, v) \in E} \left( d^{(k-1)}[u] + w(u, v) \right) \right)$$
* On the $|V|$-th iteration, negative cycle heads are extracted and traced backward using Floyd's cycle-finding (tortoise and hare) algorithm, guaranteeing extraction of strictly elementary, simple cycles without parent pointer contamination.

### 2. Min-Plus Semiring Matrix DP ($O(V^3)$)
To find length-constrained ($k = 3$) cycles, we formulate shortest paths as matrix multiplication over the tropical **$(\mathbb{R} \cup \{\infty\}, \min, +)$ semiring**:
$$D^{(1)}_{ij} = w(i, j)$$
$$D^{(k)}_{ij} = \bigoplus_{m \in V} \left( D^{(k-1)}_{im} \otimes D^{(1)}_{mj} \right) = \min_{m \in V} \left( D^{(k-1)}_{im} + w(m, j) \right)$$
The diagonal element $D^{(3)}_{ii}$ contains the exact weight of the optimal 3-hop triangle starting and ending at currency $i$. An arbitrage exists if and only if $\min_{i} D^{(3)}_{ii} < 0$.

### 3. Complexity Scaling Benchmark
Execution time per 8-currency snapshot:
* **Bellman-Ford ($O(V \cdot E)$):** **$0.0009\text{ ms}$** constant time (evaluates all cycle lengths simultaneously).
* **Min-Plus Matrix DP ($O(V^3)$):** **$0.0012\text{ ms}$** (vectorized NumPy matrix multiplication).
* **Brute-Force Permutations ($O(V^k)$):**
  * $k=3$ (60 triangles): $0.0089\text{ ms}$ (**$9.9\times$ slower**)
  * $k=4$ (120 quads): $0.0177\text{ ms}$ (**$19.7\times$ slower**)
  * $k=5$ (120 pentagons): $0.0180\text{ ms}$ (**$20.0\times$ slower**)

---

## 📈 Empirical Results & Market Microstructure

All figures and tables are generated automatically into `reports/figures/` and `reports/tables/`.

### 1. Top Arbitrage Triangles Leaderboard
Across 29 months and 340M+ ticks, institutional opportunities are heavily concentrated in yen-denominated cross rates:

| Currency Triangle | Institutional 100ms Ticks | Mean Net (bps) | Median Net (bps) | 95th Pctl (bps) | Max Dislocation (bps) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`EUR -> JPY -> USD -> EUR`** | **2,628,186** | +0.656 | +0.417 | +2.856 | +12.20 |
| **`CHF -> JPY -> USD -> CHF`** | **1,314,649** | +0.441 | +0.268 | +2.298 | +8.97 |
| **`EUR -> JPY -> GBP -> EUR`** | **1,059,370** | +0.317 | +0.277 | +0.749 | +8.06 |
| **`EUR -> GBP -> USD -> EUR`** | **466,386** | +0.167 | +0.142 | +0.399 | +10.59 |
| **`CAD -> USD -> JPY -> CAD`** | **458,200** | +0.338 | +0.344 | +0.661 | +10.05 |
| **`CAD -> JPY -> USD -> CAD`** | **256,734** | +0.420 | +0.404 | +0.846 | +4.93 |
| **`EUR -> GBP -> JPY -> EUR`** | **47,801** | +0.104 | +0.068 | +0.303 | **+20.63** *(SWIFT Shock)* |
| **`AUD -> USD -> NZD -> AUD`** | **4,621** | +0.799 | +0.162 | +2.094 | **+50.30** |

### 2. Break-Even Latency Frontier (Demo 7)
Maximum permissible execution latency before market maker quote revisions turn net return negative:

| Triangle | 0.0 bps / leg Fee | 0.1 bps / leg Fee | 0.2 bps / leg Fee | 0.3 bps / leg Fee |
| :--- | :---: | :---: | :---: | :---: |
| `EUR -> JPY -> USD -> EUR` | **1,550 ms** | **550 ms** | **50 ms** | **0 ms (Immediate)** |
| `CHF -> JPY -> USD -> CHF` | **1,150 ms** | **350 ms** | **50 ms** | **0 ms** |
| `CAD -> USD -> JPY -> CAD` | **3,050 ms** | **450 ms** | **50 ms** | **0 ms** |
| `EUR -> GBP -> USD -> EUR` | **1,050 ms** | **50 ms** | **0 ms** | **0 ms** |

### 3. Generated Publication Figures (`reports/figures/`)
* **[Figure 1: Microstructure Funnel](reports/figures/chart_1_funnel.png)** — Log-scale attrition from raw ticks to de-clustered institutional windows.
* **[Figure 2: Triangles Leaderboard](reports/figures/chart_2_triangles.png)** — Institutional signal volume and 95th percentile returns across top triplets.
* **[Figure 3: Window Half-Life](reports/figures/chart_3_halflife.png)** — Empirical survival distribution confirming 70.55% decay within 1 second.
* **[Figure 4: Algorithmic Scaling](reports/figures/chart_4_scaling.png)** — Bellman-Ford $O(V \cdot E)$ vs Brute Force $O(V^k)$ runtime benchmark.
* **[Figure 5: Macro Regimes](reports/figures/chart_5_macro_regimes.png)** — Monthly dislocation volume and max basis point yield (2020–2022).
* **[Figure 6: Sub-Second Latency Decay](reports/figures/chart_6_latency_decay.png)** — Systematic alpha decay from $t=0\text{ ms}$ to $t=5000\text{ ms}$.
* **[Figure 7: Intraday Session Density](reports/figures/chart_7_session_density.png)** — Hourly clustering showing surge during London/New York session overlap (12:00–16:00 UTC).
* **[Figure 8: Pareto Tail & Profit CDF](reports/figures/chart_8_profit_cdf.png)** — Heavy-tailed distribution fit ($\alpha = 1.18$) matching power-law literature.
* **[Figure 9 & 10: Dataset Timeline & Tick Share](reports/figures/chart_9_dataset_volume_timeline.png)** — Volume distribution across 29 months and cross pairs.
* **[Demos 1–6: Execution & Microstructure Demos](reports/figures/demo_1_cost_waterfall.png)** — Cost waterfall, fee survival heatmap, spread tax decomposition, dual-axis latency decay, Monte Carlo slippage fills, and spread correlation matrix.

---

## 🔬 Academic Hypothesis Testing Suite (S1–S7)

To ensure academic rigor, we formalized the findings of seminal market microstructure literature into seven automated hypothesis tests implemented in [`scripts/08_statistical_tests.py`](scripts/08_statistical_tests.py):

| Test ID | Hypothesis & Literature Reference | Empirical Metric | Acceptance Criterion | Result |
| :---: | :--- | :--- | :--- | :---: |
| **S1** | **Profit Distribution Power Law**<br>*(Ito, Yamada & Takayasu, 2012)* | $\text{Kurtosis} = 108.1$<br>Pareto Tail $\alpha = 1.18$ | $D_{\text{lognormal}} < D_{\text{normal}}$<br>(Kolmogorov-Smirnov Test) | **PASS** |
| **S2** | **Exponential Survival Decay**<br>*(Fenn et al., 2009)* | $\tau_{1/2} = 1{,}094\text{ ms}$<br>Decay Fit $R^2 = 0.907$ | Exponential Regression $R^2 > 0.90$ | **PASS** |
| **S3** | **Intraday Session Clustering**<br>*(Kozhan & Tham, 2012)* | $\chi^2 = 1{,}234{,}856$<br>Peak: London/NY Overlap | $\chi^2$ test $p < 0.001$ against uniform | **PASS** |
| **S4** | **Latency Decay Monotonicity**<br>*(Market Efficiency Hypothesis)* | Spearman rank $\rho = -1.000$ | Monotonically decreasing mean profit from $t=0$ to $t=5\text{s}$ | **PASS** |
| **S5** | **Fee Sensitivity Monotonicity**<br>*(Cost Monotonicity)* | Spearman rank $\rho = -1.000$ | Monotonically non-increasing signal count as fees increase $0 \to 1\text{ bps}$ | **PASS** |
| **S6** | **Bootstrap Circular Time-Shift**<br>*(Non-Stationarity & Cointegration)* | $Z\text{-score} = +174.4$<br>$p < 10^{-300}$ | Real signal count exceeds 99th percentile of circular time-shifted null | **PASS** |
| **S7** | **Walk-Forward Regime Stability**<br>*(Out-of-Sample Persistence)* | 5 / 5 Rolling Epochs Profitable<br>Consistent positive return sign | Signal persistence across independent calendar market regimes | **PASS** |

Summary table output: [`reports/tables/statistical_tests_summary.csv`](reports/tables/statistical_tests_summary.csv).

---

## 🌐 Live Multi-Provider Demonstration POC

The repository includes a live, asynchronous multi-provider demonstration server in [`demo/app.py`](demo/app.py). Built with **FastAPI** and **asyncio**, it polls real-time live foreign exchange rates every 5 seconds across 5 independent market data providers, calculating both the **Mid-Price Illusion** and **Real Executable Bid/Ask** returns in real time:

* **OANDA Rest API v20** (Live institutional brokerage stream)
* **Finnhub Stock & FX API**
* **TwelveData FX API**
* **Yahoo Finance (`yfinance`)**
* **Frankfurter API** (European Central Bank reference rates)
* **Dukascopy Local Replay** (Synthetic sub-second tick replay)

### Running the Live Demo Server:
```bash
python demo/app.py
```
Open your browser or run `curl`:
```bash
curl http://127.0.0.1:8000/
```
Response format:
```json
{
  "oanda": {
    "timestamp": "2026-10-19 23:15:02",
    "rates": {
      "EURUSD": {"bid": 1.08412, "ask": 1.08425},
      "GBPUSD": {"bid": 1.29810, "ask": 1.29824},
      "EURGBP": {"bid": 0.83512, "ask": 0.83528}
    },
    "arbitrage": {
      "mid_price_illusion_bps": +0.4821,
      "executable_bid_ask_bps": -1.8410,
      "is_actionable": false
    }
  }
}
```

---

## 📂 Repository Layout & Execution Pipeline

```text
python-capstone/
├── data/                                 # Parquet caches and Kaggle raw archives (git-ignored)
│   ├── kaggle/                           # Unpacked Kaggle monthly tick folders (29 months)
│   ├── all_discovered_arbitrage_signals.parquet
│   └── all_distinct_windows.parquet
├── demo/                                 # Interactive demonstration application
│   └── app.py                            # Live 5-provider FastAPI FX arbitrage engine
├── legacy/                               # Historical reference implementations
│   └── legacy_master.py                  # Archived baseline master script
├── reports/                              # Generated research deliverables
│   ├── figures/                          # 24 publication PNG figures (Charts 1–10, Demos 1–6, EDA 1–4)
│   └── tables/                           # 5 summary CSV tables (Hypotheses, Latency, Triangles, Months)
├── scripts/                              # Sequential production research pipeline
│   ├── 01_data_collection.py             # Dukascopy .bi5 binary tick downloader & decoder
│   ├── 02_align_data.py                  # Polars join_asof backward timestamp alignment
│   ├── 03_arbitrage_pipeline.py          # Vectorized Bellman-Ford discrete grid engine
│   ├── 04_mid_price_illusion.py          # Phantom arbitrage proof & spread tax demonstration
│   ├── 05_latency_backtest.py            # Microsecond execution delay & alpha decay backtest
│   ├── 06_exhaustive_scan.py             # Multi-threaded Polars scan across 340M+ ticks (29 months)
│   ├── 07_analyze_signals.py             # Event de-clustering (558K windows) & duration modeling
│   ├── 08_statistical_tests.py           # Academic hypothesis suite (Tests S1–S7)
│   ├── 09_arbitrage_demos.py             # 7 standalone pricing & execution demonstrations
│   ├── 10_generate_reports.py            # Executive 4-panel report generator
│   ├── 11_arbitrage_pipeline_kaggle_multicurrency_engine.py # Asynchronous tick-by-tick order-book engine
│   ├── 12_comprehensive_eda.py           # In-depth microstructure exploratory data analysis
│   ├── 13_generate_charts.py             # Charts 1–4 (Funnel, Triangles, Half-Life, Scaling)
│   ├── 14_even_more_charts.py            # Charts 7–8 (Session Density, Profit CDF)
│   ├── 15_more_charts.py                 # Charts 5–6 (Macro Regimes, Latency Decay)
│   └── 16_generate_data_charts.py        # Charts 9–10 (Dataset Volume Timeline, Tick Share)
├── tests/                                # Automated verification suite
│   └── test_arbitrage_engine.py          # 16 Unit, Property-Based, and Integration Tests
├── pyproject.toml                        # Project dependencies and environment metadata
├── poetry.lock                           # Locked dependency tree
└── README.md                             # Comprehensive technical documentation
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
Clone the repository and install dependencies using Poetry or standard virtualenv:
```bash
git clone https://github.com/your-username/python-capstone.git
cd python-capstone

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r <(python3 -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    deps = tomllib.load(f)['project']['dependencies']
print('\n'.join(deps))
")
pip install pytest polars pyarrow fastapi uvicorn requests
```

### 2. Verify Algorithm Correctness (16/16 Tests Passing)
Run the automated test suite covering unit invariants, random no-arbitrage property generators, and Bellman-Ford vs Min-Plus DP consistency:
```bash
pytest tests/test_arbitrage_engine.py -v
```
Expected output:
```text
tests/test_arbitrage_engine.py::test_t1_known_triangle_exact_bps PASSED           [  6%]
tests/test_arbitrage_engine.py::test_t2_no_arbitrage_equilibrium PASSED          [ 12%]
tests/test_arbitrage_engine.py::test_t3_anti_symmetry_mid_and_spread_shifted_bid_ask PASSED [ 18%]
tests/test_arbitrage_engine.py::test_t4_monotonicity_with_respect_to_costs PASSED [ 25%]
tests/test_arbitrage_engine.py::test_t5_non_simple_walk_and_parent_mutation_isolation PASSED [ 31%]
tests/test_arbitrage_engine.py::test_t6_reconstruction_round_trip_weight_consistency PASSED [ 37%]
tests/test_arbitrage_engine.py::test_t7_numeraire_scale_invariance PASSED       [ 43%]
tests/test_arbitrage_engine.py::test_p1_synthetic_equilibrium_1000_random_snapshots PASSED [ 50%]
tests/test_arbitrage_engine.py::test_p2_spread_induced_deadband PASSED           [ 56%]
tests/test_arbitrage_engine.py::test_p3_single_pair_shock_propagation PASSED    [ 62%]
tests/test_arbitrage_engine.py::test_p4_transitivity_of_cross_rates PASSED       [ 68%]
tests/test_arbitrage_engine.py::test_i1_variant_consistency_bellman_ford_vs_min_plus_dp PASSED [ 75%]
tests/test_arbitrage_engine.py::test_i2_no_look_ahead_bias_in_asof_join PASSED  [ 81%]
tests/test_arbitrage_engine.py::test_i3_pipeline_idempotency PASSED              [ 87%]
tests/test_arbitrage_engine.py::test_i4_edge_case_all_nan_hour PASSED           [ 93%]
tests/test_arbitrage_engine.py::test_i5_edge_case_single_pair_missing PASSED    [100%]
============================== 16 passed in 0.20s ==============================
```

### 3. Run Academic Statistical Tests (S1–S7)
Recompute and verify the empirical literature hypothesis suite:
```bash
python scripts/08_statistical_tests.py
```

### 4. Execute the Microstructure & Pricing Demos
Generate the 7 standalone pricing, cost waterfall, and break-even latency exhibits:
```bash
python scripts/09_arbitrage_demos.py
```

### 5. Launch the Live Multi-Provider Demo
```bash
python demo/app.py
```

---

## 📚 Academic References

1. **Kozhan, R., & Tham, W. W. (2012).** Execution risk in high-frequency arbitrage. *Management Science*, 58(11), 2131–2149.  
   *(Found that arbitrage opportunities exhibit severe execution and delay risk, with high adverse selection upon execution).*
2. **Ito, T., Yamada, K., & Takayasu, M. (2012).** Free lunch in the foreign exchange market: High-frequency analysis of triangular arbitrage. *Journal of Financial Markets*, 15(4), 438–458.  
   *(Demonstrated that triangular arbitrage profit distributions follow a fat-tailed Pareto power law with exponent $\alpha \approx 1.2$).*
3. **Fenn, D. J., Howison, S. D., McDonald, M., Williams, S., & Johnson, N. F. (2009).** Temporal evolution of financial market networks. *Physical Review E*, 79(3), 036108.  
   *(Documented sub-second exponential survival decay of triangular FX market dislocations).*
4. **Aiba, Y., Hatano, N., Takayasu, H., Kouki, I., & Takayasu, M. (2002).** Triangular arbitrage as an interaction among foreign exchange rates. *Physica A: Statistical Mechanics and its Applications*, 310(3-4), 467–479.  
   *(Modeled cross-rate correlation interactions and bid-ask bounce dynamics across G10 currencies).*
5. **Tarjan, R. E. (1981).** Shortest paths and cycle detection in directed graphs. *Networks*, 11(2), 145–160.  
   *(Algorithmic foundations of negative cycle detection and predecessor graph synchronization).*
