# Key results (generated 2026-10-23T00:58:35+00:00 UTC)

Configuration chosen on 2020-04..2021-08: θ = 1.0 bps, A = 2000 ms, latency 100 ms, fee 0.1 bps per leg. Test period 2021-09..2022-08.

| Metric | in-sample (training) | out-of-sample (test) |
|---|---|---|
| Sampled days | 51 | 36 |
| Trades | 129 | 173 |
| Trades per day | 2.5 | 4.8 |
| Mean net return per trade (bps) | -0.585 | -0.634 |
|   95 % CI (day-block bootstrap) | [-1.096, -0.232] | [-2.848, -0.020] |
| Hit rate | 38.0 % | 54.3 % |
| Cumulative return (sampled days, $5M capital) | -0.151 % | -0.219 % |
| Sharpe ratio (annualised, √252) | -6.30 | -5.16 |
|   95 % CI (day-block bootstrap) | [-8.86, -3.67] | [-11.08, -0.26] |
| Maximum drawdown | 0.151 % | 0.220 % |
| Worst day (USD) | -1,527 | -3,680 |
| One-sided p-value, H0: mean net ≤ 0 | 1.000 | 0.979 |
| Trades with no quote at fill time (excluded) | 0 | 0 |

**H0 is not rejected** (p = 0.979; mean net -0.634 bps per trade out-of-sample).

## Walk-forward (pooled out-of-sample)

| trades | mean net bps | 95 % CI | p (H0: ≤ 0) | Sharpe | cumulative return | max drawdown |
|---|---|---|---|---|---|---|
| 2,819 | +0.345 | [-1.995, +0.413] | 0.363 | +1.85 | +1.943 % | 0.284 % |
