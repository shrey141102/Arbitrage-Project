**Name:** Shreyansh Khaitan (CID: 06084579)
**Submission date:** 23 October 2026
**Project:** Can you still make money from triangular arbitrage in FX once you pay real costs and real delays?

---

## 1. Hypothesis Pitch

We covered arbitrage in class recently, and it stuck with me. As an international student in the UK, I check FX rates a lot. I send money home, I pay fees in pounds, and I am always comparing rates. So I wanted to know if someone can actually make money in forex from arbitrage, or if it only works on paper.

**The idea.** Take three currencies, like EUR, GBP and USD. If EUR/USD, GBP/USD and EUR/GBP don't line up, you can go EUR to GBP to USD and back to EUR and end up with more than you started with. That is triangular arbitrage. The same thing works with 4 or 5 currencies.

**What I think is going on.** I think most of these chances are fake. Each currency pair updates its price at different moments. So for a split second, one leg of the loop can be stale. If you look at mid prices, the loop looks profitable. But a real trader has to pay the spread on every leg, pay a fee, and can't trade instantly. By the time the order arrives, market makers have already fixed the stale price.

**My hypothesis (H0).** After paying the bid/ask spread on every leg, paying fees, and filling 100 ms after the signal, the average net profit per trade is not above zero out-of-sample.

**What would prove me wrong.** If the out-of-sample average profit per trade is positive, with a p-value under 5%, and it still holds when I add more delay, higher costs, remove the best trades, and test on other periods. Then I would have to say real arbitrage exists in this data.

---

## 2. Data Pipeline

**Source.** A public Kaggle dataset of FX tick data, originally from Dukascopy: [X: paste Kaggle link here]. It has one folder per month and one CSV per pair. Each row has the pair, a timestamp in milliseconds, the bid and the ask. It covers [X] months, from [X] to [X], with 20 pairs.

**Getting the data.** The notebook can download the data automatically with the Kaggle API (`DOWNLOAD_RAW = True`). You need your own Kaggle token for this. No keys are in the submission.

**The snapshot.** The full archive is about 12 GB, which is too big to submit. So the notebook picks 3 random full trading days per month, using a fixed seed (2026). The days are random, not hand picked. It keeps every raw tick for those days and saves them as parquet files in `data/snapshot/`. It also writes a `manifest.json` with the chosen days, row counts and a SHA-256 checksum for every file. The notebook checks every checksum before running. The snapshot has [X] trading days, [X] ticks and is [X] MB.

**Cleaning.** I used six rules, and every removed row is counted:
- null or missing values
- prices at or below zero
- crossed quotes (ask below bid)
- duplicate timestamps (keep the last one)
- spreads wider than 200 bps
- price jumps more than 100 bps away from the median of the previous 51 ticks

The jump rule only looks at earlier ticks, never later ones, because a live trader can't see the future. In total, cleaning removed [X]% of [X] ticks. The full counts per pair are in `cleaning_audit_by_pair.csv`.

**Missing days and gaps.** Days that are holidays or half empty are skipped when picking days, since a day must have at least 50% of the month's normal tick count. If a pair is missing on a day, its cycles are just not checked that day. Nothing is filled in. Splits and dividends don't apply to FX.

**Lining up the pairs.** Pairs tick at different times. So I made a 100 ms grid. At each grid time T, each pair uses its last tick from strictly before T. I also keep the age of every quote, so I know how stale each leg is.

---

## 3. Strategy and Backtest

**The graph.** Each currency is a node. Each pair gives two edges. Selling the base uses the bid. Buying the base uses 1/ask. So the spread is built into the graph. A loop is profitable if the product of its rates is above 1. With weights of -log(rate), that is a negative cycle.

**Finding loops.** There are [X] currencies and [X] possible loops of 3 to 5 legs. A fast scanner checks all of them at every grid time. I also wrote Bellman-Ford from scratch, which finds a negative cycle of any length and rebuilds it with parent pointers. Both are tested against each other (see below).

**The trading rule.** At each time T:
1. Look at all loops where every leg is at most A ms old.
2. Pick the loop with the best profit after fees.
3. Trade it only if that profit is above θ bps.
4. After a trade, wait 1 second before trading again.
5. Fill the trade 100 ms later, at the prices at that later time.

Step 5 is the key part. The signal only uses old information, and the result only uses prices from the fill time.

**Settings.** Fee 0.1 bps per leg. $1M per leg. $5M of capital. θ and A were picked from a grid of 25 options using training data only.

**Train and test.** The first 60% of months are for training. The rest were not touched until the settings were locked. I also did a walk-forward test, where I retune on all past months and then test on the next 3 months, again and again.

**Key results**

| Metric | In-sample (training) | Out-of-sample (test) |
|---|---|---|
| Trades | [X] | [X] |
| Mean net profit per trade (bps) | [X] | [X] |
| 95% CI | [X] | [X] |
| Hit rate | [X] | [X] |
| Cumulative return | [X] | [X] |
| Sharpe ratio | [X] | [X] |
| Max drawdown | [X] | [X] |
| p-value (H0: mean ≤ 0) | [X] | [X] |

Walk-forward, all test blocks pooled: [X] trades, mean [X] bps, Sharpe [X], cumulative return [X], max drawdown [X].

Selected settings: θ = [X] bps, A = [X] ms.

**What the results say.** [X: 2 to 3 sentences. For example: "On paper, at 0 ms, the trades make [X] bps each. Filled 100 ms later, that drops to [X] bps. So most of the edge is just stale quotes." Then state whether H0 is rejected.]

The funnel chart shows how the chances shrink at each step. [X]% of grid times look profitable at mid prices. Only [X]% are still profitable at real bid/ask prices. Only [X] trades survive all the filters.

**Testing the code.** 14 checks run inside the notebook and all pass. They include a hand-built triangle with exactly +2.5 bps, a fee formula check, 800 random graphs where Bellman-Ford is compared with brute force, and a check of the scanner against a slow pure Python version on real data.

---

## 4. Methodology Defense

**Why FX and why these pairs.** FX trades almost 24 hours a day and is very liquid, so if arbitrage exists anywhere, this is a fair place to look. I used all 20 pairs in the dataset. Pairs that don't close any loop are dropped automatically.

**Why bid/ask and not mid prices.** You can't trade at the mid. If I used mids, I would find lots of fake profit. Using bid for selling and ask for buying is the honest way to price each leg.

**Why 100 ms delay.** It is one grid step. That is the smallest delay this data can show, and it is still very generous. Real retail or even normal institutional traders are slower. I also tested 0 to 2000 ms.

**Why the right edge grid.** If the grid labelled each bin by its start, a decision at T could see ticks up to T + 100 ms. That is look-ahead. I built that wrong version on purpose as a check, and the test catches it.

**Why 0.1 bps fee per leg.** It is roughly what a big player pays. It is optimistic for a small trader. I also tested 0 to 0.5 bps.

**Why a 1 second cool-down.** The same mispricing often shows up in several loops at once and over several grid times. Without a cool-down I would count one chance many times.

**Why day-block bootstrap.** Trades on the same day are not independent. So I resample whole days, not single trades, to get honest confidence intervals.

**Why 3 to 5 legs.** Triangles are the classic case. Longer loops give more chances but pay more spread. I wanted to see if they ever help.

**What would break if an assumption is wrong.** If real delay is longer than 100 ms, results get worse, as the latency chart shows. If fees are higher, results get worse. If the order is bigger than what is shown at the best price, there is extra slippage. The Kaggle data has no volume, so the main backtest can't charge for this. That makes my results too optimistic, not too pessimistic.

---

## 5. Pitfall Checklist

### Look-ahead bias
I checked this in three ways. First, at every grid time, the newest tick used is always older than the decision time (smallest gap: [X] ms). Second, I rebuild the data using only ticks up to a random cut-off and check that nothing before the cut-off changes. It passed at all 40 cut-offs. Third, I built a version with a known look-ahead bug, and the test caught it. Fill prices are always taken at the later fill time, never before.

### Survivorship bias
Currencies don't get delisted like stocks, so this is a smaller problem here. But pairs can drop out. The coverage chart shows which pairs are missing on which days. [X: e.g. "USDRUB disappears after early 2022."] I didn't fill in missing data. If a pair is missing, its loops are just not traded. Also, the dataset only has 20 major and some emerging pairs. Less liquid pairs, which might have bigger mispricings, are not in it.

### Overfitting / data-snooping
I tuned only 2 settings (θ and A) on a grid of 25 options, using training months only. The test months were not touched until the settings were locked. In total, [X] setups were tried, including walk-forward retuning. I also checked how well training results predict test results across all 25 options. The rank correlation is [X], and the chosen setup ranks [X] out of [X] on the test set. The hypothesis and the test plan were fixed before I saw any results.

### Transaction costs and slippage
The spread is paid on every leg, since the graph uses bid and ask. The fee is 0.1 bps per leg. The 100 ms delay acts as slippage, because prices move before the fill. Results with other costs: [X: e.g. "the mean profit turns negative at a fee of [X] bps per leg" or "at a delay of [X] ms"]. Order book depth is not in the Kaggle data, so I can't model the cost of large orders. That part is missing.

### Out-of-sample validation
Test period: [X] to [X] ([X] days). In-sample mean was [X] bps per trade, out-of-sample was [X]. Walk-forward had [X] folds, and [X] of them had a positive mean. [X: one sentence on whether the test period looked like training.]

---

## 6. Efficiency Touchpoint

The most expensive step is the cycle scan. It took [X] seconds, [X]% of total run time. Its cost is O(S × C × k). S is the number of grid times ([X]), C is the number of loops ([X]) and k is the average legs per loop ([X]). Each delay level also needs a sorted as-of join, which costs O(n log n).

Bellman-Ford costs O(V × E), with V = [X] currencies and E = [X] edges, no matter how long the loop is. In my benchmark, the vectorised scanner took [X] µs per grid time, a plain Python loop took [X] µs, and Bellman-Ford took [X] µs.

Listing every loop is fine here, because the graph is small. But the number of loops grows very fast with loop length and the number of currencies. With many more currencies, Bellman-Ford would be the better choice. I also only process 3 days per month, and I process data one day at a time in chunks, so memory stays low.

---

## 7. Risk Touchpoint

**Size.** Each trade puts $1M on each leg, so an average trade moves about [X] times the capital. Daily turnover is about [X] times capital. That is a lot of trading for a very small edge per trade.

**Concentration.** [X] loops were traded. The top loop ([X]) makes up [X]% of trades and the top 5 make up [X]%. So it is less diversified than it looks. Most trades share a few currencies, like [X].

**Bad scenarios.** If spreads double, like in a liquidity crunch, the mean becomes [X] bps. If they triple, it becomes [X] bps. With a 500 ms delay it is [X] bps. The worst single trade lost [X] bps and the worst day lost $[X].

The biggest real risk is leg risk. You might fill two legs and miss the third, and then you are stuck holding a currency you didn't want. My backtest assumes all legs fill together, so it does not show this risk.

---

## 8. Reflection

**What I learned.** Before this, I thought arbitrage was just finding loops where the numbers multiply to more than 1. Now I see that the hard part is not finding them. It is whether you can actually trade them. Most of the profit came from stale quotes, and it was gone within one grid step. I also learned how easy it is to cheat by accident. Just labelling a time bin the wrong way gives the strategy a peek into the future.

**What I got wrong.** [X: pick what fits the results. For example: "I expected longer loops (4 or 5 legs) to help, but they paid more spread and did worse." / "I thought fees would be the main killer, but delay mattered much more." / "I thought the data would be clean, but [X]."]

**Next steps.**
- Use data with order book depth, like Dukascopy volumes, to price large orders properly. The notebook already has an optional stage for this.
- Use real timestamps from an exchange, not a data vendor, to measure true latency.
- Model leg risk, where one leg fails to fill.
- Test less liquid pairs, where mispricings may last longer.
- To get closer to real trading, the whole thing would need to run on a live feed, in a faster language, close to the exchange servers. That is what real HFT firms do, and it is probably why normal traders can't win this game.

---

## References

- Kaggle FX tick dataset: [X: link]
- Dukascopy historical data feed (original source of the ticks).
