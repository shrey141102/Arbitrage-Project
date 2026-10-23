# Can you still make money from triangular arbitrage in FX after real costs and real delays?

This is my capstone for the Data Structures and Algorithms module at Imperial College Business School.
It tests whether triangular and longer currency loops in FX still give profit after paying the bid/ask spread
on every leg, paying fees, and filling the trade 100 ms after the signal.
The data is real tick data for 15 currency pairs (April 2020 to August 2022).

## Result

Out-of-sample (Sep 2021 to Aug 2022, 36 days), the strategy loses money: mean net return per trade is -0.634 bps
over 173 trades (95% CI [-2.848, -0.020]), cumulative return -0.219%, Sharpe -5.16, max drawdown 0.220%.
On paper (fill at 0 ms) the same trades make +1.59 bps each, but with a 100 ms fill they make -0.63 bps.
The one-sided p-value is 0.979, so H0 (mean net return per trade is not above zero) is not rejected.

## Dataset

Kaggle FX tick data (originally from Dukascopy): https://www.kaggle.com/datasets/zahinawosaf/forex-tick-data

## How to run

1. `pip install -r requirements.txt`
2. Put the Kaggle month folders in `data/kaggle/`, and set `REBUILD_SNAPSHOT = True` in the notebook on the first run.
   (Or put an existing snapshot in `data/snapshot/` and leave it as it is.)
3. Run `notebook.ipynb` from top to bottom.

The snapshot is about 1 GB, so it is not stored in this repo.

## Files

- `notebook.ipynb`: the full pipeline (data acquisition, cleaning, strategy, backtest, tests). All final numbers come from here.
- `report.pdf`: the written report (methodology, results, pitfall checklist, reflection).
- `data/README.md`: what is in the data folder.
- `data/derived/`: the results the notebook writes (`results.json`, `key_results.md`, tables and figures).
- `legacy/`: earlier exploratory scripts, tests and notes. They are NOT the final results. Do not use their numbers.

## AI tools

I used AI tools in this project: Gemini 4 Argon and Claude Opus 5.5. The idea, the choices and the final checks are mine.
