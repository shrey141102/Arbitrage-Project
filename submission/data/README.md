# data/ folder

- `snapshot/` is the data used in this project. It has every raw tick for 3 random trading days per month (87 days, April 2020 to August 2022), saved as parquet files.
- `snapshot/manifest.json` lists the chosen days, row counts and a SHA-256 checksum for every file. The notebook checks these before it runs.
- `derived/` is everything the notebook writes: results.json, key_results.md, tables and figures.

Source: Kaggle FX tick dataset (originally from Dukascopy), https://www.kaggle.com/datasets/zahinawosaf/forex-tick-data

The snapshot was built by Stage 2 of notebook.ipynb from the full archive. To rebuild it, put the monthly Kaggle folders in data/kaggle/ and set REBUILD_SNAPSHOT = True. To run the notebook as submitted, only snapshot/ is needed.
