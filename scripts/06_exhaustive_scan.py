import time
from pathlib import Path
from itertools import combinations
import polars as pl

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KAGGLE_DIR = DATA_DIR / "kaggle"

# All pairs present in your Kaggle folders
TARGET_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURNZD", "EURCAD", "EURPLN",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD", "GBPCAD",
    "CHFJPY", "AUDJPY", "NZDJPY", "CADJPY",
    "AUDCHF", "AUDNZD", "AUDCAD",
    "USDMXN", "USDRUB", "USDTRY", "USDZAR"
]


def discover_month_folders(kaggle_dir: Path) -> list[Path]:
    """
    Finds the top-level monthly folders inside data/kaggle/
    (e.g., April2020, April2021, March2022) using recursive rglob.
    """
    search_root = kaggle_dir if kaggle_dir.exists() else DATA_DIR
    month_folders = []
    for child in sorted(search_root.iterdir(), key=lambda x: x.name):
        if child.is_dir():
            # Check recursively if this monthly folder contains any .csv inside its pair subfolders
            if any(child.rglob("*.csv")) or any(child.rglob("*.CSV")):
                month_folders.append(child)
    return month_folders


def map_csvs_in_folder(month_folder: Path) -> dict[str, Path]:
    """
    Recursively searches inside month_folder (e.g., April2020/AUDJPY-2020-04/*.csv)
    to map each pair to its CSV file for that specific month.
    """
    found = {}
    for p in month_folder.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".csv":
            clean = p.name.upper().replace("/", "").replace("-", "").replace("_", "")
            for pair in TARGET_PAIRS:
                if pair in clean:
                    found[pair] = p
                    break
    return found


def load_pair_lazy(pair: str, csv_path: Path) -> pl.LazyFrame:
    """Lazily parses a 4-column headerless Kaggle CSV: Pair, Timestamp, Bid, Ask."""
    lf = pl.scan_csv(csv_path, has_header=False, try_parse_dates=False)
    schema = lf.collect_schema().names()

    time_col = schema[1] if len(schema) > 1 else schema[0]
    bid_col = schema[2] if len(schema) > 2 else schema[1]
    ask_col = schema[3] if len(schema) > 3 else schema[2]

    return (
        lf.select([
            pl.col(time_col).str.slice(0, 23).str.to_datetime("%Y%m%d %H:%M:%S%.f", strict=False).alias("time"),
            pl.col(bid_col).cast(pl.Float64, strict=False).alias(f"{pair}_bid"),
            pl.col(ask_col).cast(pl.Float64, strict=False).alias(f"{pair}_ask"),
        ])
        .drop_nulls()
        .filter((pl.col(f"{pair}_bid") > 0) & (pl.col(f"{pair}_ask") >= pl.col(f"{pair}_bid")))
        .sort("time")
    )


def find_valid_triangles(active_pairs: list[str]) -> list[tuple[str, list]]:
    """Finds all closed 3-currency triangles (both directions) in the monthly folder."""
    pair_set = set(active_pairs)
    currencies = sorted(list({p[:3] for p in active_pairs} | {p[3:] for p in active_pairs}))
    triangles = []

    def get_pair_and_dir(u: str, v: str):
        if f"{u}{v}" in pair_set:
            return f"{u}{v}", "direct"   # u -> v sells Base (u) at BID
        if f"{v}{u}" in pair_set:
            return f"{v}{u}", "inverse"  # u -> v buys Base (v) at 1 / ASK
        return None, None

    for a, b, c in combinations(currencies, 3):
        p1, d1 = get_pair_and_dir(a, b)
        p2, d2 = get_pair_and_dir(b, c)
        p3, d3 = get_pair_and_dir(c, a)
        if p1 and p2 and p3:
            triangles.append((f"{a}->{b}->{c}->{a}", [(p1, d1), (p2, d2), (p3, d3)]))
            p1r, d1r = get_pair_and_dir(a, c)
            p2r, d2r = get_pair_and_dir(c, b)
            p3r, d3r = get_pair_and_dir(b, a)
            triangles.append((f"{a}->{c}->{b}->{a}", [(p1r, d1r), (p2r, d2r), (p3r, d3r)]))

    return triangles


def build_cycle_expr(cycle_name: str, legs: list, use_mid: bool = False) -> pl.Expr:
    """Vectorized Polars expression for cycle return in basis points (bps)."""
    prod_expr = pl.lit(1.0)
    for pair, direction in legs:
        if use_mid:
            mid = 0.5 * (pl.col(f"{pair}_bid") + pl.col(f"{pair}_ask"))
            leg_rate = mid if direction == "direct" else (1.0 / mid)
        else:
            leg_rate = pl.col(f"{pair}_bid") if direction == "direct" else (1.0 / pl.col(f"{pair}_ask"))
        prod_expr = prod_expr * leg_rate
    return ((prod_expr - 1.0) * 1e4).alias(cycle_name)


def scan_folder(folder: Path, grid_freq: str = "100ms") -> pl.DataFrame:
    """
    Synchronizes all pairs inside a monthly folder (e.g. April2020) and scans
    every active 100ms event across the entire month for arbitrage.
    """
    t0 = time.perf_counter()
    csv_map = map_csvs_in_folder(folder)
    if len(csv_map) < 3:
        print(f"[{folder.name:14s}] Only {len(csv_map)} pairs found, skipping.")
        return pl.DataFrame()

    triangles = find_valid_triangles(list(csv_map.keys()))
    if not triangles:
        print(f"[{folder.name:14s}] {len(csv_map)} pairs found, but 0 closed triangles.")
        return pl.DataFrame()

    dfs = []
    for pair, path in csv_map.items():
        df_pair = (
            load_pair_lazy(pair, path)
            .with_columns(pl.col("time").dt.truncate(grid_freq))
            .group_by("time", maintain_order=True)
            .agg([
                pl.col(f"{pair}_bid").last(),
                pl.col(f"{pair}_ask").last(),
            ])
            .collect()
        )
        if df_pair.height > 0:
            dfs.append(df_pair)

    if len(dfs) < 3:
        return pl.DataFrame()

    master = (
        pl.concat([d.select("time") for d in dfs])
        .unique()
        .sort("time")
    )

    for d in dfs:
        master = master.join_asof(d, on="time", strategy="backward", tolerance="10s")

    raw_exprs = [build_cycle_expr(name, legs, use_mid=False) for name, legs in triangles]
    cycle_names = [name for name, _ in triangles]

    evaluated = (
        master.lazy()
        .select(["time"] + raw_exprs)
        .with_columns(pl.max_horizontal(cycle_names).alias("best_raw_bps"))
        .filter(pl.col("best_raw_bps") > 0.0)
        .collect()
    )

    elapsed = time.perf_counter() - t0

    if evaluated.height > 0:
        best_rows = []
        for row in evaluated.to_dicts():
            best_bps = row["best_raw_bps"]
            winner = max(cycle_names, key=lambda c: row[c] if row[c] is not None else -999.0)
            best_rows.append({
                "folder": folder.name,
                "time": row["time"],
                "cycle": winner,
                "raw_bid_ask_bps": round(best_bps, 3),
                "institutional_net_bps": round(best_bps - 0.30, 3),
            })
        res_df = pl.DataFrame(best_rows)
        inst_count = res_df.filter(pl.col("institutional_net_bps") > 0).height
        print(
            f"[{folder.name:14s}] {len(csv_map):2d} pairs | {len(triangles):2d} cycles | "
            f"{master.height:>9,} active ticks ({elapsed:5.2f}s) -> "
            f"FOUND {res_df.height:>6,} Raw Signals ({inst_count:>5,} Inst > 0.3 bps) | "
            f"Max: +{res_df['raw_bid_ask_bps'].max():.2f} bps"
        )
        return res_df
    else:
        print(
            f"[{folder.name:14s}] {len(csv_map):2d} pairs | {len(triangles):2d} cycles | "
            f"{master.height:>9,} active ticks ({elapsed:5.2f}s) -> 0 signals"
        )
        return pl.DataFrame()


def main():
    folders = discover_month_folders(KAGGLE_DIR)
    print(f"Discovered {len(folders)} monthly folders in {KAGGLE_DIR}:")
    print([f.name for f in folders])
    print("\nRunning Vectorized Polars 100ms Event-Driven Arbitrage Sweep...\n")

    all_opportunities = []
    for folder in folders:
        try:
            df_res = scan_folder(folder, grid_freq="100ms")
            if df_res.height > 0:
                all_opportunities.append(df_res)
        except Exception as e:
            print(f"[{folder.name:14s}] Skipped due to error: {e}")

    print("\n================ EXHAUSTIVE MULTI-MONTH SUMMARY ================")
    if all_opportunities:
        top_df = pl.concat(all_opportunities).sort("raw_bid_ask_bps", descending=True)
        inst_df = top_df.filter(pl.col("institutional_net_bps") > 0)

        print(f"Total Executable Raw Bid/Ask Signals Found : {top_df.height:,}")
        print(f"Total Institutional Signals (>0.3 bps fee) : {inst_df.height:,}")
        print("\nTop 20 Largest Real Executable Arbitrage Dislocations:")
        print(top_df.head(20))

        out_path = DATA_DIR / "all_discovered_arbitrage_signals.parquet"
        top_df.write_parquet(out_path)
        print(f"\nSaved all {top_df.height:,} discovered signals to: {out_path}")
    else:
        print("0 executable signals found across all scanned folders.")
    print("================================================================\n")


if __name__ == "__main__":
    main()