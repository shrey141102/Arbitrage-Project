import os
import time
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager

import requests
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
GRID_FILE = BASE_DIR / "data" / "aligned_grid_20261014_13.parquet"

# API Credentials from Environment
OANDA_API_KEY = os.getenv("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")

# Shared In-Memory State updated every 5 seconds
LIVE_STATE = {
    "oanda": {"status": "initializing", "polls": 0},
    "yfinance": {"status": "initializing", "polls": 0},
    "finnhub": {"status": "initializing", "polls": 0},
    "twelvedata": {"status": "initializing", "polls": 0},
    "frankfurter": {"status": "initializing", "polls": 0},
    "dukascopy_replay": {"status": "initializing", "polls": 0},
}

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (FX-Capstone-POC)"})


def evaluate_triangle(quotes: dict) -> dict:
    """
    Given standardized {'EURUSD': {'bid': x, 'ask': y}, ...},
    computes both Mid-Price Illusion yield and Real Executable Bid-Ask yield (in bps).
    """
    required = ["EURUSD", "GBPUSD", "EURGBP"]
    if not all(k in quotes for k in required):
        return {"error": "Incomplete triangle quotes"}

    # Calculate Mid Prices
    mids = {p: 0.5 * (quotes[p]["bid"] + quotes[p]["ask"]) for p in required}

    # Cycle 1: EUR -> GBP -> USD -> EUR
    mid_c1 = mids["EURGBP"] * mids["GBPUSD"] * (1.0 / mids["EURUSD"])
    exec_c1 = quotes["EURGBP"]["bid"] * quotes["GBPUSD"]["bid"] * (1.0 / quotes["EURUSD"]["ask"])

    # Cycle 2: EUR -> USD -> GBP -> EUR
    mid_c2 = mids["EURUSD"] * (1.0 / mids["GBPUSD"]) * (1.0 / mids["EURGBP"])
    exec_c2 = quotes["EURUSD"]["bid"] * (1.0 / quotes["GBPUSD"]["ask"]) * (1.0 / quotes["EURGBP"]["ask"])

    best_mid_bps = (max(mid_c1, mid_c2) - 1.0) * 1e4
    best_exec_bps = (max(exec_c1, exec_c2) - 1.0) * 1e4
    direction = "EUR->GBP->USD->EUR" if exec_c1 >= exec_c2 else "EUR->USD->GBP->EUR"

    return {
        "best_direction": direction,
        "mid_price_illusion_bps": round(best_mid_bps, 4),
        "executable_bid_ask_bps": round(best_exec_bps, 4),
        "spread_bps": {
            p: round(((quotes[p]["ask"] - quotes[p]["bid"]) / mids[p]) * 1e4, 3)
            for p in required
        }
    }


# =====================================================================
# 1. OANDA REST-v20 FETCHER (Poll Interval: 5s | Limit: 120 req/sec)
# =====================================================================
def poll_oanda() -> dict:
    if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
        return {"status": "MISSING_API_KEY", "note": "Set OANDA_API_KEY and OANDA_ACCOUNT_ID"}
    
    t0 = time.perf_counter()
    url = f"https://api-fxpractice.oanda.com/v3/accounts/{OANDA_ACCOUNT_ID}/pricing"
    headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
    r = session.get(url, headers=headers, params={"instruments": "EUR_USD,GBP_USD,EUR_GBP"}, timeout=8)
    r.raise_for_status()
    data = r.json()

    quotes = {}
    for p in data.get("prices", []):
        pair = p["instrument"].replace("_", "")
        quotes[pair] = {
            "bid": float(p["bids"][0]["price"]),
            "ask": float(p["asks"][0]["price"]),
            "liquidity_m": float(p["bids"][0].get("liquidity", 0)) / 1e6
        }
    return {
        "status": "OK",
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "quotes": quotes,
        "arbitrage": evaluate_triangle(quotes)
    }


# =====================================================================
# 2. YFINANCE FETCHER (Poll Interval: 5s | No API Key Needed)
# =====================================================================
def poll_yfinance() -> dict:
    t0 = time.perf_counter()
    tickers = yf.Tickers("EURUSD=X GBPUSD=X EURGBP=X")
    quotes = {}
    for sym, clean in [("EURUSD=X", "EURUSD"), ("GBPUSD=X", "GBPUSD"), ("EURGBP=X", "EURGBP")]:
        info = tickers.tickers[sym].fast_info
        # Fallback to last_price if weekend bid/ask is 0
        last = float(info.last_price)
        bid = float(getattr(info, "bid", last) or last)
        ask = float(getattr(info, "ask", last) or last)
        quotes[clean] = {"bid": bid, "ask": ask}

    return {
        "status": "OK",
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "quotes": quotes,
        "arbitrage": evaluate_triangle(quotes)
    }


# =====================================================================
# 3. FINNHUB FETCHER (Poll Interval: 5s | Limit: 60 calls/min Free)
# =====================================================================
def poll_finnhub() -> dict:
    if not FINNHUB_API_KEY:
        return {"status": "MISSING_API_KEY", "note": "Set FINNHUB_API_KEY (free at finnhub.io)"}
    
    t0 = time.perf_counter()
    # Finnhub free forex endpoint or quote endpoint for OANDA feeds
    url = "https://finnhub.io/api/v1/forex/rates"
    r = session.get(url, params={"base": "EUR", "token": FINNHUB_API_KEY}, timeout=8)
    r.raise_for_status()
    rates = r.json().get("quote", {})
    
    eur_usd = float(rates.get("USD", 1.11))
    eur_gbp = float(rates.get("GBP", 0.84))
    gbp_usd = eur_usd / eur_gbp
    
    quotes = {
        "EURUSD": {"bid": eur_usd, "ask": eur_usd},
        "EURGBP": {"bid": eur_gbp, "ask": eur_gbp},
        "GBPUSD": {"bid": gbp_usd, "ask": gbp_usd},
    }
    return {
        "status": "OK",
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "quotes": quotes,
        "arbitrage": evaluate_triangle(quotes)
    }


# =====================================================================
# 4. TWELVE DATA FETCHER (Poll Interval: 10s | Limit: 8 calls/min Free)
# =====================================================================
def poll_twelvedata() -> dict:
    if not TWELVEDATA_API_KEY:
        return {"status": "MISSING_API_KEY", "note": "Set TWELVEDATA_API_KEY (free at twelvedata.com)"}
    
    t0 = time.perf_counter()
    url = "https://api.twelvedata.com/price"
    r = session.get(url, params={"symbol": "EUR/USD,GBP/USD,EUR/GBP", "apikey": TWELVEDATA_API_KEY}, timeout=8)
    r.raise_for_status()
    data = r.json()

    quotes = {}
    for sym, clean in [("EUR/USD", "EURUSD"), ("GBP/USD", "GBPUSD"), ("EUR/GBP", "EURGBP")]:
        if sym in data and "price" in data[sym]:
            p = float(data[sym]["price"])
            quotes[clean] = {"bid": p, "ask": p}

    return {
        "status": "OK" if len(quotes) == 3 else "RATE_LIMITED_OR_ERROR",
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "quotes": quotes,
        "arbitrage": evaluate_triangle(quotes) if len(quotes) == 3 else data
    }


# =====================================================================
# 5. FRANKFURTER FETCHER (Poll Interval: 5s | No API Key Needed)
# =====================================================================
def poll_frankfurter() -> dict:
    t0 = time.perf_counter()
    r1 = session.get("https://api.frankfurter.dev/v1/latest?base=EUR&symbols=USD,GBP", timeout=8).json()
    r2 = session.get("https://api.frankfurter.dev/v1/latest?base=GBP&symbols=USD", timeout=8).json()

    eur_usd = float(r1["rates"]["USD"])
    eur_gbp = float(r1["rates"]["GBP"])
    gbp_usd = float(r2["rates"]["USD"])

    quotes = {
        "EURUSD": {"bid": eur_usd, "ask": eur_usd},
        "EURGBP": {"bid": eur_gbp, "ask": eur_gbp},
        "GBPUSD": {"bid": gbp_usd, "ask": gbp_usd},
    }
    return {
        "status": "OK (ECB Daily Reference)",
        "date": r1.get("date"),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "quotes": quotes,
        "arbitrage": evaluate_triangle(quotes)
    }


# =====================================================================
# 6. BONUS: DUKASCOPY REAL TICK STREAMER (Streams saved parquet grid)
# =====================================================================
DUKA_DF = pd.read_parquet(GRID_FILE) if GRID_FILE.exists() else pd.DataFrame()
DUKA_IDX = 0

def poll_dukascopy_replay() -> dict:
    global DUKA_IDX
    if DUKA_DF.empty:
        return {"status": "NO_PARQUET_FOUND"}
    
    row = DUKA_DF.iloc[DUKA_IDX % len(DUKA_DF)]
    DUKA_IDX += 1
    
    quotes = {
        p: {"bid": float(row[f"{p}_bid"]), "ask": float(row[f"{p}_ask"])}
        for p in ["EURUSD", "GBPUSD", "EURGBP"]
    }
    return {
        "status": "STREAMING_REPLAY",
        "replay_timestamp": str(row["time"]),
        "quotes": quotes,
        "arbitrage": evaluate_triangle(quotes)
    }


# =====================================================================
# BACKGROUND ASYNC POLLING LOOP (Pings every 5 seconds)
# =====================================================================
async def background_poller():
    providers = [
        ("oanda", poll_oanda, 5),
        ("yfinance", poll_yfinance, 5),
        ("finnhub", poll_finnhub, 5),
        ("twelvedata", poll_twelvedata, 10),  # 10s to respect 8 calls/min free tier
        ("frankfurter", poll_frankfurter, 5),
        ("dukascopy_replay", poll_dukascopy_replay, 2),
    ]
    
    last_run = {name: 0.0 for name, _, _ in providers}
    
    while True:
        now = time.time()
        for name, func, interval in providers:
            if now - last_run[name] >= interval:
                last_run[name] = now
                polls = LIVE_STATE[name].get("polls", 0) + 1
                try:
                    res = await asyncio.to_thread(func)
                    res["polls"] = polls
                    res["updated_at_utc"] = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
                    LIVE_STATE[name] = res
                    
                    # Terminal live heartbeat log
                    arb = res.get("arbitrage", {})
                    if "executable_bid_ask_bps" in arb:
                        print(
                            f"[{res['updated_at_utc']}] {name:16s} | Poll #{polls:<3d} | "
                            f"Mid Illusion: {arb['mid_price_illusion_bps']:+7.4f} bps | "
                            f"Exec Bid/Ask: {arb['executable_bid_ask_bps']:+7.4f} bps"
                        )
                except Exception as e:
                    LIVE_STATE[name] = {"status": "ERROR", "error": str(e), "polls": polls}
        await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\nStarting Live 5-Provider FX Poller (Ctrl+C to stop)...")
    task = asyncio.create_task(background_poller())
    yield
    task.cancel()


app = FastAPI(title="FX Triangular Arbitrage Live POC", lifespan=lifespan)

@app.get("/")
def get_all_providers():
    """Returns real-time state across all 5 providers + Dukascopy replay."""
    return LIVE_STATE

@app.get("/live/{provider}")
def get_single_provider(provider: str):
    """Fetch a single provider: oanda, yfinance, finnhub, twelvedata, frankfurter, dukascopy_replay."""
    return LIVE_STATE.get(provider, {"error": "Unknown provider"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")