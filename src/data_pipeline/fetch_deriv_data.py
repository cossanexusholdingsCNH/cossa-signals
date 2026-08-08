"""
Cossa Signals — Deriv Synthetic Indices Data Fetcher

Pulls historical candle data for Deriv's synthetic indices (Volatility,
Boom, Crash) via Deriv's public WebSocket API. Historical market data
does NOT require an authenticated API token — only live trading/account
actions do. You do need a free app_id, registered at:
https://api.deriv.com/dashboard/  (takes ~2 minutes, no cost)

IMPORTANT — this script has not been tested against a live connection
in the environment that built it (network access there is restricted
to package registries, not deriv.com). Run this locally / on your own
machine or server before trusting its output. If Deriv has changed
their API shape since this was written, the error message from the
server response will tell you what changed — check
https://developers.deriv.com/docs/data/ticks-history/ against it.

Usage:
    python src/data_pipeline/fetch_deriv_data.py
    python src/data_pipeline/fetch_deriv_data.py --symbols R_75 BOOM1000 --count 5000
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

import pandas as pd
import websockets
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Load config/.env if present (real API keys, never committed)
ENV_PATH = Path(__file__).resolve().parents[2] / "config" / ".env"
load_dotenv(ENV_PATH)

DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")  # 1089 is Deriv's public demo app_id
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Symbol codes as registered on Deriv's API (confirmed against Deriv's
# published symbol list — see developers.deriv.com)
DERIV_SYMBOLS = {
    "VOLATILITY_75": "R_75",
    "BOOM_1000": "BOOM1000",
    "BOOM_500": "BOOM500",
    "CRASH_1000": "CRASH1000",
    "CRASH_500": "CRASH500",
}


async def fetch_candles(
    symbol: str,
    count: int = 5000,
    granularity: int = 60,
) -> pd.DataFrame:
    """
    Fetch historical candles for one Deriv synthetic index.

    Args:
        symbol: Deriv API symbol code, e.g. "R_75", "BOOM1000".
        count: Number of candles to request (Deriv caps this per call —
               if you need more history than one call returns, this
               function would need to be extended to paginate using
               successive 'end' timestamps; that's a next-step
               enhancement, not built into this first version).
        granularity: Candle size in seconds. 60 = 1-minute candles.
                     Other common values: 300 (5min), 900 (15min),
                     3600 (1hr), 86400 (daily).

    Returns:
        DataFrame with columns: Open, High, Low, Close, indexed by
        datetime. Empty DataFrame on failure.
    """
    request = {
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": count,
        "end": "latest",
        "start": 1,
        "style": "candles",
        "granularity": granularity,
    }

    logger.info(f"Requesting {count} candles for {symbol} (granularity={granularity}s)")

    try:
        async with websockets.connect(DERIV_WS_URL) as ws:
            await ws.send(json.dumps(request))
            response_raw = await ws.recv()
            response = json.loads(response_raw)
    except Exception as exc:
        logger.error(f"Connection/request failed for {symbol}: {exc}")
        return pd.DataFrame()

    if "error" in response:
        logger.error(f"Deriv API error for {symbol}: {response['error'].get('message')}")
        return pd.DataFrame()

    candles = response.get("candles")
    if not candles:
        logger.warning(f"No candle data returned for {symbol}. Full response: {response}")
        return pd.DataFrame()

    df = pd.DataFrame(candles)
    df["Date"] = pd.to_datetime(df["epoch"], unit="s")
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df = df.set_index("Date")[["Open", "High", "Low", "Close"]].astype(float)

    logger.info(f"Fetched {len(df)} candles for {symbol} ({df.index.min()} to {df.index.max()})")
    return df


def save_symbol(df: pd.DataFrame, name: str) -> Path:
    """Save a symbol's DataFrame to data/raw/{name}.csv."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / f"{name}.csv"
    df.to_csv(out_path)
    logger.info(f"Saved {name} data to {out_path}")
    return out_path


async def fetch_all(symbols: list[str], count: int = 5000, granularity: int = 60) -> None:
    """Fetch and save data for a list of Deriv symbol codes."""
    results = {}
    for symbol in symbols:
        df = await fetch_candles(symbol, count=count, granularity=granularity)
        if df.empty:
            logger.error(f"Skipping save for {symbol} — no data fetched.")
            results[symbol] = "FAILED"
            continue
        save_symbol(df, symbol)
        results[symbol] = "OK"
        # Be a polite API citizen — small delay between requests
        await asyncio.sleep(1)

    logger.info("=== Fetch summary ===")
    for symbol, status in results.items():
        logger.info(f"  {symbol}: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Deriv synthetic index data for Cossa Signals.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DERIV_SYMBOLS.values()),
        help=f"Deriv symbol codes to fetch (default: {list(DERIV_SYMBOLS.values())})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5000,
        help="Number of candles per symbol (default: 5000, subject to Deriv's per-call cap)",
    )
    parser.add_argument(
        "--granularity",
        type=int,
        default=60,
        help="Candle size in seconds (default: 60 = 1-minute candles)",
    )
    args = parser.parse_args()

    asyncio.run(fetch_all(args.symbols, count=args.count, granularity=args.granularity))
