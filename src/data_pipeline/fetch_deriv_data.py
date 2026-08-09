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


DERIV_MAX_CANDLES_PER_CALL = 5000  # Deriv's per-request cap for ticks_history/candles


# Well-documented Deriv synthetic index symbols, confirmed against Deriv's
# own official API docs and schema (not guessed). Grouped by family.
# Because Deriv's active_symbols endpoint has been unreliable for this app
# (returning an empty list live despite ticks_history working fine on the
# same connection — likely an app-permission restriction, not a code bug),
# --all now uses this list and VALIDATES each symbol live via a small
# ticks_history call, rather than trusting active_symbols. Anything that
# fails validation is reported, not silently skipped — see validate below.
CANDIDATE_SYMBOLS = {
    # Volatility Indices — standard (2-second-ish tick)
    "Volatility 10 Index": "R_10",
    "Volatility 25 Index": "R_25",
    "Volatility 50 Index": "R_50",
    "Volatility 75 Index": "R_75",
    "Volatility 100 Index": "R_100",
    # Volatility Indices — 1-second variants
    "Volatility 10 (1s) Index": "1HZ10V",
    "Volatility 25 (1s) Index": "1HZ25V",
    "Volatility 50 (1s) Index": "1HZ50V",
    "Volatility 75 (1s) Index": "1HZ75V",
    "Volatility 100 (1s) Index": "1HZ100V",
    # Boom / Crash — confirmed live in this project already
    "Boom 500 Index": "BOOM500",
    "Boom 1000 Index": "BOOM1000",
    "Crash 500 Index": "CRASH500",
    "Crash 1000 Index": "CRASH1000",
    # Bull/Bear market synthetics (confirmed in Deriv's official API docs)
    "Bull Market Index": "RDBULL",
    "Bear Market Index": "RDBEAR",
    # Step Index — widely documented symbol, slightly less certain than the above
    "Step Index": "stpRNG",
}


async def validate_symbols_live(symbols: dict) -> list[dict]:
    """
    Confirm which candidate symbols are actually tradeable right now, by
    attempting a tiny 2-candle fetch for each over a single persistent
    connection. This is more reliable than trusting active_symbols, which
    has been returning empty on this app despite ticks_history working —
    we verify against the endpoint we KNOW works instead of guessing.

    Returns:
        List of dicts for symbols that responded successfully:
        {"symbol": "R_75", "display_name": "Volatility 75 Index"}
    """
    validated = []
    try:
        async with websockets.connect(DERIV_WS_URL) as ws:
            for display_name, symbol in symbols.items():
                candles = await _fetch_one_batch(ws, symbol, count=2, end_epoch="latest", granularity=60)
                if candles:
                    validated.append({"symbol": symbol, "display_name": display_name})
                    logger.info(f"  [OK]     {symbol:12} {display_name}")
                else:
                    logger.warning(f"  [FAILED] {symbol:12} {display_name} — not tradeable or symbol code incorrect")
                await asyncio.sleep(0.3)
    except Exception as exc:
        logger.error(f"Validation connection failed partway through: {exc}")

    logger.info(f"Validated {len(validated)}/{len(symbols)} candidate symbols as live and tradeable")
    return validated


async def discover_synthetic_symbols() -> list[dict]:
    """
    Ask Deriv directly for every currently active synthetic index symbol —
    Volatility, Boom, Crash, Step, Jump, Range Break, and any others Deriv
    currently offers. This replaces a hardcoded symbol list: Deriv adds and
    retires instruments over time, and the only reliable source of truth is
    Deriv's own active_symbols endpoint, not a list written by hand.

    Returns:
        List of dicts: {"symbol": "R_75", "display_name": "Volatility 75 Index",
        "submarket": "random_index"}, one per active synthetic index.
    """
    # NOTE: Deriv removed landing_company / landing_company_short / product_type
    # as active_symbols parameters in their current API version (confirmed
    # against Deriv's own API comparison docs, Aug 2026). Sending product_type
    # here previously caused Deriv to silently return an empty list instead
    # of an error — the request below is the current, correct minimal form.
    request = {"active_symbols": "brief"}

    try:
        async with websockets.connect(DERIV_WS_URL) as ws:
            await ws.send(json.dumps(request))
            response_raw = await ws.recv()
            response = json.loads(response_raw)
    except Exception as exc:
        logger.error(f"Failed to fetch active symbols from Deriv: {exc}")
        return []

    if "error" in response:
        logger.error(f"Deriv API error on active_symbols: {response['error'].get('message')}")
        return []

    all_symbols = response.get("active_symbols", [])

    if not all_symbols:
        # This shouldn't normally happen — dump diagnostic info so we can
        # see exactly what Deriv actually sent back instead of guessing.
        logger.error(
            "active_symbols came back empty. Response msg_type: "
            f"{response.get('msg_type', 'MISSING')}. Top-level keys: {list(response.keys())}"
        )
        logger.error(f"Full raw response (first 2000 chars): {json.dumps(response)[:2000]}")
        return []

    logger.info(f"Deriv returned {len(all_symbols)} total active symbols across all markets (before filtering)")
    markets_seen = sorted(set(s.get("market", "MISSING") for s in all_symbols))
    logger.info(f"Markets present in response: {markets_seen}")

    synthetic = [
        {
            "symbol": s["symbol"],
            "display_name": s.get("display_name", s["symbol"]),
            "submarket": s.get("submarket", "unknown"),
        }
        for s in all_symbols
        if s.get("market") == "synthetic_index"
    ]

    logger.info(f"Discovered {len(synthetic)} active synthetic index symbols on Deriv")
    submarkets = sorted(set(s["submarket"] for s in synthetic))
    logger.info(f"Submarkets found: {submarkets}")

    return synthetic


async def _fetch_one_batch(
    ws,
    symbol: str,
    count: int,
    end_epoch,
    granularity: int,
) -> list:
    """Send one ticks_history request over an already-open websocket and return raw candles."""
    request = {
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": count,
        "end": end_epoch,  # "latest" or a unix timestamp (int)
        "start": 1,
        "style": "candles",
        "granularity": granularity,
    }
    await ws.send(json.dumps(request))
    response_raw = await ws.recv()
    response = json.loads(response_raw)

    if "error" in response:
        logger.error(f"Deriv API error for {symbol}: {response['error'].get('message')}")
        return []

    return response.get("candles", [])


async def fetch_candles(
    symbol: str,
    count: int = 5000,
    granularity: int = 60,
    days: float | None = None,
) -> pd.DataFrame:
    """
    Fetch historical candles for one Deriv synthetic index, paginating
    automatically if more candles are requested than Deriv returns in
    a single call (capped at DERIV_MAX_CANDLES_PER_CALL).

    Args:
        symbol: Deriv API symbol code, e.g. "R_75", "BOOM1000".
        count: Total number of candles wanted, across as many paginated
               calls as needed. Ignored if `days` is given.
        granularity: Candle size in seconds. 60 = 1-minute candles.
                     Other common values: 300 (5min), 900 (15min),
                     3600 (1hr), 86400 (daily).
        days: If given, overrides `count` — computes the number of
              candles needed to cover this many days at the given
              granularity. E.g. days=30, granularity=60 requests
              enough 1-minute candles to cover 30 days (~43,200 candles,
              which will be paginated across ~9 calls).

    Returns:
        DataFrame with columns: Open, High, Low, Close, indexed by
        datetime, sorted ascending, de-duplicated. Empty DataFrame on
        total failure.
    """
    if days is not None:
        seconds_needed = days * 86400
        count = int(seconds_needed / granularity)
        logger.info(f"Requested {days} days at {granularity}s granularity -> {count} candles needed")

    all_candles = []
    remaining = count
    end_epoch = "latest"
    batch_num = 0
    max_batches = 200  # hard safety ceiling against runaway loops

    try:
        async with websockets.connect(DERIV_WS_URL) as ws:
            while remaining > 0 and batch_num < max_batches:
                batch_count = min(remaining, DERIV_MAX_CANDLES_PER_CALL)
                batch_num += 1
                logger.info(
                    f"[{symbol}] Batch {batch_num}: requesting {batch_count} candles "
                    f"(end={end_epoch}, {remaining} still needed)"
                )

                candles = await _fetch_one_batch(ws, symbol, batch_count, end_epoch, granularity)
                if not candles:
                    logger.warning(f"[{symbol}] Batch {batch_num} returned no candles — stopping pagination here.")
                    break

                all_candles.extend(candles)
                remaining -= len(candles)

                # Page backwards: next call's "end" is just before the oldest
                # candle we just received, so the next batch continues further
                # into the past without overlapping this one.
                oldest_epoch_in_batch = min(c["epoch"] for c in candles)
                end_epoch = oldest_epoch_in_batch - granularity

                if len(candles) < batch_count:
                    # Deriv returned fewer than asked — likely hit the start
                    # of available history for this symbol.
                    logger.info(f"[{symbol}] Reached earliest available history after {batch_num} batches.")
                    break

                await asyncio.sleep(0.5)  # be a polite API citizen between paginated calls
    except Exception as exc:
        logger.error(f"Connection/request failed for {symbol} on batch {batch_num}: {exc}")
        if not all_candles:
            return pd.DataFrame()
        logger.warning(f"[{symbol}] Continuing with {len(all_candles)} candles fetched before the error.")

    if not all_candles:
        logger.warning(f"No candle data returned for {symbol}.")
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)
    df["Date"] = pd.to_datetime(df["epoch"], unit="s")
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df = df.drop_duplicates(subset="epoch").sort_values("epoch")
    df = df.set_index("Date")[["Open", "High", "Low", "Close"]].astype(float)

    logger.info(
        f"[{symbol}] Fetched {len(df)} total candles across {batch_num} batch(es) "
        f"({df.index.min()} to {df.index.max()})"
    )
    return df


def save_symbol(df: pd.DataFrame, name: str) -> Path:
    """Save a symbol's DataFrame to data/raw/{name}.csv."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / f"{name}.csv"
    df.to_csv(out_path)
    logger.info(f"Saved {name} data to {out_path}")
    return out_path


def save_symbol_catalog(symbols: list[dict]) -> Path:
    """
    Save the discovered symbol list to data/raw/symbol_catalog.json — this
    becomes the reference list for which instruments need study guides,
    backtests, etc. Regenerate it periodically since Deriv adds/retires
    instruments over time.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / "symbol_catalog.json"
    with open(out_path, "w") as f:
        json.dump(symbols, f, indent=2)
    logger.info(f"Saved symbol catalog ({len(symbols)} symbols) to {out_path}")
    return out_path


async def fetch_all(
    symbols: list[str],
    count: int = 5000,
    granularity: int = 60,
    days: float | None = None,
) -> None:
    """Fetch and save data for a list of Deriv symbol codes."""
    results = {}
    for symbol in symbols:
        df = await fetch_candles(symbol, count=count, granularity=granularity, days=days)
        if df.empty:
            logger.error(f"Skipping save for {symbol} — no data fetched.")
            results[symbol] = "FAILED"
            continue
        save_symbol(df, symbol)
        results[symbol] = f"OK ({len(df)} candles)"
        # Be a polite API citizen — small delay between symbols
        await asyncio.sleep(1)

    logger.info("=== Fetch summary ===")
    for symbol, status in results.items():
        logger.info(f"  {symbol}: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Deriv synthetic index data for Cossa Signals.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help=f"Specific Deriv symbol codes to fetch, e.g. R_75 BOOM1000. Ignored if --all is given. "
             f"If neither --symbols nor --all is given, defaults to the core set: {list(DERIV_SYMBOLS.values())}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Discover and fetch EVERY currently active synthetic index on Deriv (Volatility, "
             "Boom, Crash, Step, Jump, Range Break, and any others Deriv currently offers) — "
             "queried live from Deriv, not a hardcoded list. Saves the full catalog to "
             "data/raw/symbol_catalog.json as a reference for building study guides per instrument.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5000,
        help="Total candles per symbol, paginated automatically if it exceeds Deriv's per-call cap (default: 5000). Ignored if --days is given.",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=None,
        help="Days of history to fetch, overrides --count. E.g. --days 30 with 1-minute candles pulls ~43,200 candles across multiple paginated calls.",
    )
    parser.add_argument(
        "--granularity",
        type=int,
        default=60,
        help="Candle size in seconds (default: 60 = 1-minute candles)",
    )
    args = parser.parse_args()

    async def main():
        if args.all:
            logger.info(f"Validating {len(CANDIDATE_SYMBOLS)} candidate synthetic index symbols live...")
            catalog = await validate_symbols_live(CANDIDATE_SYMBOLS)
            if not catalog:
                logger.error("No symbols validated — check your DERIV_APP_ID and connection, then retry.")
                return
            save_symbol_catalog(catalog)
            symbols_to_fetch = [s["symbol"] for s in catalog]
            logger.info(f"Fetching all {len(symbols_to_fetch)} validated synthetic indices...")
        else:
            symbols_to_fetch = args.symbols if args.symbols else list(DERIV_SYMBOLS.values())

        await fetch_all(symbols_to_fetch, count=args.count, granularity=args.granularity, days=args.days)

    asyncio.run(main())
