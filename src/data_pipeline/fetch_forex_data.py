"""
Cossa Signals — Forex Data Fetcher

Pulls historical OHLCV data for major forex pairs using yfinance (free,
no API key required) and saves it to data/raw/ as CSV, ready for the
backtesting engine.

This is the MVP data source. Once the strategy is validated, we swap or
supplement this with the OANDA API for live/streaming data — see
fetch_oanda_data.py (to be added in the next phase).

Usage:
    python src/data_pipeline/fetch_forex_data.py
    python src/data_pipeline/fetch_forex_data.py --pairs EURUSD GBPUSD --period 5y
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# yfinance ticker format for major forex pairs (Yahoo Finance convention)
FOREX_TICKERS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDZAR": "USDZAR=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
}

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_pair(pair: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for one forex pair.

    Args:
        pair: Pair code, e.g. "EURUSD" (must be a key in FOREX_TICKERS).
        period: How far back to pull data. yfinance accepts values like
                "1y", "2y", "5y", "max".
        interval: Candle size. Daily ("1d") is the right starting point
                  for MVP backtesting — intraday data is noisier and
                  should wait until the daily-timeframe strategy proves
                  a real edge.

    Returns:
        A DataFrame with columns: Open, High, Low, Close, Volume,
        indexed by date. Empty DataFrame if the fetch fails.
    """
    if pair not in FOREX_TICKERS:
        raise ValueError(
            f"Unknown pair '{pair}'. Available pairs: {list(FOREX_TICKERS.keys())}"
        )

    ticker = FOREX_TICKERS[pair]
    logger.info(f"Fetching {pair} ({ticker}) — period={period}, interval={interval}")

    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
    except Exception as exc:
        logger.error(f"Failed to fetch {pair}: {exc}")
        return pd.DataFrame()

    if df.empty:
        logger.warning(f"No data returned for {pair}. Check ticker/period/interval.")
        return df

    # Keep only what we need for backtesting; drop Dividends/Stock Splits
    # columns yfinance adds by default (not meaningful for forex).
    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep_cols].copy()
    df.index.name = "Date"

    logger.info(f"Fetched {len(df)} rows for {pair} ({df.index.min()} to {df.index.max()})")
    return df


def save_pair(df: pd.DataFrame, pair: str) -> Path:
    """Save a pair's DataFrame to data/raw/{pair}.csv."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / f"{pair}.csv"
    df.to_csv(out_path)
    logger.info(f"Saved {pair} data to {out_path}")
    return out_path


def fetch_all(pairs: list[str], period: str = "5y", interval: str = "1d") -> None:
    """Fetch and save data for a list of pairs, logging any failures clearly."""
    results = {}
    for pair in pairs:
        df = fetch_pair(pair, period=period, interval=interval)
        if df.empty:
            logger.error(f"Skipping save for {pair} — no data fetched.")
            results[pair] = "FAILED"
            continue
        save_pair(df, pair)
        results[pair] = "OK"

    logger.info("=== Fetch summary ===")
    for pair, status in results.items():
        logger.info(f"  {pair}: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical forex data for Cossa Signals.")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["EURUSD", "GBPUSD", "USDZAR"],
        help="Forex pairs to fetch (default: EURUSD GBPUSD USDZAR)",
    )
    parser.add_argument(
        "--period",
        default="5y",
        help="How far back to pull data, e.g. 1y, 2y, 5y, max (default: 5y)",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Candle interval, e.g. 1d, 1h (default: 1d)",
    )
    args = parser.parse_args()

    fetch_all(args.pairs, period=args.period, interval=args.interval)
