"""
Cossa Signals — Live Paper Trading Harness

Watches Deriv's live price feed for RDBULL and RDBEAR and simulates the
daily_reset_reversion strategy in real time. NO REAL MONEY IS EVER
TOUCHED — this only logs what a trade would have done. This is the
gate between "backtest looks good" and "ready to discuss real capital."

WHY THIS EXISTS: four batch backtest runs, a Sharpe-annualization bug fix,
a position-sizing fix, a buy-and-hold benchmark, and a confirmed structural
daily-reset pattern all point to a real edge on RDBULL/RDBEAR. None of
that is worth anything until it's been watched trade on data the strategy
never saw during development. That's what this script is for.

IMPORTANT — like fetch_deriv_data.py, this has NOT been tested against a
live connection in the environment that built it (no route to deriv.com
from that sandbox). Run this locally first and watch the first few
scheduled entries/exits closely before trusting it to run unattended.

WHAT IT DOES:
  - Connects to Deriv's public WebSocket API (no auth needed — this only
    reads price data, it never places a real order).
  - Subscribes to live 5-min candles for RDBULL and RDBEAR.
  - RDBULL: opens a simulated long at 00:00 GMT, closes it 30 min before
    the next reset (or sooner if stop-loss/take-profit is hit).
  - RDBEAR: opens a simulated long 30 min before reset, closes it at
    00:00 GMT (or sooner if stop-loss/take-profit is hit).
  - Applies the same 2% stop-loss / 4% take-profit / fee assumption as
    the backtest, so results are directly comparable.
  - Logs every trade to data/paper_trades/trades.csv (append-only) and
    persists open-position state to data/paper_trades/state.json, so a
    restart (crash, reboot, closed terminal) resumes correctly instead
    of losing track of an open position.
  - Reconnects automatically on a dropped connection — this is meant to
    run unattended for weeks, not require babysitting.

Usage:
    python src/delivery/paper_trade.py
    python src/delivery/paper_trade.py --symbols RDBULL RDBEAR --capital 10000

Stop with Ctrl+C at any time — state is saved continuously, not just on
a clean exit, so this is always safe to interrupt.
"""

import argparse
import asyncio
import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import websockets
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ENV_PATH = Path(__file__).resolve().parents[2] / "config" / ".env"
load_dotenv(ENV_PATH)

DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

PAPER_TRADE_DIR = Path(__file__).resolve().parents[2] / "data" / "paper_trades"
STATE_PATH = PAPER_TRADE_DIR / "state.json"
TRADES_CSV_PATH = PAPER_TRADE_DIR / "trades.csv"

GRANULARITY_SEC = 300  # 5-min candles, matching the validated backtest

# direction: "post_reset_long" (RDBULL) or "pre_reset_long" (RDBEAR) —
# same semantics as daily_reset_reversion() in src/backtesting/strategies.py
SYMBOL_CONFIG = {
    "RDBULL": {"direction": "post_reset_long", "entry_window_min": 30, "exit_window_min": 30},
    "RDBEAR": {"direction": "pre_reset_long", "entry_window_min": 30, "exit_window_min": 30},
}

STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04
FEES_PCT = 0.0005  # keep in sync with --fees-pct findings from run_all_backtests.py


def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"positions": {}, "equity": {}, "last_processed_epoch": {}}


def save_state(state: dict) -> None:
    PAPER_TRADE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def append_trade_log(row: dict) -> None:
    PAPER_TRADE_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = TRADES_CSV_PATH.exists()
    with open(TRADES_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "symbol", "entry_time", "entry_price", "exit_time", "exit_price",
            "exit_reason", "return_pct", "equity_after",
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def minutes_since_midnight_gmt(epoch: int) -> int:
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return dt.hour * 60 + dt.minute


def should_enter(symbol: str, epoch: int) -> bool:
    cfg = SYMBOL_CONFIG[symbol]
    m = minutes_since_midnight_gmt(epoch)
    if cfg["direction"] == "post_reset_long":
        return m < cfg["entry_window_min"]
    else:  # pre_reset_long
        return m >= (1440 - cfg["entry_window_min"])


def should_exit_on_schedule(symbol: str, epoch: int) -> bool:
    cfg = SYMBOL_CONFIG[symbol]
    m = minutes_since_midnight_gmt(epoch)
    if cfg["direction"] == "post_reset_long":
        return m >= (1440 - cfg["exit_window_min"])
    else:  # pre_reset_long
        return m < cfg["exit_window_min"]


def process_candle(symbol: str, candle: dict, state: dict, capital: float) -> float:
    """
    Evaluate one new candle against the current position state for a
    symbol. Returns the (possibly updated) capital for this symbol.
    Mutates `state` in place and writes a trade log row on any close.
    """
    epoch = int(candle["epoch"])
    close_price = float(candle["close"])
    open_price = float(candle["open"])
    high_price = float(candle["high"])
    low_price = float(candle["low"])
    candle_time = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

    # De-dupe: Deriv can resend the same in-progress candle before it
    # closes. Only act once per completed candle epoch.
    last_seen = state["last_processed_epoch"].get(symbol)
    if last_seen == epoch:
        return capital
    state["last_processed_epoch"][symbol] = epoch

    position = state["positions"].get(symbol)

    if position is None:
        if should_enter(symbol, epoch):
            state["positions"][symbol] = {
                "entry_time": candle_time,
                "entry_price": close_price,
            }
            logger.info(f"[{symbol}] ENTRY at {candle_time} price={close_price}")
        return capital

    # Have an open position — check stop-loss / take-profit first (using
    # this candle's high/low, not just close, since a real stop can be
    # hit intra-candle), then the scheduled exit window.
    entry_price = position["entry_price"]
    sl_price = entry_price * (1 - STOP_LOSS_PCT)
    tp_price = entry_price * (1 + TAKE_PROFIT_PCT)

    exit_price = None
    exit_reason = None
    if low_price <= sl_price:
        exit_price = sl_price
        exit_reason = "stop_loss"
    elif high_price >= tp_price:
        exit_price = tp_price
        exit_reason = "take_profit"
    elif should_exit_on_schedule(symbol, epoch):
        exit_price = close_price
        exit_reason = "scheduled_exit"

    if exit_price is not None:
        gross_return_pct = (exit_price - entry_price) / entry_price * 100
        net_return_pct = gross_return_pct - (FEES_PCT * 100 * 2)  # entry + exit fee
        capital = capital * (1 + net_return_pct / 100)
        append_trade_log({
            "symbol": symbol,
            "entry_time": position["entry_time"],
            "entry_price": entry_price,
            "exit_time": candle_time,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "return_pct": round(net_return_pct, 4),
            "equity_after": round(capital, 2),
        })
        logger.info(
            f"[{symbol}] EXIT ({exit_reason}) at {candle_time} price={exit_price} "
            f"return={net_return_pct:.3f}%  equity={capital:.2f}"
        )
        state["positions"][symbol] = None

    return capital


async def run_symbol_stream(ws, symbol: str, state: dict, capital_holder: dict):
    """Subscribe to live candles for one symbol and process each update."""
    request = {
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": 1,
        "end": "latest",
        "start": 1,
        "style": "candles",
        "granularity": GRANULARITY_SEC,
        "subscribe": 1,
    }
    await ws.send(json.dumps(request))

    while True:
        response_raw = await ws.recv()
        response = json.loads(response_raw)

        if "error" in response:
            logger.error(f"[{symbol}] Deriv API error: {response['error'].get('message')}")
            continue

        # Initial response has "candles" (a list); streamed updates have "ohlc" (one candle)
        candles = response.get("candles", [])
        if "ohlc" in response:
            candles = [response["ohlc"]]

        for candle in candles:
            capital_holder[symbol] = process_candle(symbol, candle, state, capital_holder[symbol])
            save_state(state)  # persist after every candle — never lose an open position


async def main(symbols: list[str], initial_capital: float):
    state = load_state()
    for symbol in symbols:
        state["positions"].setdefault(symbol, None)
        state["equity"].setdefault(symbol, initial_capital)
        state["last_processed_epoch"].setdefault(symbol, None)
    capital_holder = {s: state["equity"][s] for s in symbols}

    logger.info(f"Paper trading started for {symbols}. NO REAL MONEY IS TOUCHED. "
                f"Trade log: {TRADES_CSV_PATH}")
    if any(state["positions"].get(s) for s in symbols):
        logger.info(f"Resumed with open position(s) from previous run: "
                    f"{ {s: state['positions'][s] for s in symbols if state['positions'][s]} }")

    reconnect_delay = 5
    while True:
        try:
            async with websockets.connect(DERIV_WS_URL) as ws:
                logger.info("Connected to Deriv WebSocket.")
                reconnect_delay = 5  # reset backoff on a successful connection
                tasks = [
                    asyncio.create_task(run_symbol_stream(ws, symbol, state, capital_holder))
                    for symbol in symbols
                ]
                await asyncio.gather(*tasks)
        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            logger.warning(f"Connection lost ({e}). Reconnecting in {reconnect_delay}s...")
            for symbol in symbols:
                state["equity"][symbol] = capital_holder[symbol]
            save_state(state)
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 300)  # exponential backoff, capped at 5 min


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", default=["RDBULL", "RDBEAR"],
                         choices=list(SYMBOL_CONFIG.keys()),
                         help="Which symbols to paper trade. Default: both.")
    parser.add_argument("--capital", type=float, default=10_000.0,
                         help="Starting simulated capital per symbol (default: 10000).")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.symbols, args.capital))
    except KeyboardInterrupt:
        logger.info("Stopped by user. State has been saved continuously — safe to resume later.")
