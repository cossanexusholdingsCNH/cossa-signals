"""
Cossa Signals — Backtest Engine

Runs a strategy against historical OHLC data using vectorbt, applies
mandatory risk management (stop-loss, take-profit, position sizing),
and reports the metrics that actually matter: win rate, Sharpe ratio,
max drawdown, and total trades — not just "does it make money on this
one run."

This is the gate every strategy must pass before it's allowed to
generate a live or paper-trading signal. No strategy in strategies.py
should be trusted until it's been run through here on real historical
data with results that hold up.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

from strategies import STRATEGY_REGISTRY

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "backtest_results"


def load_data(symbol: str) -> pd.DataFrame:
    """Load a symbol's historical CSV, as saved by the data pipeline scripts."""
    path = RAW_DATA_DIR / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No data file found at {path}. Run the fetch script for this "
            f"symbol first (fetch_forex_data.py or fetch_deriv_data.py)."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def run_backtest(
    df: pd.DataFrame,
    strategy_name: str,
    initial_capital: float = 10_000.0,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
    fees_pct: float = 0.0005,
    **strategy_kwargs,
) -> dict:
    """
    Run one strategy through vectorbt with mandatory risk controls
    applied — every backtest in this project uses a stop-loss and
    take-profit by default. A strategy is not allowed to run "naked"
    (no risk limit) even in testing, because that hides how it would
    actually behave with real capital.

    Args:
        df: OHLC data (must have Open, High, Low, Close columns).
        strategy_name: Key into STRATEGY_REGISTRY.
        initial_capital: Starting capital for the simulation.
        stop_loss_pct: Stop-loss as a fraction of entry price (0.02 = 2%).
        take_profit_pct: Take-profit as a fraction of entry price.
        fees_pct: Per-trade fee/spread cost as a fraction (0.0005 = 5 bps) —
                  never backtest as if trading is free; spread/commission
                  costs are what turn a "profitable on paper" strategy
                  into a loser in practice.
        **strategy_kwargs: Passed through to the strategy function.

    Returns:
        Dict of performance metrics plus the vectorbt Portfolio object
        under "portfolio" for further inspection/plotting.
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. Available: {list(STRATEGY_REGISTRY.keys())}"
        )

    strategy_fn = STRATEGY_REGISTRY[strategy_name]
    entries, exits = strategy_fn(df, **strategy_kwargs)

    portfolio = vbt.Portfolio.from_signals(
        close=df["Close"],
        entries=entries,
        exits=exits,
        init_cash=initial_capital,
        fees=fees_pct,
        sl_stop=stop_loss_pct,
        tp_stop=take_profit_pct,
        freq="1min",  # adjust to match the granularity of the input data
    )

    total_trades = portfolio.trades.count()

    if total_trades == 0:
        return {
            "strategy": strategy_name,
            "total_trades": 0,
            "warning": "No trades were generated — strategy never fired on this data/parameters.",
        }

    win_rate = portfolio.trades.win_rate()
    sharpe = portfolio.sharpe_ratio()
    max_dd = portfolio.max_drawdown()
    total_return = portfolio.total_return()

    metrics = {
        "strategy": strategy_name,
        "total_trades": int(total_trades),
        "win_rate_pct": round(float(win_rate) * 100, 2) if not np.isnan(win_rate) else None,
        "sharpe_ratio": round(float(sharpe), 3) if not np.isnan(sharpe) else None,
        "max_drawdown_pct": round(float(max_dd) * 100, 2) if not np.isnan(max_dd) else None,
        "total_return_pct": round(float(total_return) * 100, 2) if not np.isnan(total_return) else None,
        "final_capital": round(initial_capital * (1 + total_return), 2) if not np.isnan(total_return) else None,
        "portfolio": portfolio,
    }
    return metrics


def print_report(metrics: dict) -> None:
    """Print a plain-English summary of backtest results — honest framing, no hype."""
    print(f"\n{'=' * 50}")
    print(f"Strategy: {metrics['strategy']}")
    print(f"{'=' * 50}")

    if metrics["total_trades"] == 0:
        print(f"  {metrics.get('warning', 'No trades generated.')}")
        return

    print(f"  Total trades:      {metrics['total_trades']}")
    print(f"  Win rate:          {metrics['win_rate_pct']}%")
    print(f"  Sharpe ratio:      {metrics['sharpe_ratio']}")
    print(f"  Max drawdown:      {metrics['max_drawdown_pct']}%")
    print(f"  Total return:      {metrics['total_return_pct']}%")
    print(f"  Final capital:     {metrics['final_capital']}")
    print()
    print("  Reminder: a positive backtest is not a guarantee. Validate with")
    print("  paper trading before any real signal goes to a client.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a backtest for Cossa Signals.")
    parser.add_argument("--symbol", required=True, help="Symbol name matching a file in data/raw/")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=list(STRATEGY_REGISTRY.keys()),
        help="Which strategy to test",
    )
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--stop-loss", type=float, default=0.02)
    parser.add_argument("--take-profit", type=float, default=0.04)
    args = parser.parse_args()

    data = load_data(args.symbol)
    result = run_backtest(
        data,
        strategy_name=args.strategy,
        initial_capital=args.capital,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
    )
    print_report(result)
