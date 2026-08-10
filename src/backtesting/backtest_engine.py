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


def train_test_split_ohlc(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split OHLC data chronologically into a training portion and a held-out
    test portion. This is the real safeguard against the trap we just hit:
    if a strategy only looks good on the exact data we tuned it against
    (or the exact timeframe we happened to pick), that's overfitting, not
    edge. A strategy with genuine edge should perform reasonably on data
    it never saw. One that only works on the training slice and falls
    apart on the test slice was fooling us, not working.

    Args:
        df: OHLC data, chronologically ordered (as loaded by load_data()).
        train_frac: Fraction of the data (earliest portion) used for
                    training/tuning. The remainder is the untouched
                    test set — do not tune parameters against it.

    Returns:
        (train_df, test_df) — chronologically split, no overlap.
    """
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample 1-minute OHLC data to a wider candle size, e.g. "5min" or "15min".

    Why this matters: at 1-minute granularity, RSI mean-reversion fired
    14-28 times over 30 days and was the only strategy showing any positive
    signal; momentum fired 90-180 times and lost heavily everywhere. High
    trade frequency on tight stop/take-profit levels means per-trade fees
    compound fast regardless of whether the entry logic has real merit.
    Wider candles mean fewer, more deliberate signals — this tests whether
    fee drag was masking real edge, or whether the strategies simply don't
    work here at any timeframe.

    Args:
        df: OHLC DataFrame with a datetime index (as loaded by load_data()).
        rule: Pandas resample rule, e.g. "5min", "15min", "1h".

    Returns:
        Resampled OHLC DataFrame — Open of first candle in each bucket,
        High/Low as the max/min across the bucket, Close of the last candle.
    """
    resampled = df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }).dropna()
    return resampled


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


def infer_freq(df: pd.DataFrame) -> str:
    """
    Infer the pandas-compatible bar frequency directly from the data's own
    datetime index, instead of trusting a caller to pass the right one.

    Why this exists: run_backtest() previously hardcoded freq="1min". That
    was silently wrong the moment any script fed it 5-minute or resampled
    data — vectorbt uses `freq` to annualize Sharpe (periods-per-year), so
    a 5-minute series interpreted as 1-minute bars overstates the number of
    trading periods per year by 5x, which inflates Sharpe by roughly
    sqrt(5) (~2.2x). This surfaced for real on the 180-day/5-min batch run:
    RDBULL reported Sharpe 10.9 — a number no real strategy produces —
    largely because of this bug, not genuine risk-adjusted performance.

    Inferring freq from the median gap between consecutive index timestamps
    means this is correct automatically, regardless of what granularity was
    fetched or what --resample rule was applied upstream. No caller needs
    to remember to pass it.
    """
    if len(df.index) < 2:
        return "1min"  # can't infer from <2 points; harmless fallback
    median_gap = df.index.to_series().diff().median()
    return pd.tseries.frequencies.to_offset(median_gap).freqstr


def run_backtest(
    df: pd.DataFrame,
    strategy_name: str,
    initial_capital: float = 10_000.0,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
    fees_pct: float = 0.0005,
    risk_pct_per_trade: float = 0.10,
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
        risk_pct_per_trade: Fraction of AVAILABLE CASH deployed per trade
                  (0.10 = 10%). Without this, vectorbt's default is to
                  deploy 100% of available cash on every entry, which
                  compounds unrealistically over a long trade sequence —
                  this is what actually drove the 800-900%+ "returns" on
                  RDBULL/RDBEAR in the 180-day/5-min run, not real edge.
                  No real trader risks 100% of capital on a single
                  stop/take-profit bracket; this was always the missing
                  half of "mandatory risk management" (SL/TP levels were
                  implemented, position sizing was not — until now).
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
        size=risk_pct_per_trade,
        size_type="percent",  # deploy risk_pct_per_trade of available cash, not 100% of it
        freq=infer_freq(df),  # correct annualization regardless of candle width
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
