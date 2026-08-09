"""
Cossa Signals — Batch Backtest Runner

Runs every relevant strategy against every symbol found in data/raw/,
in one command, and produces a single ranked summary — instead of
running backtest_engine.py by hand, once per symbol/strategy pair.

Strategy assignment is instrument-aware: spike_reversion_boom_crash only
makes sense on Boom/Crash indices (their whole premise is the spike
behavior), while momentum and mean-reversion are tested on everything
else. Running a Boom/Crash-specific strategy against a Volatility index
would be a meaningless test — this runner doesn't do that.

Usage:
    python src/backtesting/run_all_backtests.py
    python src/backtesting/run_all_backtests.py --capital 5000 --stop-loss 0.015
"""

import argparse
from pathlib import Path

import pandas as pd

from backtest_engine import load_data, run_backtest
from strategies import STRATEGY_REGISTRY

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "backtest_results"

# Which strategies make sense for which instrument family. Symbol prefixes
# are used to classify — this mirrors the actual symbol codes the data
# pipeline fetches (R_*, 1HZ*V, BOOM*, CRASH*, RDBULL/RDBEAR, stpRNG).
BOOM_CRASH_PREFIXES = ("BOOM", "CRASH")
GENERAL_STRATEGIES = ["momentum_ma_crossover", "rsi_mean_reversion"]
BOOM_CRASH_STRATEGIES = ["spike_reversion_boom_crash"]


def classify_symbol(symbol: str) -> list[str]:
    """Return the list of strategy names appropriate to test for this symbol."""
    if symbol.startswith(BOOM_CRASH_PREFIXES):
        return BOOM_CRASH_STRATEGIES
    return GENERAL_STRATEGIES


def discover_available_symbols() -> list[str]:
    """Find every symbol with fetched data sitting in data/raw/."""
    if not RAW_DATA_DIR.exists():
        return []
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    return [f.stem for f in csv_files]


def run_batch(
    symbols: list[str],
    initial_capital: float = 10_000.0,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
) -> pd.DataFrame:
    """
    Run every applicable strategy against every symbol, collect results
    into a single ranked DataFrame.
    """
    rows = []

    for symbol in symbols:
        try:
            df = load_data(symbol)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {symbol}: {exc}")
            continue

        strategies_to_test = classify_symbol(symbol)
        candle_span = f"{len(df)} candles ({df.index.min()} to {df.index.max()})"

        for strategy_name in strategies_to_test:
            result = run_backtest(
                df,
                strategy_name=strategy_name,
                initial_capital=initial_capital,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )

            if result.get("total_trades", 0) == 0:
                rows.append({
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "data_span": candle_span,
                    "trades": 0,
                    "win_rate_pct": None,
                    "sharpe": None,
                    "max_dd_pct": None,
                    "total_return_pct": None,
                    "note": "no trades generated",
                })
                continue

            rows.append({
                "symbol": symbol,
                "strategy": strategy_name,
                "data_span": candle_span,
                "trades": result["total_trades"],
                "win_rate_pct": result["win_rate_pct"],
                "sharpe": result["sharpe_ratio"],
                "max_dd_pct": result["max_drawdown_pct"],
                "total_return_pct": result["total_return_pct"],
                "note": "",
            })
            print(f"  [DONE] {symbol:12} {strategy_name:28} "
                  f"trades={result['total_trades']:>5}  "
                  f"win_rate={result['win_rate_pct']}%  "
                  f"return={result['total_return_pct']}%")

    return pd.DataFrame(rows)


def save_and_summarize(results: pd.DataFrame) -> None:
    """Save the full results table and print a ranked, honest summary."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "batch_backtest_summary.csv"
    results.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")

    tested = results[results["trades"] > 0].copy()
    if tested.empty:
        print("\nNo strategy/symbol combination generated any trades. Nothing to rank.")
        return

    print(f"\n{'=' * 70}")
    print("RANKED BY TOTAL RETURN (highest first) — trades > 0 only")
    print(f"{'=' * 70}")
    ranked = tested.sort_values("total_return_pct", ascending=False)
    for _, row in ranked.iterrows():
        print(
            f"  {row['symbol']:12} {row['strategy']:28} "
            f"return={row['total_return_pct']:>7}%  win_rate={row['win_rate_pct']:>6}%  "
            f"sharpe={row['sharpe']:>7}  trades={row['trades']:>5}  max_dd={row['max_dd_pct']}%"
        )

    print(f"\n{'=' * 70}")
    print("REMINDER — before trusting any row above:")
    print(f"{'=' * 70}")
    print("  - A positive backtest is a candidate for paper trading, not a proven edge.")
    print("  - Check the data_span column in the CSV — a strategy tested on only a")
    print("    few days of data is far less trustworthy than one tested on 30+ days.")
    print("  - Low trade counts (under ~30-50) mean the win rate/return numbers")
    print("    could easily be noise, not a real pattern.")
    print("  - The next real step for any promising result is paper trading — never")
    print("    real capital straight off a backtest, no matter how good it looks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run every strategy against every fetched symbol.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Specific symbols to test (matching filenames in data/raw/, no .csv). Default: every symbol found.",
    )
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--stop-loss", type=float, default=0.02)
    parser.add_argument("--take-profit", type=float, default=0.04)
    args = parser.parse_args()

    symbols_to_test = args.symbols if args.symbols else discover_available_symbols()

    if not symbols_to_test:
        print("No data found in data/raw/. Run the fetch scripts first.")
        exit(1)

    print(f"Running batch backtest across {len(symbols_to_test)} symbol(s): {symbols_to_test}\n")
    results = run_batch(
        symbols_to_test,
        initial_capital=args.capital,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
    )
    save_and_summarize(results)
