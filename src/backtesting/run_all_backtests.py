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

from backtest_engine import load_data, run_backtest, resample_ohlc, train_test_split_ohlc
from strategies import STRATEGY_REGISTRY

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "backtest_results"

# Which strategies make sense for which instrument family. Symbol prefixes
# are used to classify — this mirrors the actual symbol codes the data
# pipeline fetches (R_*, 1HZ*V, BOOM*, CRASH*, RDBULL/RDBEAR, stpRNG).
BOOM_CRASH_PREFIXES = ("BOOM", "CRASH")

# RDBULL/RDBEAR get a purpose-built strategy in addition to the generic
# rsi_mean_reversion, so results are directly comparable in the same run.
# Direct measurement (check_daily_reset.py) confirmed both instruments have
# a real, large price discontinuity at 00:00 GMT — RDBULL drops ~8.6% at
# reset, RDBEAR jumps ~5.3% — and rsi_mean_reversion's only real edge
# across all 17 instruments showed up on exactly these two. daily_reset_
# reversion tests whether trading that known window directly beats RSI
# stumbling into it by accident.
DAILY_RESET_STRATEGIES = {
    "RDBULL": "daily_reset_reversion_bull",
    "RDBEAR": "daily_reset_reversion_bear",
}

# Momentum crossover lost money on all 17/17 instruments in the first full
# batch run (1-minute candles, 30 days) — a structurally consistent result,
# not noise, since these instruments are engineered to have no persistent
# trend (see the trading guide, Chapter 1). Not worth re-testing by default;
# pass --include-momentum to run it anyway if you want the comparison point.
GENERAL_STRATEGIES = ["rsi_mean_reversion"]
GENERAL_STRATEGIES_WITH_MOMENTUM = ["momentum_ma_crossover", "rsi_mean_reversion"]
BOOM_CRASH_STRATEGIES = ["spike_reversion_boom_crash"]

# Below this many trades, win rate and return are statistically meaningless —
# a single trade "winning" is not a 100% win rate in any usable sense. The
# 15-minute resample run surfaced exactly this trap (1HZ50V: 2 trades,
# "100% win rate") — this threshold exists so results like that never get
# ranked alongside genuinely tested ones without a clear warning.
MIN_TRUSTWORTHY_TRADES = 30


def classify_symbol(symbol: str, include_momentum: bool = False) -> list[str]:
    """Return the list of strategy names appropriate to test for this symbol."""
    if symbol.startswith(BOOM_CRASH_PREFIXES):
        return BOOM_CRASH_STRATEGIES
    base = GENERAL_STRATEGIES_WITH_MOMENTUM if include_momentum else GENERAL_STRATEGIES
    if symbol in DAILY_RESET_STRATEGIES:
        return base + [DAILY_RESET_STRATEGIES[symbol]]
    return base


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
    resample_rule: str | None = None,
    include_momentum: bool = False,
    validate_split: bool = False,
    risk_pct_per_trade: float = 0.10,
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

        original_len = len(df)
        if resample_rule:
            df = resample_ohlc(df, resample_rule)

        # Buy-and-hold benchmark: simply holding the instrument over the same
        # window, no strategy at all. Exists because RDBULL/RDBEAR returned
        # 800-900%+ on the 180-day/5-min run and it turned out to be almost
        # entirely the instrument's own structural drift (Deriv's "Bull
        # Market"/"Bear Market" indices are designed to trend persistently),
        # not genuine strategy edge. Any strategy result should be judged
        # against this number — a strategy that barely beats buy-and-hold,
        # or loses to it, has NOT found an edge no matter how good its
        # standalone return looks.
        buy_hold_pct = round(
            float((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100), 2
        )

        strategies_to_test = classify_symbol(symbol, include_momentum=include_momentum)
        span_note = f" -> resampled to {resample_rule} ({len(df)} candles)" if resample_rule else ""
        candle_span = f"{original_len} candles ({df.index.min()} to {df.index.max()}){span_note}"

        for strategy_name in strategies_to_test:
            result = run_backtest(
                df,
                strategy_name=strategy_name,
                initial_capital=initial_capital,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                risk_pct_per_trade=risk_pct_per_trade,
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
                    "buy_hold_pct": buy_hold_pct,
                    "beats_buy_hold": None,
                    "note": "no trades generated",
                })
                continue

            beats_buy_hold = result["total_return_pct"] is not None and result["total_return_pct"] > buy_hold_pct

            rows.append({
                "symbol": symbol,
                "strategy": strategy_name,
                "data_span": candle_span,
                "trades": result["total_trades"],
                "win_rate_pct": result["win_rate_pct"],
                "sharpe": result["sharpe_ratio"],
                "max_dd_pct": result["max_drawdown_pct"],
                "total_return_pct": result["total_return_pct"],
                "buy_hold_pct": buy_hold_pct,
                "beats_buy_hold": beats_buy_hold,
                "trustworthy": result["total_trades"] >= MIN_TRUSTWORTHY_TRADES,
                "note": "" if result["total_trades"] >= MIN_TRUSTWORTHY_TRADES
                        else f"only {result['total_trades']} trades — below {MIN_TRUSTWORTHY_TRADES}, treat as noise",
            })
            print(f"  [DONE] {symbol:12} {strategy_name:28} "
                  f"trades={result['total_trades']:>5}  "
                  f"win_rate={result['win_rate_pct']}%  "
                  f"return={result['total_return_pct']}%  "
                  f"buy_hold={buy_hold_pct}%  "
                  f"{'BEATS B&H' if beats_buy_hold else 'loses to B&H'}")

            # Train/test split validation — the real check before trusting
            # any result. A strategy that only works on the slice it was
            # (implicitly) eyeballed against is overfitting to that window,
            # not finding real edge. Nothing in this project had been run
            # through this before the 180-day/5-min batch, which is exactly
            # why RDBULL/RDBEAR's numbers made it as far as a printout
            # before anyone caught the problem.
            if validate_split and result["total_trades"] > 0:
                train_df, test_df = train_test_split_ohlc(df, train_frac=0.7)
                train_result = run_backtest(
                    train_df, strategy_name=strategy_name,
                    initial_capital=initial_capital,
                    stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
                    risk_pct_per_trade=risk_pct_per_trade,
                )
                test_result = run_backtest(
                    test_df, strategy_name=strategy_name,
                    initial_capital=initial_capital,
                    stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
                    risk_pct_per_trade=risk_pct_per_trade,
                )
                train_trades = train_result.get("total_trades", 0)
                test_trades = test_result.get("total_trades", 0)
                train_ret = train_result.get("total_return_pct")
                test_ret = test_result.get("total_return_pct")

                holds_up = (
                    test_trades >= 10  # too few test-slice trades to say anything
                    and test_ret is not None and train_ret is not None
                    and test_ret > 0 and train_ret > 0  # BOTH halves must agree, not just test —
                    # a strategy that loses overall but got a lucky positive test slice
                    # (e.g. R_100, R_50 on the 180-day/5-min run: train negative, test
                    # barely positive, overall negative) is not a validated edge, it's
                    # noise that happened to land on the right side once. Only requiring
                    # test_ret > 0 let exactly that slip through as "HOLDS UP" — fixed here.
                )
                verdict = (
                    "HOLDS UP on unseen data" if holds_up
                    else "DOES NOT HOLD UP on unseen data — overfit to this window, not a real edge"
                )
                print(
                    f"           [SPLIT] train: {train_trades} trades, {train_ret}%  |  "
                    f"test: {test_trades} trades, {test_ret}%  -->  {verdict}"
                )
                rows[-1]["split_train_return_pct"] = train_ret
                rows[-1]["split_train_trades"] = train_trades
                rows[-1]["split_test_return_pct"] = test_ret
                rows[-1]["split_test_trades"] = test_trades
                rows[-1]["split_holds_up"] = holds_up

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

    trustworthy = tested[tested["trustworthy"]].sort_values("total_return_pct", ascending=False)
    noise = tested[~tested["trustworthy"]].sort_values("total_return_pct", ascending=False)

    print(f"\n{'=' * 70}")
    print(f"TRUSTWORTHY RESULTS (>= {MIN_TRUSTWORTHY_TRADES} trades) — ranked by return")
    print(f"{'=' * 70}")
    if trustworthy.empty:
        print("  None. Every combination this run had too few trades to draw a real")
        print("  conclusion from — see below. This is a legitimate outcome to report,")
        print("  not a gap to paper over: it means we need more data or a different")
        print("  approach before any result here can be trusted.")
    else:
        for _, row in trustworthy.iterrows():
            flag = "" if row.get("beats_buy_hold") else "  <-- DOES NOT BEAT BUY-AND-HOLD, likely riding instrument drift, not real edge"
            print(
                f"  {row['symbol']:12} {row['strategy']:28} "
                f"return={row['total_return_pct']:>7}%  buy_hold={row.get('buy_hold_pct', 'n/a'):>7}%  "
                f"win_rate={row['win_rate_pct']:>6}%  "
                f"sharpe={row['sharpe']:>7}  trades={row['trades']:>5}  max_dd={row['max_dd_pct']}%{flag}"
            )

    print(f"\n{'=' * 70}")
    print(f"TOO FEW TRADES TO TRUST (< {MIN_TRUSTWORTHY_TRADES} trades) — for reference only, NOT ranked as findings")
    print(f"{'=' * 70}")
    for _, row in noise.iterrows():
        print(
            f"  {row['symbol']:12} {row['strategy']:28} "
            f"return={row['total_return_pct']:>7}%  win_rate={row['win_rate_pct']:>6}%  "
            f"trades={row['trades']:>3}  <- statistically meaningless at this count"
        )

    print(f"\n{'=' * 70}")
    print("REMINDER — before trusting any row above:")
    print(f"{'=' * 70}")
    print("  - A positive backtest is a candidate for paper trading, not a proven edge.")
    print("  - Check the data_span column in the CSV — a strategy tested on only a")
    print("    few days of data is far less trustworthy than one tested on 30+ days.")
    print(f"  - Trade counts under {MIN_TRUSTWORTHY_TRADES} are reported separately above")
    print("    on purpose — they are not ranked as findings, only shown for reference.")
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
    parser.add_argument(
        "--resample",
        default=None,
        help="Resample 1-minute data to a wider candle before testing, e.g. 5min, 15min, 1h. "
             "Reduces trade frequency and fee drag — worth trying since RSI mean-reversion was "
             "the only strategy showing any positive signal on the 1-minute batch run.",
    )
    parser.add_argument(
        "--include-momentum",
        action="store_true",
        help="Also test momentum_ma_crossover, which lost money on 17/17 instruments in the "
             "first batch run. Off by default now — pass this to re-include it anyway.",
    )
    parser.add_argument(
        "--validate-split",
        action="store_true",
        help="For every symbol tested, also run the SAME strategy separately on the first "
             "70%% (train) and last 30%% (test) of the data using train_test_split_ohlc(), "
             "and report both. This is the real check before trusting any result: a strategy "
             "that only performs on the train slice and falls apart on the untouched test "
             "slice was overfit to that specific window, not genuinely predictive. Nothing "
             "in this project has been run through this check yet — do this before paper "
             "trading anything, especially R_75.",
    )
    parser.add_argument(
        "--risk-pct",
        type=float,
        default=0.10,
        help="Fraction of available cash deployed per trade (0.10 = 10%%, the default). "
             "Lower this (e.g. 0.02) to sanity-check whether a promising result survives "
             "a more conservative, institutional-style position size, or whether it's "
             "still partly an artifact of how much capital gets compounded per trade.",
    )
    args = parser.parse_args()

    symbols_to_test = args.symbols if args.symbols else discover_available_symbols()

    if not symbols_to_test:
        print("No data found in data/raw/. Run the fetch scripts first.")
        exit(1)

    print(f"Running batch backtest across {len(symbols_to_test)} symbol(s): {symbols_to_test}")
    if args.resample:
        print(f"Resampling to {args.resample} candles before testing")
    print()

    results = run_batch(
        symbols_to_test,
        initial_capital=args.capital,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        resample_rule=args.resample,
        include_momentum=args.include_momentum,
        validate_split=args.validate_split,
        risk_pct_per_trade=args.risk_pct,
    )
    save_and_summarize(results)
