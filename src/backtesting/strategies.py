"""
Cossa Signals — Baseline Strategies

These are STARTING-POINT strategies for the backtesting engine to
evaluate — not proven, not recommended for live capital. Every one of
these must be backtested and shown to have real, statistically
significant edge before it generates a live signal. Treat this file
as a library of hypotheses to test, not a source of truth.

Each strategy function takes a DataFrame with OHLC columns and returns
two boolean Series: entries and exits, aligned to the same index —
the format vectorbt's Portfolio.from_signals() expects.
"""

import pandas as pd
import ta


def momentum_ma_crossover(
    df: pd.DataFrame,
    fast_window: int = 10,
    slow_window: int = 30,
) -> tuple[pd.Series, pd.Series]:
    """
    Simple moving-average crossover — long when fast MA crosses above
    slow MA, exit when it crosses back below.

    Suited to trending instruments. On Volatility indices this tests
    whether short-term momentum persists; on Boom/Crash it will mostly
    trade the drift between spikes, and needs the risk layer (stop-loss)
    to handle the spike itself, which this signal alone does not predict.
    """
    fast_ma = df["Close"].rolling(fast_window).mean()
    slow_ma = df["Close"].rolling(slow_window).mean()

    entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

    return entries.fillna(False), exits.fillna(False)


def rsi_mean_reversion(
    df: pd.DataFrame,
    rsi_window: int = 14,
    oversold: int = 30,
    overbought: int = 70,
) -> tuple[pd.Series, pd.Series]:
    """
    RSI mean-reversion — long when RSI drops below the oversold
    threshold (expecting a bounce), exit when RSI recovers above the
    overbought threshold.

    Relevant to Volatility indices, which have no real "trend" driver
    and tend to oscillate around statistical norms — a genuinely
    different behavior from trending forex pairs.
    """
    rsi = ta.momentum.RSIIndicator(close=df["Close"], window=rsi_window).rsi()

    entries = (rsi < oversold) & (rsi.shift(1) >= oversold)
    exits = (rsi > overbought) & (rsi.shift(1) <= overbought)

    return entries.fillna(False), exits.fillna(False)


def spike_reversion_boom_crash(
    df: pd.DataFrame,
    lookback: int = 20,
    spike_std_threshold: float = 3.0,
    hold_bars: int = 5,
) -> tuple[pd.Series, pd.Series]:
    """
    Spike-reversion, purpose-built for Boom/Crash index behavior.

    Boom/Crash indices drift steadily in one direction and then have a
    sharp, large spike in the opposite direction at semi-random tick
    intervals. This strategy does NOT try to predict when a spike
    happens — that's the whole point of these instruments being
    statistically unpredictable at the individual-event level. Instead,
    it tests a narrower, more honest hypothesis: after a spike has just
    occurred (detected via an abnormal price move relative to recent
    volatility), does price tend to resume its normal drift for a
    short period afterward?

    This is a genuinely different kind of signal from the other two —
    it's reactive to a spike that already happened, not predictive of
    the next one. Whether it has real edge is exactly what the
    backtest needs to answer; this is a hypothesis to test, not a
    strategy to trust.

    Args:
        lookback: Window used to compute the rolling volatility
                  baseline that defines what counts as a "spike".
        spike_std_threshold: How many standard deviations a single-bar
                  move must exceed to be flagged as a spike.
        hold_bars: How many bars to hold the position after entry
                  before force-exiting, regardless of price action.
    """
    returns = df["Close"].pct_change()
    rolling_std = returns.rolling(lookback).std()
    z_score = returns / rolling_std

    spike_detected = z_score.abs() > spike_std_threshold
    entries = spike_detected.fillna(False)

    # Fixed-bar exit: force-close `hold_bars` candles after each entry
    exits = pd.Series(False, index=df.index)
    entry_positions = df.index[entries]
    for entry_time in entry_positions:
        entry_idx = df.index.get_loc(entry_time)
        exit_idx = min(entry_idx + hold_bars, len(df) - 1)
        exits.iloc[exit_idx] = True

    return entries, exits


STRATEGY_REGISTRY = {
    "momentum_ma_crossover": momentum_ma_crossover,
    "rsi_mean_reversion": rsi_mean_reversion,
    "spike_reversion_boom_crash": spike_reversion_boom_crash,
}
