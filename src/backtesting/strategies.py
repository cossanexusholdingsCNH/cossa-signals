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


def daily_reset_reversion(
    df: pd.DataFrame,
    direction: str = "post_reset_long",
    entry_window_min: int = 30,
    exit_window_min: int = 30,
) -> tuple[pd.Series, pd.Series]:
    """
    Purpose-built for RDBULL/RDBEAR's confirmed daily reset behavior —
    NOT a generic technical indicator applied blind.

    Deriv classifies these as "Daily Reset Indices": price resets to a
    baseline (~1000) at 00:00 GMT every day. Direct measurement against
    the fetched 180-day/5-min data confirmed this is a real, large
    discontinuity — the overnight gap (prev close -> next open) has
    ~10x the standard deviation of a normal candle-to-candle move, with
    RDBULL gapping DOWN ~8.6% on average at reset and RDBEAR gapping UP
    ~5.3% on average. That's the opposite of noise; it's a scheduled,
    designed event.

    This came out of the batch backtest: generic rsi_mean_reversion
    showed a real edge on exactly these two symbols and nowhere else
    among the 17 instruments tested — which lines up precisely with
    "daily reset" vs "continuous", too cleanly to be coincidence. This
    strategy tests the more honest, direct hypothesis: trade the known
    reset window itself, rather than relying on RSI to stumble into it.

    Args:
        direction: "post_reset_long" — buy shortly after the reset,
                   sell shortly before the next one. Fits RDBULL, which
                   climbs from baseline intraday then drops back at
                   reset: ride the climb, exit before the drop.
                   "pre_reset_long" — buy shortly before the reset,
                   sell shortly after the next one. Fits RDBEAR, which
                   falls from baseline intraday then jumps back at
                   reset: buy right before the bounce, sell once it's
                   landed.
        entry_window_min: Minutes from the reset boundary (00:00 GMT)
                   during which an entry signal can fire.
        exit_window_min: Minutes from the reset boundary during which
                   an exit signal can fire.
    """
    minutes_since_midnight = df.index.hour * 60 + df.index.minute

    if direction == "post_reset_long":
        entries = minutes_since_midnight < entry_window_min
        exits = minutes_since_midnight >= (1440 - exit_window_min)
    elif direction == "pre_reset_long":
        entries = minutes_since_midnight >= (1440 - entry_window_min)
        exits = minutes_since_midnight < exit_window_min
    else:
        raise ValueError(f"Unknown direction '{direction}'. Use 'post_reset_long' or 'pre_reset_long'.")

    entries = pd.Series(entries, index=df.index)
    exits = pd.Series(exits, index=df.index)

    # Only the first entry signal and first exit signal within each
    # window should fire — otherwise every candle in a 30-minute window
    # generates a redundant re-entry/re-exit signal. Note: .shift(1) on a
    # bool Series inserts NaN at the start, upcasting the whole series to
    # object dtype — ~ on that is bitwise NOT on Python bools (~False ==
    # -1, which is truthy), silently breaking the dedup. .astype(bool)
    # after .fillna(False) avoids that trap.
    entries = entries & ~(entries.shift(1).fillna(False).astype(bool))
    exits = exits & ~(exits.shift(1).fillna(False).astype(bool))

    return entries, exits


STRATEGY_REGISTRY = {
    "momentum_ma_crossover": momentum_ma_crossover,
    "rsi_mean_reversion": rsi_mean_reversion,
    "spike_reversion_boom_crash": spike_reversion_boom_crash,
    "daily_reset_reversion": daily_reset_reversion,
    # Pre-bound direction variants so the batch runner can route by symbol
    # without needing a kwargs-passing mechanism it doesn't currently have.
    "daily_reset_reversion_bull": lambda df, **kw: daily_reset_reversion(df, direction="post_reset_long", **kw),
    "daily_reset_reversion_bear": lambda df, **kw: daily_reset_reversion(df, direction="pre_reset_long", **kw),
}
