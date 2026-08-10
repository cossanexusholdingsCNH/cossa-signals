"""
One-off diagnostic — NOT part of the main pipeline, just a check.

Deriv classifies RDBULL/RDBEAR as "Daily Reset Indices" — they reset to a
baseline at 00:00 GMT each day. If that reset shows up as a real price
discontinuity in our fetched data, the RSI mean-reversion "edge" we found
may be partly or entirely the strategy catching that scheduled reset
rather than genuine market inefficiency. This checks directly against the
data instead of guessing.

Usage: python check_daily_reset.py
(run from the repo root, needs data/raw/RDBULL.csv and RDBEAR.csv)
"""
import pandas as pd

for symbol in ["RDBULL", "RDBEAR"]:
    df = pd.read_csv(f"data/raw/{symbol}.csv", index_col=0, parse_dates=True)
    df.index = df.index.tz_localize(None) if df.index.tz else df.index

    # Find the candle closest to each day's 00:00 GMT boundary
    df["date"] = df.index.date
    daily_first = df.groupby("date").first()  # first candle of each day (closest to 00:00 GMT)
    daily_last = df.groupby("date").last()     # last candle of each day (closest to 23:55 GMT)

    # If there's a real reset, the FIRST candle's Open each day should cluster
    # around a consistent value, and the jump from previous day's last Close
    # to next day's first Open should be large and directionally consistent
    # (not just normal candle-to-candle noise).
    prev_close = daily_last["Close"].shift(1)
    next_open = daily_first["Open"]
    overnight_gap_pct = ((next_open - prev_close) / prev_close * 100).dropna()

    # Normal intraday candle-to-candle move, for comparison
    intraday_move_pct = (df["Close"].pct_change() * 100).dropna()

    print(f"\n=== {symbol} ===")
    print(f"Daily open value range: {daily_first['Open'].min():.4f} to {daily_first['Open'].max():.4f}")
    print(f"  (if this range is tight relative to the instrument's overall price range, "
          f"that's evidence of a real baseline reset)")
    print(f"Overnight gap (prev close -> next day open), mean: {overnight_gap_pct.mean():.4f}%, "
          f"std: {overnight_gap_pct.std():.4f}%, max abs: {overnight_gap_pct.abs().max():.4f}%")
    print(f"Typical single-candle move, mean abs: {intraday_move_pct.abs().mean():.4f}%, "
          f"std: {intraday_move_pct.std():.4f}%")
    print(f"Overnight gap is {overnight_gap_pct.std() / intraday_move_pct.std():.1f}x the size "
          f"of a normal candle-to-candle move (std ratio)")
    print(f"  (if this ratio is well above 1, midnight is NOT a normal candle — "
          f"something structurally different happens there)")
