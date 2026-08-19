# Cossa Signals

AI-powered forex and shares trading signals platform — built and owned by Cossa Tech, under Cossa Nexus Holdings.

## What this is

A risk-managed, backtested trading signals engine — not a guarantee of outcomes. Every signal ships with a confidence score, stop-loss, and position sizing logic. Nothing here constitutes financial advice.

## Project structure

```
cossa-signals/
├── data/
│   ├── raw/              # untouched pulled data (gitignored)
│   └── processed/        # cleaned/feature-engineered data (gitignored)
├── src/
│   ├── data_pipeline/    # fetch/clean scripts (yfinance, OANDA, Alpha Vantage)
│   ├── backtesting/      # strategy backtest logic (vectorbt)
│   ├── signals/          # signal-generation models
│   └── delivery/         # WhatsApp/Telegram bot integration
├── notebooks/            # exploration only — not production code
├── config/
│   └── .env.example      # template — copy to .env locally, never commit .env
├── tests/
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
cp config/.env.example config/.env   # then fill in your real API keys
```

## Status

- [x] Repo structure and dependencies defined
- [x] Forex data fetcher (yfinance, daily OHLCV — EUR/USD, GBP/USD, USD/ZAR)
- [x] Deriv synthetic indices data fetcher — `--all` flag validated live: 17/17 candidate symbols confirmed tradeable and fetched successfully (Volatility 10/25/50/75/100, all five 1s variants, Boom 500/1000, Crash 500/1000, Bull/Bear Market, Step Index). Jump Indices and Range Break not yet added — exact symbol codes need confirming before including them.
- [x] Pagination added to fetch_deriv_data.py (--days flag) — chains paginated API calls to pull real months of history. Live-pulled 180 days at 5-minute granularity for all 17 instruments — the current working dataset (data/raw/, gitignored).
- [x] Backtesting harness (vectorbt-based) — mandatory stop-loss/take-profit/fee modeling, Sharpe annualization fix, position sizing, buy-and-hold benchmark, train/test split validation (`train_test_split_ohlc`), minimum-trade-count enforcement (`MIN_TRUSTWORTHY_TRADES = 30`) so low-sample results can't be mistaken for edge.
- [x] Batch backtest runner (src/backtesting/run_all_backtests.py) — every applicable strategy against every symbol in one command, instrument-aware routing (Boom/Crash → spike-reversion, RDBULL/RDBEAR → daily-reset-reversion, everything else → momentum/RSI mean-reversion), ranked CSV output.
- [x] **First real finding: RDBULL/RDBEAR daily-reset reversion.** Generic RSI mean-reversion's only real edge across all 17 instruments landed on exactly these two. `check_daily_reset.py` confirmed why: a real ~10x-normal-move price discontinuity at the 00:00 GMT reset (RDBULL gaps down ~8.6%, RDBEAR gaps up ~5.3%). A purpose-built `daily_reset_reversion` strategy trading that window directly passed the 70/30 train/test split (held up on unseen data). Still not validated on live data — see next item.
- [x] Live paper-trading harness (src/delivery/paper_trade.py) — watches RDBULL/RDBEAR live and simulates daily_reset_reversion in real time, no real money touched, state persists across restarts. **Currently blocked**: live-stream connection is rejecting symbol subscriptions; most recent commits are diagnosing an app-authorization gap. Not yet run successfully end-to-end.
- [x] Educational resource: docs/Deriv_Synthetic_Indices_Trading_Guide.docx — plain-language guide to synthetic indices, risk management, and strategy frameworks for newcomers
- [x] Baseline strategies: momentum crossover (tested, lost money 17/17 on 1-min real data — structurally wrong for these instruments, deprioritized), RSI mean-reversion (only edge found on RDBULL/RDBEAR), spike-reversion (Boom/Crash-specific, lost on all 4 tested), daily-reset-reversion (see finding above)
- [x] **web/index.html** — Cossa Signals Agent Console, a live status dashboard reflecting the real state of this table (deployed to Vercel). Update it by hand whenever this checklist changes; it is not yet wired to read live pipeline/paper-trade state automatically.
- [ ] Fix live paper-trading stream authorization, then run a 4–6 week unattended paper-trading window on RDBULL/RDBEAR before any conversation about real capital
- [ ] Re-run momentum/RSI/spike-reversion against the 180-day/5-min dataset on the remaining 15 instruments (not yet done — daily-reset finding took priority)
- [ ] ML signal layer — next after the above, or after ruling out classical strategies on the rest of the 17
- [ ] Risk management / position sizing logic beyond per-trade stop-loss
- [ ] Telegram delivery bot
- [ ] Wire the dashboard to live pipeline/backtest/paper-trade state instead of manual updates
- [ ] Copy-trade execution — **on hold** pending backtest validation, paper trading, and a compliance/FSP licensing check (see project decisions in chat history)

## A note on Deriv synthetic indices (Volatility, Boom, Crash)

These instruments carry a materially different risk profile from forex majors — no real underlying asset, engineered volatility, and in the case of Boom/Crash, deliberate large directional spikes at semi-random intervals. Strategies here (especially `spike_reversion_boom_crash`) are explicitly experimental. Do not treat a good backtest result on these instruments as proof of edge without extended paper-trading validation — short winning streaks on these markets are a known pattern before larger losses, not a reliable signal of skill.

## Ownership

All code, models, and IP in this repository belong to Cossa Nexus Holdings (Pty) Ltd, under Cossa Tech. Private repository — do not share access outside CNH without authorization.
