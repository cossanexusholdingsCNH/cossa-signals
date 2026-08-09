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
- [x] Deriv synthetic indices data fetcher (Volatility 75, Boom 1000/500, Crash 1000/500) — written and syntax-verified; **not yet tested against a live connection**, see note in fetch_deriv_data.py
- [x] Backtesting harness (vectorbt-based, with mandatory stop-loss/take-profit and fee modeling) — tested end-to-end against synthetic sample data, confirmed working
- [x] Deriv synthetic indices data fetcher — `--all` flag now validates a researched candidate list (Volatility 10/25/50/75/100, their 1s variants, Boom/Crash 500/1000, Bull/Bear Market, Step Index) live via a real 2-candle test per symbol, since Deriv's active_symbols endpoint has proven unreliable for this app (returns empty despite ticks_history working fine); logic verified against a mock, live validation results pending
- [x] Pagination added to fetch_deriv_data.py (--days flag) — automatically chains multiple API calls to pull real months of history instead of one 5000-candle batch (~3.5 days); logic verified against a mocked connection, not yet run against live Deriv for a large pull
- [x] Educational resource: docs/Deriv_Synthetic_Indices_Trading_Guide.docx — plain-language guide to synthetic indices, risk management, and strategy frameworks for newcomers
- [x] Baseline strategies: momentum crossover, RSI mean-reversion, spike-reversion (Boom/Crash-specific) — all are untested hypotheses on real data, not proven edges
- [ ] Run backtests against real historical data once fetch_deriv_data.py is validated locally
- [ ] ML signal layer
- [ ] Risk management / position sizing logic (beyond per-trade stop-loss)
- [ ] Telegram delivery bot
- [ ] Dashboard (Lovable + Supabase)
- [ ] Copy-trade execution — **on hold** pending backtest validation, paper trading, and a compliance/FSP licensing check (see project decisions in chat history)

## A note on Deriv synthetic indices (Volatility, Boom, Crash)

These instruments carry a materially different risk profile from forex majors — no real underlying asset, engineered volatility, and in the case of Boom/Crash, deliberate large directional spikes at semi-random intervals. Strategies here (especially `spike_reversion_boom_crash`) are explicitly experimental. Do not treat a good backtest result on these instruments as proof of edge without extended paper-trading validation — short winning streaks on these markets are a known pattern before larger losses, not a reliable signal of skill.

## Ownership

All code, models, and IP in this repository belong to Cossa Nexus Holdings (Pty) Ltd, under Cossa Tech. Private repository — do not share access outside CNH without authorization.
