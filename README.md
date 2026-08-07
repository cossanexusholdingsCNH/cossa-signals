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
- [ ] Backtesting harness
- [ ] Baseline technical strategies (momentum, mean-reversion)
- [ ] ML signal layer
- [ ] Risk management / position sizing logic
- [ ] Telegram delivery bot
- [ ] Dashboard (Lovable + Supabase)

## Ownership

All code, models, and IP in this repository belong to Cossa Nexus Holdings (Pty) Ltd, under Cossa Tech. Private repository — do not share access outside CNH without authorization.
