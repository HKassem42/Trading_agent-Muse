# TradingAgent-Muse — Brainstorm & Architecture

**Date:** 2026-10-19
**Idea:** One mega-project combining every trading bot Kass has built, as a
self-learning BTC/USD paper-trading agent with two live dashboards.

## What I pulled from the existing projects

| Source | What's reusable |
|---|---|
| **Project-Bot** (`Bot1.ipynb` + `best_trading_model.pkl`) | The only actually-trained model: a **stacking classifier** (RandomForest + XGBoost + LightGBM → LogisticRegression) predicting next-5m candle direction on BTCUSDT. Feature recipe: vwap, macd, signal_line, rsi, atr, sma_50, price_change, volatility, volume_change. Paper-trading loop with **$10,000 starting balance**. (Note: trained on Binance.US klines; API keys for that are the ones leaked in the repo.) |
| **Notion "TRADING AGENT" blueprint** | The architecture skeleton: Data Pipeline → Trading Environment (State: market data + indicators + portfolio; Actions: buy/sell/hold; Rewards) → PPO Agent (Policy + Risk Control) → Training → Paper → Live → Monitoring (Sharpe, Max Drawdown, Win Rate, ROI). This becomes our PPO brain. |
| **VisualML** (PLAN.md) | **Champion/challenger pattern**: named training runs + registry, composite scoring, auto-ranking, best-model promotion, leaderboard. This is exactly how the nightly retrain should work — each midnight run is a named experiment; the challenger only replaces the champion if it backtests better. |
| **G-Asisit** | SQLite paper-trade ledger pattern, R/K/C/A/B scoring engine, and the Library-dashboard artifact pattern (magnificent-dashboard know-how). |
| **Trading-League-Game** | Gamified trading web app (Vite + Firebase) — UI/UX inspiration for the dashboards. |
| **Crypto Bot Final Project Plan** (Drive) | 4-pillar coin scoring — parked; single-asset RL doesn't need it yet. |

## Architecture

```
yfinance BTC-USD (5m klines, 24/7, no keys)
        │
        ▼
┌─ Feature Store ─────────────────────────┐
│ Project-Bot recipe: vwap, macd, rsi,    │
│ atr, sma_50, price/volume change, vol   │
│ + portfolio state (cash, position, uPnL) │
└───────────────────┬─────────────────────┘
                    ▼
┌─ Two Brains ────────────────────────────┐
│ 1. PPO agent (Notion blueprint):         │
│    state → policy → buy/sell/hold       │
│ 2. Stacker (Project-Bot, retrainable):  │
│    next-candle direction vote → feeds   │
│    the PPO state as an extra feature    │
└───────────────────┬─────────────────────┘
                    ▼
┌─ Paper Broker ──────────────────────────┐
│ $10k start, long-only, SQLite ledger,   │
│ risk control: max position %, stop-loss │
└───────────────────┬─────────────────────┘
                    ▼
┌─ Reward (±1, exactly as specified) ──────┐
│ Episode = one trading day.              │
│ Day net P&L > 0 → +1, else → -1.        │
│ (Honest note: sparse binary rewards     │
│ learn slowly; shaping is a later lever.)│
└─────────────────────────────────────────┘
```

## Nightly retrain (00:00 ET cron)

1. Pull fresh BTC-USD data, rebuild features.
2. Warm-start PPO from champion weights (continual learning, not from scratch).
3. Train challenger stacker on the extended window.
4. Backtest champion vs challenger on recent holdout (composite score:
   Sharpe + max drawdown + win rate + ROI — the Notion metrics).
5. **Promote only if challenger wins** (VisualML pattern). Log everything as
   a named run in the training registry.
6. Push updates to both dashboards.

## Dashboards (Muse Library artifacts, per the #1-stop rule)

- **Training dashboard:** nightly runs table, champion vs challenger,
  reward history (±1 per day), loss curves, promotion events, hyperparams.
- **Trading dashboard:** live equity curve, open position, today's P&L,
  agent stance (buy/sell/hold + confidence), recent trades, 24/7 BTC price.
- Real-time feel: the paper broker syncs state to the artifacts on a short
  loop (every few minutes) via artifact actions; dashboards render latest.

## Decisions made (user delegated)

- Data: **yfinance BTC-USD** (free, no keys, proven today in G-Asisit).
  Adapter interface keeps Binance/Alpaca pluggable later.
- Paper balance: **$10,000** (matches Project-Bot).
- Reward episode: **one trading day** (aligns with midnight retrain).
- Start long-only; shorts are a later lever.

## Open questions for Kass

1. GitHub repo: I can't create repos (blocked before) — create
   `HKassem42/TradingAgent-Muse` and I'll push, or keep it local for now?
2. Data source OK as yfinance, or do you want Binance/Alpaca wired with keys?
3. Position sizing: fixed fraction per trade (e.g. 25% of equity) vs all-in
   on strong signals?

## Build order

1. Scaffold repo + config (this doc's tree below).
2. Data adapter + feature store (Project-Bot recipe, yfinance).
3. Paper broker + SQLite ledger + risk control.
4. Stacker v2 (retrainable, from the .pkl recipe).
5. PPO environment (state/action/reward per blueprint, ±1 reward).
6. Nightly retrain cron + champion/challenger promotion.
7. Two dashboards (training + trading), real-time sync.
8. Paper-trade live 24/7; watch, tune, repeat.
