# TradingAgent-Muse

A self-learning BTC/USD paper-trading agent. One mega-project combining:
- **Project-Bot** — stacking classifier (RF + XGB + LGBM → LR) on 5m BTC klines
- **Notion "TRADING AGENT" blueprint** — PPO agent architecture
- **VisualML** — champion/challenger training registry + model promotion
- **G-Asisit** — paper-trade ledger + scoring engine + dashboard craft

## How it works

- Trades **BTC/USD** (24/7) in **paper mode only**.
- **Two brains:** a PPO reinforcement-learning agent and a retrainable stacking
  classifier whose vote feeds the PPO state.
- **Learning loop:** retrains every night at 00:00 ET. Each run is a named
  experiment; the challenger only replaces the champion if it backtests better.
- **Reward:** +1 for a profitable paper-trading day, −1 for a losing one.
- **Dashboards:** two Library dashboards — one for training, one for live trading.

## Layout

```
config/          config.yaml — symbols, windows, risk, retrain schedule
src/data/        market data adapters (yfinance now; Binance/Alpaca later)
src/features/    technical indicators + price features (Project-Bot recipe)
src/environment/ state_space, action_space, reward_function (±1)
src/models/      ppo_agent, policy/value networks, stacker
src/training/    nightly retrain, champion/challenger promotion, registry
src/trading/     paper broker, risk control, SQLite ledger
src/dashboards/  sync scripts for the two artifacts
data/ models/ logs/   gitignored runtime state
```

## Status

🚧 Brainstormed 2026-10-19. Build order is in `BRAINSTORM.md`.
