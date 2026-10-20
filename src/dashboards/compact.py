"""Compact snapshot builders for dashboard pushes.

The artifact push actions reject large payloads (observed ceiling roughly
1.5-2.5KB depending on the dashboard), so the sync pipeline sends several
small, machine-generated feeds instead of one big snapshot. NEVER hand-type
these payloads -- always generate them here and pass the JSON verbatim.

Feeds (python -m src.dashboards.compact <feed>):
  trading        -> push_snapshot on tradingagent-trading-dashboard.
                    Core stats, position, trade stats, calendar, activity,
                    equity curve. (Drawdown is derived in the frontend from
                    the equity curve: dd = equity / running peak - 1.)
  ledger         -> push_ledger on tradingagent-trading-dashboard.
                    {"ledger": [[ts, side, price, qty, fee], ...] newest
                    first (last 24 fills), "closed": [[close_ts, qty, pnl],
                    ...] last 12 FIFO round trips}
  market0        -> push_market on tradingagent-trading-dashboard.
                    First (older) 48 of the last 96 closed 1h candles:
                    {"start": epoch_s, "step": 3600, "candles": [[o,h,l,c],
                    ...]} with integer prices.
  market1        -> push_market, newest 48 candles, plus
                    {"markers": [[epoch_s, "B"|"S", price, qty], ...],
                     "entry_price": ..., "stop_price": ...}
  vision         -> push_vision on tradingagent-trading-dashboard
                    (added 2026-10-20, features v3 / 1h era).
                    {"ts": iso, "p_up": stacker P(up) 0..1,
                     "f": [[key, label, value], ...]} -- the latest closed
                    candle's 12 feature values (v3 contract: all in
                    [-1, 1]) with a plain-English label per feature, so the
                    dashboard can show what the bot "sees". Keys are the
                    FEATURE_COLUMNS names (rsi_14n renamed to rsi_c in v3).
  training       -> push_snapshot on tradingagent-training-dashboard.
                    Short-key run objects (legend below), stacker champion,
                    reward_series (real PPO mean-reward per run, ascending).
                    NOTE: no live-trading data here -- the old live-derived
                    reward_history was removed as duplication (2026-10-20).

Training run short keys:
  i=id, t=ts, k=kind ("ppo"|"stacker"), p=promoted (0/1),
  s=score (challenger duel score; stacker: challenger accuracy),
  cs=champion score (stacker: champion accuracy),
  pl=challenger holdout P&L, cpl=champion holdout P&L,
  dd=challenger max drawdown, r=challenger mean reward,
  ep=evaluation episodes (sample size), mix=challenger action mix
  {"hold": x, "buy": y, "sell": z} when recorded (PPO runs from
  2026-10-20 onward).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from src.dashboards import snapshot  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _downsample_curve(curve: list[dict]) -> list[dict]:
    """Keep full resolution where anything changed. Only the *leading* flat
    pre-live run at exactly $100,000.0 is collapsed to its first and last
    point (lossless); later rows can pass back through $100,000.0 and must
    not be reclassified as flat. The tail uses an adaptive stride so the
    point count stays bounded as the curve grows."""
    if not curve:
        return curve
    out: list[dict] = []
    i = 0
    n = len(curve)
    while i < n and curve[i].get("equity") == 100000.0:
        i += 1
    if i > 0:
        out.append(curve[0])
        if i > 1:
            out.append(curve[i - 1])
    active = curve[i:]
    m = len(active)
    recent = active[-8:] if m > 8 else active
    tail = active[:-8] if m > 8 else []
    stride = max(1, (len(tail) + 7) // 8)
    for j in range(0, len(tail), stride):
        out.append(tail[j])
    out.extend(recent)
    if out and out[-1] != curve[-1]:
        out.append(curve[-1])
    return [{"t": p["t"], "equity": round(p["equity"], 2)} for p in out]


def build_compact_trading() -> dict:
    s = snapshot.build_trading_snapshot()
    # Belt-and-braces (2026-10-20): a fresh account must never crash the
    # dashboard push with a KeyError on missing fields. Any non-live
    # snapshot (e.g. "not_started" when the DB file is missing entirely)
    # degrades to the minimal valid payload the dashboard expects.
    if s.get("status") != "live":
        fresh = snapshot._fresh_account_snapshot(
            {"as_of": s.get("as_of")})
        s = fresh
    return {
        "as_of": s["as_of"],
        "status": s["status"],
        "btc_price": round(s["btc_price"], 2),
        "equity": s["equity"],
        "cash": s["cash"],
        "btc": s["btc"],
        "btc_value": s["btc_value"],
        "position": s["position"],
        "entry_price": s["entry_price"],
        "stop_price": s["stop_price"],
        "unrealized_pct": s["unrealized_pct"],
        "day_pnl": s["day_pnl"],
        "day_reward": s["day_reward"],
        "agent_stance": s["agent_stance"],
        "stacker_p_up": round(s["stacker_p_up"], 4) if s["stacker_p_up"] is not None else None,
        "last_candle": s["last_candle"],
        "trade_stats": s["trade_stats"],
        "daily_calendar": s["daily_calendar"],
        "activity": s["activity"],
        "equity_curve": _downsample_curve(s["equity_curve"]),
    }


def build_ledger() -> dict:
    con = snapshot._paper_db()
    if con is None:
        return {"ledger": [], "closed": []}
    rows = con.execute(
        "SELECT ts, side, price, qty, fee FROM trades ORDER BY id DESC LIMIT 24"
    ).fetchall()
    ledger = [[r[0], r[1], round(r[2], 2), round(r[3], 6), round(r[4], 2)]
              for r in rows]
    closed = snapshot.closed_trades(con)
    con.close()
    return {
        "ledger": ledger,
        "closed": [[c["close_ts"], round(c["qty"], 6), round(c["pnl"], 2)]
                   for c in closed[-12:]],
    }


def _market_candles():
    """Last 96 closed 1h candles as (start_epoch, [[o,h,l,c]...], df)."""
    from data.yfinance_adapter import fetch_klines
    df = fetch_klines("BTC-USD", interval="1h", period="5d").iloc[:-1]
    df = df.tail(96)
    start = int(df.index[0].timestamp())
    candles = [[int(round(r["open"])), int(round(r["high"])),
                int(round(r["low"])), int(round(r["close"]))]
               for _, r in df.iterrows()]
    return start, candles, df


def build_market(part: int) -> dict:
    start, candles, df = _market_candles()
    half = 48
    if part == 0:
        return {"start": start, "step": 3600, "candles": candles[:half]}
    out = {"start": start + half * 3600, "step": 3600, "candles": candles[half:]}
    # Markers: fills inside the candle window + current entry/stop lines.
    # Fill timestamps in paper.db are UTC wall-clock strings (written from
    # the yfinance index); parse them as UTC explicitly -- a naive
    # .timestamp() would read them as local time and shift markers by the
    # UTC offset (bug caught 2026-10-20: markers landed +4h off the grid).
    from datetime import datetime, timezone
    win_start_epoch = start
    con = snapshot._paper_db()
    markers = []
    entry = stop = None
    if con is not None:
        rows = con.execute(
            "SELECT ts, side, price, qty FROM trades ORDER BY id ASC").fetchall()
        for ts, side, price, qty in rows:
            t = datetime.fromisoformat(ts)
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            ep = int(t.timestamp())
            if ep >= win_start_epoch:
                markers.append([ep, side[0], round(price, 2), round(qty, 6)])
        con.close()
    markers = markers[-20:]
    state_file = ROOT / "data" / "live_state.json"
    if state_file.exists():
        live = json.loads(state_file.read_text())
        entry = live.get("last_entry_price")
        stop = round(entry * 0.95, 2) if entry else None
    out.update({"markers": markers, "entry_price": entry, "stop_price": stop})
    return out


# Plain-English label per feature for the vision feed ("what the bot sees").
FEATURE_LABELS = {
    "body_pct": "Candle body: close vs open this hour",
    "upper_wick": "Upper wick: sellers pushed price down from the high",
    "lower_wick": "Lower wick: buyers pushed price up from the low",
    "range_pct": "Candle range: high-to-low size this hour",
    "rsi_c": "RSI momentum: -1 oversold, +1 overbought",
    "macd_hist_rel": "MACD histogram: trend momentum",
    "dist_vwap": "Distance from VWAP (average traded price)",
    "dist_sma50": "Distance from the 50-hour average price",
    "atr_rel": "ATR: typical hourly move size (volatility)",
    "price_change": "Last hour's price change",
    "volatility_21": "Volatility over the last 21 hours",
    "volume_change": "Trading volume change vs previous hour",
}


def build_vision() -> dict:
    """What the bot sees on the latest closed 1h candle: the 12 features
    (v3 contract, all in [-1, 1]) + the stacker champion's P(up)."""
    import pandas as pd

    from data.yfinance_adapter import fetch_klines
    from features.technical_indicators import FEATURE_COLUMNS, add_features
    from models.stacker import load as load_stacker

    full = fetch_klines("BTC-USD", interval="1h", period="60d").iloc[:-1]
    feat = add_features(full)
    feat = feat.dropna(subset=FEATURE_COLUMNS)
    ts, row = feat.index[-1], feat.iloc[-1]
    p_up = None
    champ = ROOT / "models" / "stacker_champion.pkl"
    if champ.exists():
        stacker = load_stacker(champ)
        p_up = round(float(stacker.predict_proba(
            row[FEATURE_COLUMNS].to_frame().T)[0, 1]), 4)
    return {
        "ts": pd.Timestamp(ts).isoformat(),
        "p_up": p_up,
        "f": [[c, FEATURE_LABELS[c], round(float(row[c]), 3)]
              for c in FEATURE_COLUMNS],
    }


def _duel_run(r: dict) -> dict:
    m = r.get("metrics") or {}
    out = {"i": r["id"], "t": r["ts"], "k": r["kind"], "p": int(r["promoted"])}
    if r["kind"] == "ppo":
        c, h = m.get("challenger") or {}, m.get("champion") or {}
        if isinstance(c.get("score"), (int, float)):
            out["s"] = round(c["score"], 1)
        if isinstance(h.get("score"), (int, float)):
            out["cs"] = round(h["score"], 1)
        if isinstance(c.get("total_pnl"), (int, float)):
            out["pl"] = round(c["total_pnl"], 2)
        if isinstance(h.get("total_pnl"), (int, float)):
            out["cpl"] = round(h["total_pnl"], 2)
        if isinstance(c.get("max_drawdown"), (int, float)):
            out["dd"] = round(c["max_drawdown"], 2)
        if isinstance(c.get("mean_reward"), (int, float)):
            out["r"] = round(c["mean_reward"], 3)
        if c.get("episodes"):
            out["ep"] = c["episodes"]
        if c.get("action_mix"):
            out["mix"] = c["action_mix"]
    else:  # stacker
        if isinstance(m.get("challenger_acc"), (int, float)):
            out["s"] = round(m["challenger_acc"], 4)
        if isinstance(m.get("champion_acc"), (int, float)):
            out["cs"] = round(m["champion_acc"], 4)
    return out


def build_compact_training() -> dict:
    s = snapshot.build_training_snapshot()
    runs = [_duel_run(r) for r in s["runs"][:6]]
    ch = s.get("champion") or {}
    reward_series = [
        {"t": r["ts"], "r": round((r["metrics"]["challenger"]["mean_reward"]), 3)}
        for r in reversed(s["runs"])
        if r["kind"] == "ppo"
        and isinstance((r.get("metrics") or {}).get("challenger", {}).get("mean_reward"), (int, float))
    ]
    return {
        "as_of": s["as_of"],
        "runs": runs,
        "champion": {
            "accuracy": round(ch["accuracy"], 4),
            "precision": round(ch["precision"], 4),
            "folds": ch.get("folds"),
        } if ch else None,
        "reward_series": reward_series,
        "next_retrain": s["next_retrain"],
    }


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "trading"
    builders = {
        "trading": build_compact_trading,
        "ledger": build_ledger,
        "market0": lambda: build_market(0),
        "market1": lambda: build_market(1),
        "vision": build_vision,
        "training": build_compact_training,
    }
    out = builders[which]()
    text = json.dumps(out, separators=(",", ":"))
    sys.stderr.write(f"bytes={len(text)}\n")
    print(text)
