"""Build JSON snapshots for the two dashboards from the local DBs.

Usage:
    python -m src.dashboards.snapshot trading   # -> trading snapshot JSON
    python -m src.dashboards.snapshot training  # -> training snapshot JSON

The cron worker passes the output to the artifact's push_snapshot action.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ET = ZoneInfo("America/New_York")


def _paper_db():
    p = DATA / "paper.db"
    return sqlite3.connect(p) if p.exists() else None


def closed_trades(con: sqlite3.Connection) -> list[dict]:
    """FIFO-match sells against buys -> closed round-trip trades with realized P&L.

    Each record: {close_ts, qty, pnl}. Fees are folded into cost/proceeds.
    """
    fills = con.execute(
        "SELECT ts, side, price, qty, fee FROM trades ORDER BY id ASC").fetchall()
    lots: list[list[float]] = []  # [qty, cost_incl_fee]
    closed = []
    for ts, side, price, qty, fee in fills:
        if side == "BUY":
            lots.append([qty, price * qty + fee])
        elif side == "SELL" and qty > 0:
            proceeds = price * qty - fee
            remaining = qty
            cost = 0.0
            while remaining > 1e-12 and lots:
                lqty, lcost = lots[0]
                take = min(lqty, remaining)
                cost += lcost * (take / lqty)
                lots[0][0] -= take
                lots[0][1] -= lcost * (take / lqty)
                if lots[0][0] < 1e-12:
                    lots.pop(0)
                remaining -= take
            if remaining > 1e-9:
                continue  # sell with no matching buy; skip
            closed.append({"close_ts": ts, "qty": qty, "pnl": proceeds - cost})
    return closed


def trade_stats(con: sqlite3.Connection) -> dict:
    """Journal stats like the reference: win %, profit factor, avg win/loss, expectancy."""
    closed = closed_trades(con)
    pnls = [t["pnl"] for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    n = len(pnls)
    return {
        "total_trades": n,
        "net_pnl": round(sum(pnls), 2),
        "win_pct": round(100 * len(wins) / n, 2) if n else 0.0,
        # No losses yet: profit factor is undefined (infinite), not 0.0 --
        # 0.0 reads as "worst possible" on the dashboard (audit 2026-10-20).
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else None,
        "avg_win": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "expectancy": round(sum(pnls) / n, 2) if n else 0.0,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
    }


def daily_calendar(con: sqlite3.Connection) -> list[dict]:
    """Per-day closed-trade P&L for the calendar heatmap: [{date, pnl, trades, win_pct}].

    Every date between the first and last closed-trade day is present:
    days with no closed trade get a flat zero cell (audit 2026-10-20: Oct 18
    was silently absent because it had buys but no close, and a missing
    day on the calendar reads as broken data)."""
    closed = closed_trades(con)
    by_day: dict[str, list[float]] = {}
    for t in closed:
        day = datetime.fromisoformat(t["close_ts"]).astimezone(ET).strftime("%Y-%m-%d")
        by_day.setdefault(day, []).append(t["pnl"])
    if not by_day:
        return []
    out = []
    day = datetime.strptime(min(by_day), "%Y-%m-%d").date()
    last = datetime.strptime(max(by_day), "%Y-%m-%d").date()
    while day <= last:
        key = day.strftime("%Y-%m-%d")
        pnls = by_day.get(key, [])
        out.append(
            {"date": key,
             "pnl": round(sum(pnls), 2),
             "trades": len(pnls),
             "win_pct": round(100 * sum(1 for p in pnls if p > 0) / len(pnls), 1) if pnls else 0.0})
        day += timedelta(days=1)
    return out


def drawdown_curve(equity_rows: list[dict]) -> list[dict]:
    """Underwater series: drawdown % below the running equity peak at each
    point. Pairs with the equity curve so risk is visible next to return."""
    out, peak = [], 0.0
    for r in equity_rows:
        peak = max(peak, r["equity"])
        dd = 100.0 * (r["equity"] / peak - 1.0) if peak > 0 else 0.0
        out.append({"t": r["t"], "dd_pct": round(dd, 3)})
    return out


def activity_stats(con: sqlite3.Connection) -> dict:
    """Process metrics: how active the bot has been (a flat stretch should
    read as information, not as a broken dashboard)."""
    now_et = datetime.now(ET)
    today = now_et.strftime("%Y-%m-%d")
    week_ago = (now_et - timedelta(days=7)).strftime("%Y-%m-%d")
    rows = con.execute(
        "SELECT ts, side FROM trades ORDER BY ts ASC").fetchall()
    days = [datetime.fromisoformat(r["ts"]).astimezone(ET).strftime("%Y-%m-%d")
            for r in rows]
    last_trade = rows[-1]["ts"] if rows else None
    days_since = None
    if last_trade:
        last_day = datetime.fromisoformat(last_trade).astimezone(ET).date()
        days_since = (now_et.date() - last_day).days
    # Candles evaluated: every processed candle leaves one equity row, so
    # the bot's "I looked and chose to hold" hours are countable -- a flat
    # stretch should read as decisions made, not silence (viz research).
    eq_rows = con.execute("SELECT ts FROM equity").fetchall()
    eq_days = [datetime.fromisoformat(r["ts"]).astimezone(ET).strftime("%Y-%m-%d")
               for r in eq_rows]
    # Current streak: signed run of same-sign closed trades, newest first
    # (+2 = last two closed trades won, -3 = last three lost, 0 = none yet).
    streak = 0
    pnls = [t["pnl"] for t in closed_trades(con)]
    if pnls:
        sign = 1 if pnls[-1] > 0 else -1
        for p in reversed(pnls):
            if (p > 0) == (sign > 0):
                streak += sign
            else:
                break
    return {
        "trades_today": sum(1 for d in days if d == today),
        "trades_7d": sum(1 for d in days if d >= week_ago),
        "trades_total": len(rows),
        "last_trade_ts": last_trade,
        "days_since_last_trade": days_since,
        "candles_total": len(eq_rows),
        "candles_today": sum(1 for d in eq_days if d == today),
        "streak": streak,
    }


def _fresh_account_snapshot(snap: dict) -> dict:
    """Minimal valid trading snapshot for a fresh paper account: the DB
    exists (tables created) but no equity rows yet because the bot has not
    traded. Returns the full live-shaped payload (FLAT, equity = $100k,
    zeroed stats) so the dashboard can show the new account immediately
    instead of crashing on missing fields. Added 2026-10-20 after the
    $10k -> $100k migration exposed a KeyError in the compact builder."""
    try:
        from data.yfinance_adapter import latest_price
        btc_price = latest_price()
    except Exception:
        btc_price = None
    zero_stats = {
        "total_trades": 0, "net_pnl": 0.0, "win_pct": 0.0,
        "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        "expectancy": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
    }
    snap.update({
        "status": "live",
        "btc_price": btc_price,
        "equity": 100000.0,
        "cash": 100000.0,
        "btc": 0.0,
        "btc_value": 0.0,
        "position": "FLAT",
        "entry_price": None,
        "stop_price": None,
        "unrealized_pct": 0.0,
        "day_pnl": 0.0,
        "day_reward": 0,  # no day yet -- neither +1 nor -1
        "agent_stance": None,
        "stacker_p_up": None,
        "last_candle": None,
        "trade_stats": zero_stats,
        "daily_calendar": [],
        "activity": {
            "trades_today": 0, "trades_7d": 0, "trades_total": 0,
            "last_trade_ts": None, "days_since_last_trade": None,
            "candles_total": 0, "candles_today": 0, "streak": 0,
        },
        "equity_curve": [],
        "drawdown_curve": [],
        "recent_trades": [],
    })
    return snap


def build_trading_snapshot() -> dict:
    snap: dict = {"as_of": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    con = _paper_db()
    if con is None:
        return {**snap, "status": "not_started"}
    con.row_factory = sqlite3.Row

    eq = con.execute(
        "SELECT * FROM equity ORDER BY ts DESC LIMIT 500").fetchall()
    if not eq:
        # Fresh account: DB exists but the bot hasn't traded yet. Return a
        # minimal valid live-shaped snapshot ($100k, FLAT, zeroed stats)
        # instead of a field-less "no_data" payload that crashed the
        # compact builder (2026-10-20).
        return _fresh_account_snapshot(snap)
    latest = eq[0]
    price = latest["price"]
    equity = latest["equity"]
    cash, btc = latest["cash"], latest["btc"]

    # today's PnL: vs the account's equity at the start of the current ET
    # day (the last snapshot at or before ET midnight). Audit 2026-10-20:
    # the old code string-compared naive-UTC equity timestamps against an
    # ET-offset ISO string ("2026-10-20T00:00:00-04:00"), which matched the
    # 01:00 UTC row -- 21:00 ET *yesterday* -- and overstated the day
    # (+$554.59 shown vs +$198.73 true). The ET day start is converted to
    # a UTC string in the same format the ledger stores before comparing.
    day_start_et = datetime.now(ET).replace(hour=0, minute=0, second=0, microsecond=0)
    day_start_utc = day_start_et.astimezone(timezone.utc).isoformat()
    base = con.execute(
        "SELECT equity FROM equity WHERE ts <= ? ORDER BY ts DESC LIMIT 1",
        (day_start_utc,)).fetchone()
    if base is None:
        base = con.execute(
            "SELECT equity FROM equity WHERE ts >= ? ORDER BY ts ASC LIMIT 1",
            (day_start_utc,)).fetchone()
    day_pnl = equity - (base["equity"] if base else 100000.0)

    trades = con.execute(
        "SELECT ts, side, price, qty, fee, reason FROM trades ORDER BY id DESC LIMIT 20"
    ).fetchall()

    state_file = DATA / "live_state.json"
    live = json.loads(state_file.read_text()) if state_file.exists() else {}

    position = "LONG" if btc > 0 else "FLAT"
    entry = live.get("last_entry_price")
    stats = trade_stats(con)
    calendar = daily_calendar(con)
    curve = [
        {"t": r["ts"], "equity": round(r["equity"], 2)} for r in reversed(eq)
    ]
    snap.update({
        "status": "live",
        "btc_price": price,
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "btc": round(btc, 6),
        "btc_value": round(btc * price, 2),
        "position": position,
        "entry_price": entry,
        "stop_price": round(entry * 0.95, 2) if entry else None,
        "unrealized_pct": round(price / entry - 1, 4) if entry else 0.0,
        "day_pnl": round(day_pnl, 2),
        "day_reward": 1 if day_pnl > 0 else -1,
        "agent_stance": live.get("last_stance"),
        "stacker_p_up": live.get("last_p_up"),
        "last_candle": live.get("last_ts"),
        "trade_stats": stats,
        "daily_calendar": calendar,
        "activity": activity_stats(con),
        "equity_curve": curve,
        "drawdown_curve": drawdown_curve(curve),
        "recent_trades": [
            {"ts": r["ts"], "side": r["side"], "price": r["price"],
             "qty": round(r["qty"], 6), "reason": r["reason"]} for r in trades
        ],
    })
    return snap


def build_training_snapshot() -> dict:
    snap: dict = {"as_of": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    tdb = DATA / "training.db"
    runs = []
    if tdb.exists():
        con = sqlite3.connect(tdb)
        con.row_factory = sqlite3.Row
        runs = [
            {"id": r["id"], "ts": r["ts"], "kind": r["kind"],
             "metrics": json.loads(r["metrics"]), "promoted": bool(r["promoted"])}
            for r in con.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 30")
        ]
        con.close()

    champ_file = ROOT / "models" / "stacker_champion.json"
    champion = json.loads(champ_file.read_text()) if champ_file.exists() else None

    # daily +/-1 reward history from paper equity, grouped by ET day
    reward_history = []
    con = _paper_db()
    if con is not None:
        rows = con.execute(
            "SELECT ts, equity FROM equity ORDER BY ts ASC").fetchall()
        by_day: dict[str, list[float]] = {}
        for ts, eq in rows:
            day = datetime.fromisoformat(ts).astimezone(ET).strftime("%Y-%m-%d")
            by_day.setdefault(day, []).append(eq)
        for day in sorted(by_day):
            eqs = by_day[day]
            pnl = eqs[-1] - eqs[0]
            reward_history.append({"date": day, "reward": 1 if pnl > 0 else -1,
                                   "pnl": round(pnl, 2)})
        con.close()

    now_et = datetime.now(ET)
    next_mid = (now_et + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    snap.update({
        "runs": runs,
        "champion": champion,
        "reward_history": reward_history[-60:],
        "next_retrain": next_mid.isoformat(),
        "promotions": [r for r in runs if r["promoted"]][:10],
    })
    return snap


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "trading"
    snap = build_trading_snapshot() if which == "trading" else build_training_snapshot()
    print(json.dumps(snap))
