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


def build_trading_snapshot() -> dict:
    snap: dict = {"as_of": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    con = _paper_db()
    if con is None:
        return {**snap, "status": "not_started"}
    con.row_factory = sqlite3.Row

    eq = con.execute(
        "SELECT * FROM equity ORDER BY ts DESC LIMIT 500").fetchall()
    if not eq:
        return {**snap, "status": "no_data"}
    latest = eq[0]
    price = latest["price"]
    equity = latest["equity"]
    cash, btc = latest["cash"], latest["btc"]

    # today's PnL: vs first equity snapshot of the current ET day
    day_start = datetime.now(ET).replace(hour=0, minute=0, second=0, microsecond=0)
    day_rows = con.execute(
        "SELECT equity FROM equity WHERE ts >= ? ORDER BY ts ASC LIMIT 1",
        (day_start.isoformat(),)).fetchone()
    day_pnl = equity - (day_rows["equity"] if day_rows else 10000.0)

    trades = con.execute(
        "SELECT ts, side, price, qty, fee, reason FROM trades ORDER BY id DESC LIMIT 20"
    ).fetchall()

    state_file = DATA / "live_state.json"
    live = json.loads(state_file.read_text()) if state_file.exists() else {}

    position = "LONG" if btc > 0 else "FLAT"
    entry = live.get("last_entry_price")
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
        "equity_curve": [
            {"t": r["ts"], "equity": round(r["equity"], 2)} for r in reversed(eq)
        ],
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
