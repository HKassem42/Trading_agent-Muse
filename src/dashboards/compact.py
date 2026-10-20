"""Compact snapshot builder for dashboard pushes.

Full snapshots from src.dashboards.snapshot are tens of KB (500-point
equity/drawdown curves, full training-run metrics), but the artifact
push_snapshot actions reject payloads over ~1.5-1.7KB. This keeps the
informative fields and downsamples the rest losslessly where possible.

Usage:
    python -m src.dashboards.compact trading   -> compact trading JSON
    python -m src.dashboards.compact training  -> compact training JSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from src.dashboards import snapshot  # noqa: E402


def _downsample_curve(curve: list[dict]) -> list[dict]:
    """Keep full resolution where anything changed. Only the *leading* flat
    pre-live run at exactly $10,000.0 is collapsed to its first and last
    point (it is constant, so this is lossless); everything after the first
    real movement is kept in chronological order, because later rows can
    pass back through $10,000.0 and must not be reclassified as flat."""
    if not curve:
        return curve
    out: list[dict] = []
    i = 0
    n = len(curve)
    # leading flat run
    while i < n and curve[i].get("equity") == 10000.0:
        i += 1
    if i > 0:
        out.append(curve[0])
        if i > 1:
            out.append(curve[i - 1])
    active = curve[i:]
    # active portion: full resolution for the last 8 candles, hourly before
    # that (the shape of the day is what matters, not every 15m wiggle).
    m = len(active)
    for j, pt in enumerate(active):
        if j >= m - 8 or j % 4 == 0:
            out.append(pt)
    # always keep the last point
    if out and out[-1] != curve[-1]:
        out.append(curve[-1])
    return [{"t": p["t"], "equity": round(p["equity"], 2)} for p in out]


def _drawdown(curve: list[dict]) -> list[dict]:
    out, peak = [], 0.0
    for r in curve:
        peak = max(peak, r["equity"])
        dd = 100.0 * (r["equity"] / peak - 1.0) if peak > 0 else 0.0
        out.append({"t": r["t"], "dd_pct": round(dd, 3)})
    return out


def build_compact_trading() -> dict:
    s = snapshot.build_trading_snapshot()
    if s.get("status") not in ("live", "no_data"):
        return s
    curve = _downsample_curve(s["equity_curve"])
    trades = (s.get("recent_trades") or [])[:5]
    compact = {
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
        "equity_curve": curve,
        "drawdown_curve": _drawdown(curve),
        "recent_trades": trades,
    }
    return compact


def _run_score(m: dict, kind: str):
    if kind == "ppo":
        c = m.get("challenger") or {}
        sc = c.get("score")
        return round(sc, 1) if isinstance(sc, (int, float)) else None
    if kind == "stacker":
        acc = m.get("challenger_acc")
        return round(acc, 4) if isinstance(acc, (int, float)) else None
    return None


def build_compact_training() -> dict:
    s = snapshot.build_training_snapshot()
    slim = [
        {
            "id": r["id"],
            "ts": r["ts"],
            "kind": r["kind"],
            "promoted": r["promoted"],
            "score": _run_score(r.get("metrics") or {}, r["kind"]),
        }
        for r in s["runs"]
    ]
    ch = s.get("champion") or {}
    compact = {
        "as_of": s["as_of"],
        "runs": slim,
        "champion": {
            "accuracy": round(ch["accuracy"], 4),
            "precision": round(ch["precision"], 4),
            "folds": ch.get("folds"),
        } if ch else None,
        "reward_history": s["reward_history"],
        "next_retrain": s["next_retrain"],
        "promotions": [r for r in slim if r["promoted"]][:10],
    }
    return compact


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "trading"
    compact = build_compact_trading() if which == "trading" else build_compact_training()
    text = json.dumps(compact, separators=(",", ":"))
    sys.stderr.write(f"bytes={len(text)}\n")
    print(text)
