"""Hourly rollups of the Coinbase Level 2 per-minute aggregates.

Companion to data.l2_collector: the collector writes one row per minute
to l2_minutes in data/l2_book.db; this module folds each completed UTC
hour into one row in l2_hours, aligned to the project's candle
convention (hour-start ts, hour H's row built ONLY from minute rows
inside [H, H+3600) -- causal by construction: the row is final at H's
close, exactly when the hourly trading loops make their decisions).

Honesty rules (see refs/level2-collector-2026-11-18.md):
- Feature means and "end" values are computed over VALID minutes only
  (a minute whose book was dead at sample time carries NULL features;
  it still contributes its true coverage to coverage_mean).
- "end" is the last valid minute of the hour, never a trailing NULL.
- usable = 1 only when minutes_present >= 50 AND coverage_mean >= 55s.
  A thin hour is flagged, never silently trusted.
- An hour with minute rows but no valid minute still gets a rollup row
  (usable=0, NULL means) so "we were down" is distinguishable from
  "no data was ever collected".
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "l2_book.db"

FEATURE_FIELDS = (
    "spread_bps",
    "micro_dev_bps",
    "imb_10bps",
    "imb_50bps",
    "imb_100bps",
    "imb_200bps",
)

MINUTES_PRESENT_MIN = 50
COVERAGE_MEAN_MIN = 55.0

HOUR_SCHEMA = """
CREATE TABLE IF NOT EXISTS l2_hours (
    ts INTEGER PRIMARY KEY,
    spread_bps_mean REAL, spread_bps_end REAL,
    micro_dev_bps_mean REAL, micro_dev_bps_end REAL,
    imb_10bps_mean REAL, imb_10bps_end REAL,
    imb_50bps_mean REAL, imb_50bps_end REAL,
    imb_100bps_mean REAL, imb_100bps_end REAL,
    imb_200bps_mean REAL, imb_200bps_end REAL,
    depth_1pct_mean REAL,
    minutes_present INTEGER NOT NULL,
    coverage_mean REAL,
    usable INTEGER NOT NULL
)
"""

_HOUR_COLUMNS = (
    "ts",
    "spread_bps_mean", "spread_bps_end",
    "micro_dev_bps_mean", "micro_dev_bps_end",
    "imb_10bps_mean", "imb_10bps_end",
    "imb_50bps_mean", "imb_50bps_end",
    "imb_100bps_mean", "imb_100bps_end",
    "imb_200bps_mean", "imb_200bps_end",
    "depth_1pct_mean",
    "minutes_present", "coverage_mean", "usable",
)


def compute_hour_rollup(rows: list[dict], hour_start: int) -> dict:
    """Fold minute rows into hour `hour_start`'s rollup (pure function).

    Rows outside [hour_start, hour_start+3600) are ignored even if
    passed in -- the causality boundary is enforced here, not trusted
    to the caller.
    """
    inside = [r for r in rows if hour_start <= int(r["ts"]) < hour_start + 3600]
    valid = [r for r in inside if r.get("mid") is not None]

    out: dict = {"ts": int(hour_start), "minutes_present": len(valid)}
    for field in FEATURE_FIELDS:
        vals = [float(r[field]) for r in valid if r.get(field) is not None]
        out[f"{field}_mean"] = (sum(vals) / len(vals)) if vals else None
        out[f"{field}_end"] = float(valid[-1][field]) if valid else None
    depths = [float(r["depth_1pct_btc"]) for r in valid
              if r.get("depth_1pct_btc") is not None]
    out["depth_1pct_mean"] = (sum(depths) / len(depths)) if depths else None
    coverages = [float(r["coverage"]) for r in inside
                 if r.get("coverage") is not None]
    out["coverage_mean"] = (sum(coverages) / len(coverages)) if coverages else None
    out["usable"] = int(
        out["minutes_present"] >= MINUTES_PRESENT_MIN
        and out["coverage_mean"] is not None
        and out["coverage_mean"] >= COVERAGE_MEAN_MIN
    )
    return out


def init_hours_table(conn: sqlite3.Connection) -> None:
    conn.execute(HOUR_SCHEMA)
    conn.commit()


def fetch_hour_minutes(conn: sqlite3.Connection, hour_start: int) -> list[dict]:
    cur = conn.execute(
        "SELECT * FROM l2_minutes WHERE ts >= ? AND ts < ? ORDER BY ts",
        (int(hour_start), int(hour_start) + 3600),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def rollup_hour(conn: sqlite3.Connection, hour_start: int) -> dict | None:
    """Compute and store hour `hour_start`'s rollup; None if no minutes."""
    rows = fetch_hour_minutes(conn, hour_start)
    if not rows:
        return None
    rollup = compute_hour_rollup(rows, hour_start)
    placeholders = ", ".join("?" for _ in _HOUR_COLUMNS)
    conn.execute(
        f"INSERT OR REPLACE INTO l2_hours ({', '.join(_HOUR_COLUMNS)}) "
        f"VALUES ({placeholders})",
        [rollup[c] for c in _HOUR_COLUMNS],
    )
    conn.commit()
    return rollup


def backfill_rollups(db_path: Path | str = DEFAULT_DB,
                     now_ts: float | None = None) -> list[int]:
    """Roll up every completed hour that has minutes but no stored row.

    Also re-rolls the newest completed hour unconditionally (its row
    may predate a late minute write). Returns the hour starts rolled.
    """
    now = time.time() if now_ts is None else now_ts
    current_hour = int(now // 3600) * 3600
    conn = sqlite3.connect(str(db_path))
    try:
        init_hours_table(conn)
        minute_hours = [r[0] for r in conn.execute(
            "SELECT DISTINCT ts - (ts % 3600) FROM l2_minutes "
            "WHERE ts < ? ORDER BY 1", (current_hour,))]
        if not minute_hours:
            return []
        done = {r[0] for r in conn.execute("SELECT ts FROM l2_hours")}
        targets = [h for h in minute_hours if h not in done]
        newest = minute_hours[-1]
        if newest not in targets:
            targets.append(newest)
        rolled = []
        for hour in targets:
            if rollup_hour(conn, hour) is not None:
                rolled.append(hour)
        return rolled
    finally:
        conn.close()


if __name__ == "__main__":
    rolled = backfill_rollups()
    print(f"rolled up {len(rolled)} hour(s): {rolled}")
