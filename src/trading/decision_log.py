"""Per-candle decision logging for the live books.

Kass's order (2026-10-31): the dashboards must show "everything
the bot is thinking" -- so every live loop now records one line
per processed closed candle to data/decisions/<book_id>.jsonl,
including the quiet HOLD candles: the candle close, the eyes'
P(up) vote, the PPO stance, the final action and whether it was
taken, the reason, the gate's proba/verdict on Book #3 (TAKEN
decisions included, not just vetoes), position before/after,
equity, the effective stop level, exposure, and the exact
feature values the eyes voted on.

INSTRUMENTATION ONLY, the same discipline as grading-time vote
logging: the loops pass values they already computed for the
candle itself -- no extra model runs, no extra market-data
fetches, no recomputation of the decision -- and this module
can never change a trading outcome:

- append_decision swallows EVERY exception (prints a WARNING
  and returns False); a logging failure can never raise into
  a trading path.
- Appends are idempotent per candle: if the file's tail
  already carries a record for the same candle ts, the append
  is skipped, so loop retries / overlapping runs never
  double-log a candle.
- Nothing here is read back by any trading code. The only
  consumer is the master dashboard feed
  (src/dashboards/compact.py, _decision_log_block).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECISIONS_DIR = ROOT / "data" / "decisions"

_TAIL_BYTES = 65536  # bounded tail scan for the dedupe check
_STANCE_NAMES = ["hold", "buy", "sell", "short"]
_ACTION_NAMES = ["HOLD", "BUY", "SELL", "SHORT"]


def _already_logged(path: Path, ts: str) -> bool:
    """True when the file's tail already holds a record for ts.
    The tail window covers far more than a day of candles, so a
    retry of an older candle outside the window is the only
    miss -- loop retries are same-candle and immediate."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > _TAIL_BYTES:
                fh.seek(size - _TAIL_BYTES)
            raw = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            if json.loads(line).get("ts") == ts:
                return True
        except Exception:
            continue  # a partial first line in the window, etc.
    return False


def append_decision(book_id: str, record: dict,
                    decisions_dir=None) -> bool:
    """Append one candle's decision record for a book. Returns
    True when a line was written, False when skipped (duplicate
    candle) or on ANY failure -- never raises."""
    try:
        d = Path(decisions_dir) if decisions_dir else DECISIONS_DIR
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{book_id}.jsonl"
        ts = record.get("ts")
        if ts is not None and path.exists() \
                and _already_logged(path, ts):
            return False
        line = json.dumps(record)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return True
    except Exception as exc:  # instrumentation must never bite
        print(f"WARNING: decision logging failed for {book_id}: "
              f"{exc}", flush=True)
        return False


def describe_outcome(stance: int, stop_fill, fill,
                     gate_verdict: str | None = None):
    """(action, taken, reason) for one candle, from the values
    the loop already produced. `stance` is the PPO action int
    (0 hold / 1 buy / 2 sell / 3 short) the loop dispatched on
    (Book #3: the ORIGINAL stance, before any veto); `fill` is
    the dispatch's broker return (None when nothing filled),
    `stop_fill` the protective stop's return from the pre-dispatch
    check. Naming follows the broker's own: a buy that closes a
    short is a COVER, a sell that opens one is a SHORT, and a
    stop exit settles the candle when the stance itself did
    nothing."""
    stance = int(stance)
    if fill is not None:
        if getattr(fill, "reason", None) == "cover":
            action = "COVER"
        elif stance == 3:
            action = "SHORT"
        elif stance in (1, 2):
            action = _ACTION_NAMES[stance]
        else:
            action = getattr(fill, "side", "HOLD")
        return action, True, getattr(fill, "reason", None)
    if stop_fill is not None:
        return ("COVER" if stop_fill.side == "BUY" else "SELL"), \
            True, stop_fill.reason
    if gate_verdict == "VETOED":
        return _ACTION_NAMES[stance], False, "gate_veto"
    if stance == 0:
        return "HOLD", False, "stance_hold"
    return _ACTION_NAMES[stance], False, "no_fill"


def stance_name(stance: int) -> str:
    """The loop's own stance word for a PPO action int."""
    return _STANCE_NAMES[int(stance)]


def broker_stop_level(broker) -> float | None:
    """The broker's effective stop for the current position
    (its own stop_level() over state the loop already holds);
    None when flat or on any failure."""
    try:
        sl = broker.stop_level()
        return float(sl[0]) if sl else None
    except Exception:
        return None


def make_record(*, ts, close, p_up, stance, action, taken, reason,
                gate_proba=None, gate_threshold=None,
                gate_verdict=None, pos_before=None, pos_after=None,
                equity=None, stop_level=None, exposure_pct=None,
                features=None) -> dict:
    """The uniform decision-record shape every book writes.
    gate_* are null for the ungated books; features is included
    only when the loop holds the eyes' input values (all three
    loops do)."""
    rec = {"ts": ts,
           "close": close,
           "p_up": p_up,
           "stance": stance,
           "action": action,
           "taken": bool(taken),
           "reason": reason,
           "gate_proba": gate_proba,
           "gate_threshold": gate_threshold,
           "gate_verdict": gate_verdict,
           "pos_before": pos_before,
           "pos_after": pos_after,
           "equity": equity,
           "stop_level": stop_level,
           "exposure_pct": exposure_pct}
    if features is not None:
        rec["features"] = features
    return rec
