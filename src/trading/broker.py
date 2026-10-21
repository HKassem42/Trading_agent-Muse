"""Paper broker: simulated BTC/USD spot trading.

Rules (per config):
- Long-only, starting cash $100,000 (raised from $10,000 by Kass 2026-10-20).
- Each buy risks at most `max_position_pct` of current equity, AND total
  exposure is capped at `max_exposure_pct` of equity (v4, 2026-10-20:
  the old per-buy cap alone let the brain pyramid four tranches into
  ~100% invested with $125 cash left).
- Exit rules v4 (Kass delegated the design 2026-10-20, "whatever you
  think will get us the best results... we will tweek it from there"):
  * Initial stop: volatility-sized -- 2.0 x ATR% at entry, floored at
    2.5% and capped at 8% (falls back to the flat `stop_loss_pct` when no
    ATR is supplied). A flat 5% is a mile away in calm markets and a
    speed bump in wild ones; the stop now breathes with the market.
  * Trailing take-profit: once the position is up +3% from entry, a
    trailing stop arms at 1.5 x ATR% (floored 2%, capped 6%) below the
    highest close since entry. Winners run until momentum actually
    breaks, then gains are locked instead of round-tripped. The
    effective stop is whichever of the two levels is higher; trailing
    exits are recorded with reason "trailing_stop".
- 0.15% fee per side (0.3% round-trip: commission + spread + slippage,
  per published RL-trading cost guidance; keeps backtests honest).

Every fill is recorded in the SQLite ledger, with an optional context
JSON (the feature snapshot the decision was made on -- per-trade "why").
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

FEE_RATE = 0.0015  # 0.15% per side


@dataclass
class Fill:
    ts: str
    side: str          # BUY / SELL
    price: float
    qty: float
    fee: float
    reason: str        # signal | stop_loss | trailing_stop | rebalance
    context: str | None = None  # JSON feature snapshot at decision time


class PaperBroker:
    def __init__(
        self,
        db_path: str | Path,
        starting_balance: float = 100_000.0,
        max_position_pct: float = 0.25,
        stop_loss_pct: float = 0.05,
        max_exposure_pct: float = 0.75,
    ) -> None:
        self.db_path = Path(db_path)
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_exposure_pct = max_exposure_pct
        self.starting_balance = float(starting_balance)
        self.cash = float(starting_balance)
        self.btc = 0.0
        self.entry_price: float | None = None
        # v4 exit-rule state: ATR% captured at position entry (sizes the
        # stop), and the highest close seen while holding (trailing arm).
        self.entry_atr_pct: float | None = None
        self.high_water: float | None = None
        self._init_db()

    def reset_state(self) -> None:
        """Back to a flat, full-cash account (ledger rows are kept).

        Used by the training env so every episode starts from the same
        clean state instead of inheriting the previous episode's position
        and equity (audit 2026-10-20: episodes shared one broker and
        'teleported' positions across random dates)."""
        self.cash = self.starting_balance
        self.btc = 0.0
        self.entry_price = None
        self.entry_atr_pct = None
        self.high_water = None

    # -- ledger ---------------------------------------------------------
    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS trades(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL, side TEXT NOT NULL, price REAL NOT NULL,
                    qty REAL NOT NULL, fee REAL NOT NULL,
                    cash_after REAL NOT NULL, btc_after REAL NOT NULL,
                    reason TEXT NOT NULL, context TEXT)"""
            )
            # v4: older ledgers lack the context column -- add it in place.
            cols = [r[1] for r in con.execute("PRAGMA table_info(trades)")]
            if cols and "context" not in cols:
                con.execute("ALTER TABLE trades ADD COLUMN context TEXT")
            con.execute(
                """CREATE TABLE IF NOT EXISTS equity(
                    ts TEXT NOT NULL, equity REAL NOT NULL,
                    cash REAL NOT NULL, btc REAL NOT NULL, price REAL NOT NULL)"""
            )

    def _record(self, fill: Fill) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO trades(ts,side,price,qty,fee,cash_after,btc_after,reason,context)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (fill.ts, fill.side, fill.price, fill.qty, fill.fee,
                 self.cash, self.btc, fill.reason, fill.context),
            )

    def snapshot(self, ts: str, price: float) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO equity(ts,equity,cash,btc,price) VALUES (?,?,?,?,?)",
                (ts, self.equity(price), self.cash, self.btc, price),
            )

    # -- trading --------------------------------------------------------
    def equity(self, price: float) -> float:
        return self.cash + self.btc * price

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def buy(self, price: float, reason: str = "signal", ts: str | None = None,
            atr_pct: float | None = None,
            context: str | None = None) -> Fill | None:
        """Buy up to max_position_pct of equity, subject to the total
        exposure cap (v4). Returns the fill or None."""
        budget = self.equity(price) * self.max_position_pct
        headroom = self.equity(price) * self.max_exposure_pct - self.btc * price
        spend = min(budget, self.cash, max(headroom, 0.0))
        if spend < 10.0:
            # Minimum ticket: at the exposure cap the residual headroom is
            # dust (a few dollars); filling it would spam the ledger with
            # meaningless micro-fills (caught by tests 2026-10-20).
            return None
        qty = spend / price * (1 - FEE_RATE)
        fee = spend * FEE_RATE
        self.cash -= spend
        if self.btc == 0:
            self.entry_price = price
            self.entry_atr_pct = atr_pct
            self.high_water = price
        else:
            # weighted-average entry on adds
            self.entry_price = (
                self.entry_price * self.btc + price * qty
            ) / (self.btc + qty)
            self.high_water = max(self.high_water or price, price)
        self.btc += qty
        fill = Fill(ts or self._now(), "BUY", price, qty, fee, reason, context)
        self._record(fill)
        return fill

    def sell(self, price: float, pct: float = 1.0, reason: str = "signal",
             ts: str | None = None,
             context: str | None = None) -> Fill | None:
        """Sell pct of the position (default all)."""
        qty = self.btc * pct
        if qty <= 0:
            return None
        proceeds = qty * price
        fee = proceeds * FEE_RATE
        self.cash += proceeds - fee
        self.btc -= qty
        if self.btc < 1e-10:
            self.btc = 0.0
            self.entry_price = None
            self.entry_atr_pct = None
            self.high_water = None
        fill = Fill(ts or self._now(), "SELL", price, qty, fee, reason, context)
        self._record(fill)
        return fill

    # -- v4 exit rules (see module docstring) --
    STOP_ATR_MULT = 2.0
    STOP_MIN_PCT = 0.025
    STOP_MAX_PCT = 0.08
    TRAIL_ACTIVATE_PCT = 0.03
    TRAIL_ATR_MULT = 1.5
    TRAIL_MIN_PCT = 0.02
    TRAIL_MAX_PCT = 0.06

    def stop_level(self) -> tuple[float, bool] | None:
        """(effective stop price, trailing_active) or None when flat.

        The effective stop is the higher of the ATR-sized initial stop
        and -- once the position is up TRAIL_ACTIVATE_PCT from entry --
        the trailing stop under the highest close since entry."""
        if self.btc <= 0 or not self.entry_price:
            return None
        atr = self.entry_atr_pct
        if atr:
            base_pct = min(max(self.STOP_ATR_MULT * atr,
                               self.STOP_MIN_PCT), self.STOP_MAX_PCT)
        else:
            base_pct = self.stop_loss_pct
        level = self.entry_price * (1 - base_pct)
        trailing = False
        hw = self.high_water
        if hw is not None and hw >= self.entry_price * (1 + self.TRAIL_ACTIVATE_PCT):
            if atr:
                trail_pct = min(max(self.TRAIL_ATR_MULT * atr,
                                    self.TRAIL_MIN_PCT), self.TRAIL_MAX_PCT)
            else:
                trail_pct = 0.04
            trail_level = hw * (1 - trail_pct)
            if trail_level > level:
                level, trailing = trail_level, True
        return level, trailing

    def check_stop(self, price: float, ts: str | None = None,
                   low: float | None = None,
                   context: str | None = None) -> Fill | None:
        """Enforce the exit rules. Call on every new candle.

        Trigger (audit 2026-10-20): evaluated against the candle LOW when
        one is supplied -- a wick through the stop is a stop-out, even if
        the candle later recovers to close above it. Fill price: the stop
        level itself when the close held at/above it (the intra-candle
        touch), otherwise the close (the market gapped through and never
        came back). Without a low, legacy close-only behavior."""
        if self.btc <= 0 or not self.entry_price:
            return None
        if self.high_water is None or price > self.high_water:
            self.high_water = price
        lvl = self.stop_level()
        if lvl is None:
            return None
        level, trailing = lvl
        trigger = low if low is not None else price
        if trigger <= level:
            fill_price = level if price >= level else price
            reason = "trailing_stop" if trailing else "stop_loss"
            return self.sell(fill_price, pct=1.0, reason=reason, ts=ts,
                             context=context)
        return None

    def unrealized_pct(self, price: float) -> float:
        if self.btc == 0 or not self.entry_price:
            return 0.0
        return price / self.entry_price - 1
