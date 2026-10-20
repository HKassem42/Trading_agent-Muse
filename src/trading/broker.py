"""Paper broker: simulated BTC/USD spot trading.

Rules (per config):
- Long-only, starting cash $100,000 (raised from $10,000 by Kass 2026-10-20).
- Each buy risks at most `max_position_pct` of current equity.
- 5% stop-loss from entry: the broker auto-sells the full position.
- No take-profit.
- 0.15% fee per side (0.3% round-trip: commission + spread + slippage,
  per published RL-trading cost guidance; keeps backtests honest).

Every fill is recorded in the SQLite ledger.
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
    reason: str        # signal | stop_loss | rebalance


class PaperBroker:
    def __init__(
        self,
        db_path: str | Path,
        starting_balance: float = 100_000.0,
        max_position_pct: float = 0.25,
        stop_loss_pct: float = 0.05,
    ) -> None:
        self.db_path = Path(db_path)
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.cash = float(starting_balance)
        self.btc = 0.0
        self.entry_price: float | None = None
        self._init_db()

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
                    reason TEXT NOT NULL)"""
            )
            con.execute(
                """CREATE TABLE IF NOT EXISTS equity(
                    ts TEXT NOT NULL, equity REAL NOT NULL,
                    cash REAL NOT NULL, btc REAL NOT NULL, price REAL NOT NULL)"""
            )

    def _record(self, fill: Fill) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO trades(ts,side,price,qty,fee,cash_after,btc_after,reason)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (fill.ts, fill.side, fill.price, fill.qty, fill.fee,
                 self.cash, self.btc, fill.reason),
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

    def buy(self, price: float, reason: str = "signal", ts: str | None = None) -> Fill | None:
        """Buy up to max_position_pct of equity. Returns the fill or None."""
        budget = self.equity(price) * self.max_position_pct
        spend = min(budget, self.cash)
        if spend <= 0:
            return None
        qty = spend / price * (1 - FEE_RATE)
        fee = spend * FEE_RATE
        self.cash -= spend
        if self.btc == 0:
            self.entry_price = price
        else:
            # weighted-average entry on adds
            self.entry_price = (
                self.entry_price * self.btc + price * qty
            ) / (self.btc + qty)
        self.btc += qty
        fill = Fill(ts or self._now(), "BUY", price, qty, fee, reason)
        self._record(fill)
        return fill

    def sell(self, price: float, pct: float = 1.0, reason: str = "signal",
             ts: str | None = None) -> Fill | None:
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
        fill = Fill(ts or self._now(), "SELL", price, qty, fee, reason)
        self._record(fill)
        return fill

    def check_stop(self, price: float, ts: str | None = None) -> Fill | None:
        """Enforce the stop-loss. Call on every new price."""
        if self.btc > 0 and self.entry_price and price <= self.entry_price * (1 - self.stop_loss_pct):
            return self.sell(price, pct=1.0, reason="stop_loss", ts=ts)
        return None

    def unrealized_pct(self, price: float) -> float:
        if self.btc == 0 or not self.entry_price:
            return 0.0
        return price / self.entry_price - 1
