"""Two-angle verification of the long Coinbase history pull
(Part A, 2026-10-22 research build).

Angle 1 (internal): row counts, span, duplicate timestamps, and
gap accounting for both assembled files (expected daily count from
Coinbase's first BTC/USD candle 2015-07-20; hourly expected count
from the requested span).

Angle 2 (external cross-check): resample the v1 matrix's hourly
candles to UTC days and compare against the pulled DAILY candles
over the overlap — closes must match to the cent for the vast
majority of fully covered days; open/high/low compared too (the
matrix source frame has two known 5-candle gaps, so a handful of
days may differ in high/low while closes still agree).

Prints a JSON summary; numbers go into
refs/streaks-long-2026-10-22.md.

Usage: .venv/bin/python scripts/verify_history.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "history"
MATRIX_CSV = ROOT / "data" / "timeseries_matrix" / "btc_usd_1h_matrix.csv"


def gap_report(df: pd.DataFrame, step: pd.Timedelta) -> dict:
    idx = df.index
    diffs = idx.to_series().diff()
    gaps = diffs[diffs > step]
    missing = int(sum((g / step) - 1 for g in gaps))
    return {"rows": int(len(df)), "first": str(idx[0]),
            "last": str(idx[-1]),
            "duplicates": int(idx.duplicated().sum()),
            "n_gaps": int(len(gaps)), "missing_candles": missing,
            "largest_gap": str(gaps.max()) if len(gaps) else None,
            "gap_list": [(str(i), str(g)) for i, g in
                         list(gaps.items())[:12]]}


def main() -> int:
    daily = pd.read_csv(HIST / "btc_usd_1d_long.csv",
                        parse_dates=["Date"]).set_index("Date").sort_index()
    hourly = pd.read_csv(HIST / "btc_usd_1h_long.csv",
                         parse_dates=["Date"]).set_index("Date").sort_index()
    out: dict = {
        "daily": gap_report(daily, pd.Timedelta(days=1)),
        "hourly": gap_report(hourly, pd.Timedelta(hours=1)),
    }
    # expected counts
    span_d = (daily.index[-1] - daily.index[0]).days + 1
    out["daily"]["expected_if_complete"] = int(span_d)
    span_h = int((hourly.index[-1] - hourly.index[0])
                 / pd.Timedelta(hours=1)) + 1
    out["hourly"]["expected_if_complete"] = span_h

    # ---- angle 2: matrix hourly -> daily vs pulled daily ----
    m = pd.read_csv(MATRIX_CSV, parse_dates=["Date"])
    m = m.set_index("Date").sort_index()
    md = m.resample("1D").agg({"Open": "first", "High": "max",
                               "Low": "min", "Close": "last"})
    counts = m.resample("1D").size()
    full_days = counts[counts == 24].index
    md = md.loc[full_days]
    j = md.join(daily, lsuffix="_matrix", rsuffix="_pull", how="inner")
    cmp_out = {"overlap_full_days": int(len(j))}
    for col in ["Open", "High", "Low", "Close"]:
        a = j[f"{col}_matrix"].to_numpy(float)
        b = j[f"{col}_pull"].to_numpy(float)
        exact = np.abs(a - b) <= 0.005
        cmp_out[col] = {
            "match_to_cent": int(exact.sum()),
            "match_pct": float(100 * exact.mean()),
            "max_abs_diff": float(np.abs(a - b).max()),
            "mismatch_days": [str(d) for d in
                              j.index[~exact][:10]],
        }
    out["matrix_overlap"] = cmp_out
    print(json.dumps(out, indent=1))
    (HIST / "verify_summary.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
