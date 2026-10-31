"""Build the v2 Time-Series ML Training Data Matrix: v1's exact
rows / labels / columns IN ORDER, with the main-chart indicator
columns (src/features/main_chart.py) appended.

Kass green-lit 2026-10-22 (Part B): "find and include all the main
chart indicators we can also include into our data table that are
not in there right now."

Method:
- v1 columns are copied from data/timeseries_matrix/
  btc_usd_1h_matrix.csv, never recomputed.
- New columns are computed on the LONG hourly Coinbase frame
  (data/history/btc_usd_1h_long.csv, 2022-01-01 -> 2026-10-21,
  pulled for Part A) so every warmup (SMA/EMA 200, Ichimoku spans,
  fractal lookback, prior-day stamps) is fully covered by real
  history long before the first matrix row (2024-11-23) — v2 keeps
  all 16,735 v1 rows with no NaN in the new columns. If the long
  frame is missing, the build falls back to the v1 source frame
  (sup.pkl) and says so in the stats; its 313-row warmup trim also
  covers every new column's warmup (max 200), so rows are kept
  either way.
- A truncation causality self-check runs on every build: features
  recomputed on a truncated long frame must match the full-frame
  values exactly on a sample of new columns at shared timestamps.

Lab only: no training, no registration; trading stays PARKED.

Usage: .venv/bin/python scripts/build_timeseries_matrix_v2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import main_chart as mc  # noqa: E402

V1_CSV = ROOT / "data" / "timeseries_matrix" / "btc_usd_1h_matrix.csv"
LONG_HOURLY = ROOT / "data" / "history" / "btc_usd_1h_long.csv"
SUP_PATH = ROOT / "data" / "diagnosis" / "sup.pkl"
OUT_DIR = ROOT / "data" / "timeseries_matrix"
V2_CSV = OUT_DIR / "btc_usd_1h_matrix_v2.csv"
STATS_PATH = OUT_DIR / "build_stats_v2.json"

TRUNC_ROWS = 30000
SAMPLE_CHECK_COLUMNS = [
    "Distance to SMA 200 (%)",
    "Distance to Donchian Upper (55) (%)",
    "Ichimoku Senkou Span A Distance (%)",
    "Parabolic SAR Distance (%)",
    "SuperTrend Side (+1/-1)",
    "Distance to Prior-Day Pivot Point (%)",
    "Distance to Fractal Resistance (%)",
    "Distance to $5,000 Round Number (%)",
]


def load_feature_frame() -> tuple[pd.DataFrame, str]:
    if LONG_HOURLY.exists():
        df = pd.read_csv(LONG_HOURLY, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        df.columns = [c.lower() for c in df.columns]
        return df, (f"long hourly pull {LONG_HOURLY.name} "
                    f"({df.index[0]} -> {df.index[-1]}, "
                    f"{len(df)} rows)")
    sup = pd.read_pickle(SUP_PATH)
    df = sup[["open", "high", "low", "close", "volume"]].copy()
    return df, (f"FALLBACK sup.pkl ({df.index[0]} -> {df.index[-1]}, "
                f"{len(df)} rows) — long hourly pull unavailable")


def main() -> int:
    v1 = pd.read_csv(V1_CSV)
    v1_ts = pd.to_datetime(v1["Date"])
    print(f"v1: {len(v1)} rows x {len(v1.columns)} cols", flush=True)

    frame, src_desc = load_feature_frame()
    print(f"feature frame: {src_desc}", flush=True)

    print("building main-chart features (full frame) ...", flush=True)
    t0 = time.time()
    feat = mc.build_all(frame)
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    # ---- causality self-check on a truncated frame (v1 pattern:
    # rebuild on the first TRUNC_ROWS of the feature frame and
    # compare on the truncated frame's own timestamps) ----
    print(f"self-check: rebuilding on first {TRUNC_ROWS} rows ...",
          flush=True)
    trunc_feat = mc.build_all(frame.iloc[:TRUNC_ROWS])
    trunc_index = trunc_feat.index
    check = {}
    ok_all = True
    for colname in SAMPLE_CHECK_COLUMNS:
        a = feat[colname].reindex(trunc_index).to_numpy()
        b = trunc_feat[colname].to_numpy()
        ok = bool(np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True))
        check[colname] = ok
        ok_all = ok_all and ok
        print(f"  check {colname}: {'PASS' if ok else 'FAIL'}",
              flush=True)
    print(f"SELF-CHECK: {'PASS' if ok_all else 'FAIL'}", flush=True)
    if not ok_all:
        print("aborting: causality self-check failed", flush=True)
        return 1

    # ---- align new columns onto v1 rows ----
    aligned = feat.reindex(v1_ts)
    missing_ts = int(aligned.isna().all(axis=1).sum())
    nan_counts = {c: int(aligned[c].isna().sum()) for c in feat.columns}
    bad = {c: n for c, n in nan_counts.items() if n > 0}
    print(f"aligned: missing timestamps={missing_ts}, "
          f"columns with NaN on v1 rows: {bad}", flush=True)
    if missing_ts or bad:
        print("aborting: alignment produced holes", flush=True)
        return 1

    v2 = v1.copy()
    for c in feat.columns:
        v2[c] = aligned[c].to_numpy()
    assert len(v2.columns) == len(v1.columns) + len(feat.columns)
    v2.to_csv(V2_CSV, index=False)

    # ---- verify the written file: v1 part identical, shape right --
    back = pd.read_csv(V2_CSV)
    assert list(back.columns[:len(v1.columns)]) == list(v1.columns)
    assert len(back) == len(v1)
    max_rt_diff = 0.0
    for c in v1.columns:
        a, b = v1[c], back[c]
        if a.dtype.kind in "fiu":
            d = np.abs(a.to_numpy(float) - b.to_numpy(float))
            max_rt_diff = max(max_rt_diff, float(d.max()))
            assert np.allclose(a.to_numpy(float), b.to_numpy(float),
                               rtol=1e-9, atol=1e-6, equal_nan=True), c
        else:
            assert (a.astype(str) == b.astype(str)).all(), c
    print(f"verified: v2 {len(back)} rows x {len(back.columns)} cols; "
          f"v1 columns identical (max CSV round-trip diff "
          f"{max_rt_diff:.2e}, text-formatting precision)", flush=True)

    stats = {
        "feature_frame": src_desc,
        "v1_rows": int(len(v1)), "v1_columns": int(len(v1.columns)),
        "v2_rows": int(len(back)), "v2_columns": int(len(back.columns)),
        "new_columns": {name: defn for name, _b, defn
                        in mc.INDICATORS},
        "nan_on_v1_rows": nan_counts,
        "v1_roundtrip_max_abs_diff": max_rt_diff,
        "self_check": {"pass": ok_all, "columns": check,
                       "trunc_rows": TRUNC_ROWS,
                       "shared_timestamps": int(len(trunc_index))},
        "csv_path": str(V2_CSV), "csv_bytes": V2_CSV.stat().st_size,
    }
    STATS_PATH.write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: stats[k] for k in
                      ["v2_rows", "v2_columns", "csv_bytes"]}, indent=1),
          flush=True)
    print(f"wrote {V2_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
