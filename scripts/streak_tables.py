"""Streak tables on LONG BTC/USD history — Part A of the 2026-10-22
research build (Kass green-lit: streak tables at slower timeframes
with longer Coinbase history pulled in, plus main-chart indicators
for the data matrix as a separate part).

Timeframes: 12h (resampled from the long hourly pull), 1D (long
daily pull), 1Month (resampled from the long daily pull).
Candle types: RAW (color = close >= open) and HEIKIN-ASHI (standard
recursion; color = HA close >= HA open).

Definitions (all causal — no future information anywhere):
- runlen_ending[t] = number of consecutive same-color candles ending
  at t (t included). A "streak of EXACT length k at t" means
  runlen_ending[t] == k; the streak may continue past t.
- A completed RUN ends at t where color[t+1] != color[t] (the last
  candle of the series also ends a run, right-censored). Run-length
  stats are over completed runs.
- Continuation: P(color[t+1] == color[t] | runlen_ending[t] == k).
- Forward returns: close[t+h] / close[t] - 1 in percent, REAL
  prices, after streaks of exact length k ending at t.

CORRECTNESS (a previous version of this analysis had a run-end bug —
marking run ends by comparing runlen to the next row's runlen):
here run ends are found from the COLOR series directly, and the
script ASSERTS (a) run lengths tile the series (sum == candle
count), (b) green-run and red-run counts differ by <= 1.

Outputs: data/streaks/streaks_<tf>_<type>.csv (tidy metric rows),
data/streaks/streak_summary.json. Report:
refs/streaks-long-2026-10-22.md (written by hand from the JSON).

Usage: .venv/bin/python scripts/streak_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "history"
OUT = ROOT / "data" / "streaks"
MATRIX_CSV = ROOT / "data" / "timeseries_matrix" / "btc_usd_1h_matrix.csv"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({"open": "first", "high": "max",
                                  "low": "min", "close": "last",
                                  "volume": "sum"}).dropna()


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Standard HA recursion. HA close = (O+H+L+C)/4; HA open[0] =
    (O[0]+C[0])/2; HA open[t] = (HA open[t-1] + HA close[t-1]) / 2."""
    ha_close = (df["open"] + df["high"] + df["low"]
                + df["close"]).to_numpy(float) / 4.0
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    ha_open = np.empty(len(df))
    ha_open[0] = (o[0] + c[0]) / 2.0
    for t in range(1, len(df)):
        ha_open[t] = (ha_open[t - 1] + ha_close[t - 1]) / 2.0
    return pd.DataFrame({"open": ha_open, "close": ha_close},
                        index=df.index)


def run_lengths(color: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (runlen_ending, completed_run_lengths_with_colors).
    completed: list of (length, color) per finished run; the final
    run is included (right-censored)."""
    n = len(color)
    runlen = np.ones(n, dtype=int)
    for t in range(1, n):
        runlen[t] = runlen[t - 1] + 1 if color[t] == color[t - 1] else 1
    ends = np.where(np.diff(color) != 0)[0]  # last index of each run
    ends = np.append(ends, n - 1)
    lengths = runlen[ends]
    colors = color[ends]
    # --- correctness asserts (the tiling rule) ---
    assert int(lengths.sum()) == n, \
        f"run lengths {lengths.sum()} != candle count {n}"
    ng = int((colors == 1).sum())
    nr = int((colors == 0).sum())
    assert abs(ng - nr) <= 1, f"run counts unbalanced: G={ng} R={nr}"
    return runlen, np.stack([lengths, colors], axis=1)


def analyze(df: pd.DataFrame, color: np.ndarray, horizons: list[int]
            ) -> dict:
    n = len(df)
    close = df["close"].to_numpy(float)
    runlen, runs = run_lengths(color)
    res: dict = {"n_candles": n, "colors": {}}
    for cval, cname in [(1, "green"), (0, "red")]:
        L = runs[runs[:, 1] == cval, 0]
        dist = {str(k): int((L == k).sum()) for k in range(1, 7)}
        dist["7+"] = int((L >= 7).sum())
        res["colors"][cname] = {
            "runs": int(len(L)),
            "avg": float(L.mean()), "median": float(np.median(L)),
            "max": int(L.max()), "dist_counts": dist,
            "dist_pct": {k: float(100 * v / len(L))
                         for k, v in dist.items()},
            "continuation": {}, "forward": {},
        }
        # continuation: P(next candle same color | runlen == k)
        for k in range(1, 7):
            rows = np.where((runlen == k) & (color == cval))[0]
            rows = rows[rows < n - 1]
            if len(rows):
                p = float((color[rows + 1] == cval).mean())
            else:
                p = float("nan")
            res["colors"][cname]["continuation"][str(k)] = {
                "p": p, "n": int(len(rows))}
        # forward REAL-price returns after exact-k streaks
        for k in range(2, 9):
            rows = np.where((runlen == k) & (color == cval))[0]
            per_h = {}
            for h in horizons:
                ok = rows[rows + h < n]
                if len(ok):
                    r = close[ok + h] / close[ok] - 1.0
                    per_h[str(h)] = {
                        "mean_pct": float(100 * r.mean()),
                        "median_pct": float(100 * np.median(r)),
                        "p_gt0": float((r > 0).mean()),
                        "n": int(len(ok))}
                else:
                    per_h[str(h)] = {"mean_pct": None, "median_pct": None,
                                     "p_gt0": None, "n": 0}
            res["colors"][cname]["forward"][str(k)] = per_h
    return res


def tidy_rows(tf: str, ctype: str, a: dict) -> list[dict]:
    rows = []
    base = {"timeframe": tf, "candle_type": ctype}
    for cname, cd in a["colors"].items():
        rows.append({**base, "metric": "runlen_summary", "color": cname,
                     "k": "", "horizon": "", "n": cd["runs"],
                     "value": cd["avg"], "median": cd["median"],
                     "max": cd["max"], "mean_pct": "", "median_pct": "",
                     "p_gt0": ""})
        for k, cnt in cd["dist_counts"].items():
            rows.append({**base, "metric": "run_length_dist",
                         "color": cname, "k": k, "horizon": "",
                         "n": cnt, "value": cd["dist_pct"][k],
                         "median": "", "max": "", "mean_pct": "",
                         "median_pct": "", "p_gt0": ""})
        for k, cd2 in cd["continuation"].items():
            rows.append({**base, "metric": "continuation", "color": cname,
                         "k": k, "horizon": "", "n": cd2["n"],
                         "value": cd2["p"], "median": "", "max": "",
                         "mean_pct": "", "median_pct": "", "p_gt0": ""})
        for k, per_h in cd["forward"].items():
            for h, st in per_h.items():
                rows.append({**base, "metric": "forward_return",
                             "color": cname, "k": k, "horizon": h,
                             "n": st["n"], "value": "",
                             "median": "", "max": "",
                             "mean_pct": st["mean_pct"],
                             "median_pct": st["median_pct"],
                             "p_gt0": st["p_gt0"]})
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    daily = load(HIST / "btc_usd_1d_long.csv")
    hourly_path = HIST / "btc_usd_1h_long.csv"
    if hourly_path.exists():
        hourly = load(hourly_path)
        hourly_src = f"long hourly pull ({hourly.index[0]} -> " \
                     f"{hourly.index[-1]})"
    else:
        m = pd.read_csv(MATRIX_CSV, parse_dates=["Date"])
        hourly = m.rename(columns={"Open": "open", "High": "high",
                                   "Low": "low", "Close": "close",
                                   "Volume": "volume"})
        hourly = hourly.set_index("Date").sort_index()
        hourly = hourly[["open", "high", "low", "close", "volume"]]
        hourly_src = "FALLBACK: v1 matrix hourly frame (2024-11-23 -> "\
                     "2026-10-21) — long hourly pull unavailable"
    print(f"hourly source: {hourly_src}", flush=True)

    frames = {
        "12h": (resample(hourly, "12h"), [1, 3, 6]),
        "1d": (daily, [1, 3, 6]),
        "1mo": (resample(daily, "MS"), [1, 2, 3]),
    }
    summary = {"hourly_source": hourly_src, "timeframes": {}}
    all_rows: dict[str, list[dict]] = {}
    for tf, (frame, horizons) in frames.items():
        summary["timeframes"][tf] = {
            "n_candles": int(len(frame)),
            "first": str(frame.index[0]), "last": str(frame.index[-1]),
            "types": {}}
        raw_color = (frame["close"] >= frame["open"]).to_numpy(int)
        ha = heikin_ashi(frame)
        ha_color = (ha["close"] >= ha["open"]).to_numpy(int)
        for ctype, color in [("raw", raw_color), ("ha", ha_color)]:
            a = analyze(frame, color, horizons)
            summary["timeframes"][tf]["types"][ctype] = a
            key = f"streaks_{tf}_{ctype}"
            all_rows[key] = tidy_rows(tf, ctype, a)
            pd.DataFrame(all_rows[key]).to_csv(OUT / f"{key}.csv",
                                               index=False)
            g = a["colors"]["green"]
            r = a["colors"]["red"]
            print(f"{tf} {ctype}: n={a['n_candles']} "
                  f"G avg {g['avg']:.2f} / R avg {r['avg']:.2f}",
                  flush=True)
    (OUT / "streak_summary.json").write_text(
        json.dumps(summary, indent=1))
    print(f"wrote {OUT}/streak_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
