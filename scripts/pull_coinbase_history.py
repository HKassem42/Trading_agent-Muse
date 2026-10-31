"""Pull long BTC/USD history from the Coinbase Exchange public API.

No auth. Endpoint: GET /products/BTC-USD/candles?granularity=G&
start=ISO&end=ISO -> rows [time, low, high, open, close, volume],
max 300 candles per call, descending time order.

Part A of the 2026-10-22 research build (Kass green-lit):
- DAILY  (86400) 2015-01-01 -> 2026-10-21  (Coinbase's first BTC/USD
  candle is 2015-07-20; earlier chunks simply return nothing before it)
- HOURLY (3600)  2022-01-01 -> 2026-10-21

Checkpointed / resumable: each chunk is written to
data/history/chunks_<gran>/<chunk_start>.csv as soon as it lands and
recorded in data/history/pull_progress.json; a rerun skips chunks
already on disk. The assembled outputs are written only from chunks:
  data/history/btc_usd_1d_long.csv
  data/history/btc_usd_1h_long.csv
with columns Date,Open,High,Low,Close,Volume (UTC, ascending).

Usage: .venv/bin/python scripts/pull_coinbase_history.py [daily|hourly|all]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "history"
PROGRESS = HIST / "pull_progress.json"
API = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

END = pd.Timestamp("2026-10-22T00:00:00Z")  # exclusive-ish; filtered below
JOBS = {
    "daily": dict(gran=86400, start="2015-01-01T00:00:00Z",
                  chunk=pd.Timedelta(days=290),
                  out="btc_usd_1d_long.csv", max_ts="2026-10-21T00:00:00Z"),
    "hourly": dict(gran=3600, start="2022-01-01T00:00:00Z",
                   chunk=pd.Timedelta(hours=290),
                   out="btc_usd_1h_long.csv", max_ts="2026-10-21T23:00:00Z"),
}


def load_progress() -> dict:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {}


def save_progress(p: dict) -> None:
    PROGRESS.write_text(json.dumps(p, indent=1, sort_keys=True))


def fetch_chunk(gran: int, start: pd.Timestamp, end: pd.Timestamp,
                tries: int = 6) -> pd.DataFrame:
    params = {"granularity": gran,
              "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "end": end.strftime("%Y-%m-%dT%H:%M:%SZ")}
    last_err = None
    for attempt in range(tries):
        try:
            r = requests.get(API, params=params, timeout=45)
            if r.status_code == 200:
                rows = r.json()
                if not rows:
                    return pd.DataFrame(
                        columns=["time", "low", "high", "open", "close",
                                 "volume"])
                df = pd.DataFrame(rows, columns=["time", "low", "high",
                                                 "open", "close", "volume"])
                return df
            last_err = f"HTTP {r.status_code}: {r.text[:160]}"
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)
        time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"chunk {params} failed after {tries} tries: "
                       f"{last_err}")


def pull(job_name: str) -> pd.DataFrame:
    job = JOBS[job_name]
    gran = job["gran"]
    cdir = HIST / f"chunks_{job_name}"
    cdir.mkdir(parents=True, exist_ok=True)
    progress = load_progress()
    done = set(progress.get(job_name, []))

    start = pd.Timestamp(job["start"])
    starts = []
    t = start
    while t < END:
        starts.append(t)
        t = t + job["chunk"]

    for cs in starts:
        key = cs.strftime("%Y%m%dT%H%M")
        cpath = cdir / f"{key}.csv"
        if key in done and cpath.exists():
            continue
        ce = min(cs + job["chunk"], END)
        df = fetch_chunk(gran, cs, ce)
        df.to_csv(cpath, index=False)
        done.add(key)
        progress[job_name] = sorted(done)
        save_progress(progress)
        print(f"[{job_name}] {key}: {len(df)} candles", flush=True)
        time.sleep(0.15)

    frames = []
    for cs in starts:
        cpath = cdir / f"{cs.strftime('%Y%m%dT%H%M')}.csv"
        if cpath.exists():
            d = pd.read_csv(cpath)
            if len(d):
                frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(raw["time"], unit="s", utc=True)
    out["Open"] = raw["open"].astype(float)
    out["High"] = raw["high"].astype(float)
    out["Low"] = raw["low"].astype(float)
    out["Close"] = raw["close"].astype(float)
    out["Volume"] = raw["volume"].astype(float)
    out = (out.drop_duplicates(subset="Date")
              .sort_values("Date").reset_index(drop=True))
    out = out[out["Date"] <= pd.Timestamp(job["max_ts"])]
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    opath = HIST / job["out"]
    out.to_csv(opath, index=False)
    print(f"[{job_name}] assembled {len(out)} rows -> {opath}",
          flush=True)
    return out


def main() -> int:
    HIST.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (["daily", "hourly"] if which == "all" else [which]):
        pull(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
