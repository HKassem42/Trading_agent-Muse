"""Features-v3 contract audit.

Over ~1 year (and, as fetched, ~2 years) of real 1h BTC-USD data, assert
every feature column lies in [-1, 1] and print the actual min/max per
feature. Exits non-zero on any violation.

Usage: .venv/bin/python scripts/audit_features_v3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.yfinance_adapter import fetch_klines
from features.technical_indicators import FEATURE_COLUMNS, add_features


def main() -> int:
    df = fetch_klines("BTC-USD", interval="1h", period="730d")
    feat = add_features(df).dropna(subset=FEATURE_COLUMNS)
    print(f"audited rows={len(feat)}  {feat.index[0]} .. {feat.index[-1]}")
    print(f"{'feature':<16} {'min':>10} {'max':>10} {'ok':>4}")
    bad = []
    for c in FEATURE_COLUMNS:
        lo, hi = float(feat[c].min()), float(feat[c].max())
        ok = lo >= -1.0 and hi <= 1.0
        if not ok:
            bad.append(c)
        print(f"{c:<16} {lo:>10.4f} {hi:>10.4f} {'yes' if ok else 'NO':>4}")
    if bad:
        print(f"CONTRACT VIOLATED: {bad}")
        return 1
    print("CONTRACT OK: every feature within [-1, 1]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
