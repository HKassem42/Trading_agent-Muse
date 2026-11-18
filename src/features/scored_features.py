"""The matrix-v3 scored-indicator frame (the "scored102" feature
set): the 90 computable AZ + main-chart indicator series, each
converted to a trailing-4,320-bar percentile score (0-100, oriented
per Kass's bullish/bearish guide), plus the 4 family composites
(master / trend / momentum / volume) -- 94 feature columns.

Construction is the matrix builders' own code path
(scripts/build_timeseries_matrix_v3.py: build_series /
ranked_quantity / rolling_scores) -- no re-implementation -- ported
from the full-indicator stacker study builder
(scripts/build_full_indicator_stacker.py phase_build, Arm B), whose
fresh-build causality was verified under the v5.1 standard
(truncation rebuilds + a brute-force second method). Every score at
row p reads only candles up to p: ranked quantities are causal
series and rolling_scores windows END at the scored row.

LAB provenance: this module exists so the Training Lab can hold --
and nightly can grade -- a champion trained on the scored set
(run 64, crowned lab stacker Champion #2 on 2026-10-25). The
production feature path (technical_indicators.py, 21 features) is
untouched, and the live loop never imports this module: the
driver's seat runs the production stacker.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "matrix_v3_builder", ROOT / "scripts" / "build_timeseries_matrix_v3.py")
_b3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b3)

WINDOW = _b3.WINDOW  # 4320 trailing bars (180 days of hourly candles)
V2_CSV = ROOT / "data" / "timeseries_matrix" / "btc_usd_1h_matrix_v2.csv"
COMPOSITES = ["score_master", "score_trend", "score_momentum",
              "score_volume"]


def raw90_columns() -> list[str]:
    """The canonical 90 indicator columns of the matrix-v2 set, in
    file order (59 AZ raw + 31 main-chart) -- the same derivation the
    study used, so a frame built here lines up column-for-column
    with the artifacts the study trained."""
    with open(V2_CSV) as fh:
        header = next(csv.reader(fh))
    cols = [c for c in header if c in _b3.SPEC]
    assert len(cols) == 90, len(cols)
    return cols


def scored_columns() -> list[str]:
    """The full scored102 column list: 4 composites + the 90 scored
    series, in the study's arm-B order."""
    return COMPOSITES + raw90_columns()


def build_scored_frame(df: pd.DataFrame,
                       names: list[str] | None = None
                       ) -> tuple[pd.DataFrame, list[str]]:
    """Build the scored frame over a raw OHLCV frame (as produced by
    fetch_klines / the study's raw snapshot). Returns (frame, cols):
    the frame is indexed exactly like `df`, with NaN on every row
    before the scored region opens (the last indicator's warmup plus
    one full trailing window must exist first). `names` restricts
    the scored series (tests / partial builds); the default is the
    full 90, which yields exactly scored_columns()."""
    names = list(names) if names is not None else raw90_columns()
    feat = _b3.build_series(df)
    close = df["close"]
    qa_map: dict[str, np.ndarray] = {}
    p0 = WINDOW - 1
    for name in names:
        q = _b3.ranked_quantity(name, feat, close)
        qa = q.to_numpy(float)
        isn = np.isnan(qa) | np.isinf(qa)
        last_bad = int(np.max(np.nonzero(isn))) if isn.any() else -1
        p0 = max(p0, last_bad + WINDOW)
        qa_map[name] = qa
    positions = np.arange(p0, len(df))
    if len(positions) == 0:
        raise ValueError(
            f"scored frame needs more history: {len(df)} rows leave "
            f"no fully-warmed position (window {WINDOW})")
    scores: dict[str, np.ndarray] = {}
    for name in names:
        s = _b3.rolling_scores(qa_map[name], positions)
        if _b3.SPEC[name][2] == "FLIP":
            s = 100.0 - s
        assert s.min() >= 0.0 and s.max() <= 100.0, name
        scores[name] = s
    fam_members: dict[str, list[str]] = {"trend": [], "momentum": [],
                                         "volume": []}
    for name in names:
        fam = _b3.SPEC[name][3]
        if fam:
            fam_members[fam].append(name)
    fam_scores = {f: np.mean([scores[n] for n in ms], axis=0)
                  for f, ms in fam_members.items() if ms}
    master = np.mean([fam_scores[f] for f in
                      ("trend", "momentum", "volume")
                      if f in fam_scores], axis=0)
    comp_values = {"score_master": master,
                   "score_trend": fam_scores.get("trend"),
                   "score_momentum": fam_scores.get("momentum"),
                   "score_volume": fam_scores.get("volume")}
    cols = [c for c in COMPOSITES if comp_values[c] is not None] + names
    frame = pd.DataFrame(index=df.index)
    for cname in cols:
        arr = (comp_values[cname] if cname in comp_values
               else scores[cname])
        full = np.full(len(df), np.nan)
        full[positions] = arr
        frame[cname] = full
    return frame[cols], cols
