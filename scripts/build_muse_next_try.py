"""Muse's Next Try -- LAB study (2026-11-04, Kass's order).

Kass: "pick my next model from start to finish. WHATEVER you want
to try to get the highest model score out of all we have so far and
beat your previous try... Label it as ur next try. GO!"

Pre-registered in refs/muse-next-try-preregistration-2026-11-04.md
BEFORE any grading. In one line: Pick #2's 98 eyes PLUS 22 new
causal columns built from information that is NOT in BTC's spot
candles -- Binance perpetual positioning (funding / open interest /
long-short ratios / taker flow / basis, from the official Binance
Vision archive), ETH cross-asset moves, the Master Score's own
6h/24h trajectory, and hour/day cyclicals (120 columns total).

Selection discipline (the anti-peeking contract):
- Frame screening uses ONLY train rows (fast ExtraTrees+LightGBM
  proxy, 3-fold TimeSeriesSplit). The 792-candle holdout is touched
  once, at final grading.
- The decision threshold is picked on train out-of-fold votes only
  and used only if it beats 0.50 by >= 0.005 OOF accuracy.
- Champions (run 96 prod21, run 73 pick2_98) are re-scored on the
  identical holdout timestamps TWO independent ways each; the two
  ways must agree exactly or no number is trusted.
- The 98-column control is graded on the holdout for ablation
  reporting ONLY; it can never replace the CV-selected frame.

LAB ONLY under the promotion lock: a win records pending_approval
for Kass; nothing installs; champion/seat/shadow/gated files are
hash-guarded before/after.

Usage: .venv/bin/python scripts/build_muse_next_try.py
Phases checkpoint to data/muse_next_try/; reruns skip done phases.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.market_data import fetch_klines  # noqa: E402
from features import next_try_features, pick2_features, scored_features  # noqa: E402
from features.technical_indicators import (  # noqa: E402
    FEATURE_COLUMNS, feature_matrix, make_supervised)
from models.muse_pick import build_pick_pipeline  # noqa: E402
from training import nightly  # noqa: E402
from training.nightly import add_lab_note, log_run, vote_histogram  # noqa: E402

MODELS = ROOT / "models"
LAB = ROOT / "data" / "muse_next_try"
EXT = LAB / "external"
PROGRESS_PATH = LAB / "progress.json"
RESULTS_PATH = LAB / "results.json"
DF_PATH = LAB / "raw_ohlcv.pkl"
ETH_PATH = LAB / "eth_ohlcv.pkl"
FRAME_PATH = LAB / "frame120.pkl"
SUP_PATH = LAB / "prod_sup.pkl"
STACKER_PATH = LAB / "stacker_next_try.pkl"

HOLDOUT = 792
VISION = "https://data.binance.vision/data/futures/um"
GUARD_FILES = {
    "lab_ppo": MODELS / "ppo_champion.zip",
    "lab_stacker": MODELS / "stacker_champion.pkl",
    "lab_stacker_meta": MODELS / "stacker_champion.json",
    "seat_stacker": MODELS / "live_seat" / "stacker.pkl",
    "seat_ppo": MODELS / "live_seat" / "ppo.zip",
    "shadow_stacker": MODELS / "shadow_seat" / "stacker.pkl",
    "shadow_ppo": MODELS / "shadow_seat" / "ppo.zip",
    "gated_stacker": MODELS / "gated_seat" / "stacker.pkl",
    "run73_artifact": ROOT / "data" / "muse_pick2" / "stacker_pick2.pkl",
}
COLS98 = list(scored_features.scored_columns()) + list(
    pick2_features.NEW_COLUMNS)
DESIGN = ("Muse's Next Try stacker: Pick recipe (RF+XGB+LGBM+ET -> LR, "
          "SMOTE) on Pick #2's 98 scored eyes + 22 new causal columns "
          "(Binance perp funding/OI/L-S ratios/taker/basis, ETH "
          "cross-asset, Master-Score trajectory, calendar) = 120 feats; "
          "frame + threshold selected on train CV only")


# ---------------------------------------------------------------- state
def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def _load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return default
    return default


def _save_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_clean(obj), indent=1))
    tmp.replace(path)


def progress() -> set:
    return set(_load_json(PROGRESS_PATH, {}).get("done", []))


def mark(phase: str) -> None:
    _save_json(PROGRESS_PATH, {"done": sorted(progress() | {phase})})


def results() -> dict:
    return _load_json(RESULTS_PATH, {})


def save_result(key, value) -> None:
    res = results()
    res[key] = value
    _save_json(RESULTS_PATH, res)


def _sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _guard() -> dict:
    return {k: _sha(p) for k, p in GUARD_FILES.items()}


def _assert_guard(before, after):
    for k in GUARD_FILES:
        assert after[k] == before[k], f"GUARD TRIPPED on {k}"


def get_df():
    if DF_PATH.exists():
        return pd.read_pickle(DF_PATH)
    df = fetch_klines("BTC-USD", interval="1h", period="730d")
    LAB.mkdir(parents=True, exist_ok=True)
    df.to_pickle(DF_PATH)
    return df


def get_frames():
    df = get_df()
    prod_sup = make_supervised(df)
    split = len(prod_sup) - HOLDOUT
    return df, prod_sup, split


# ---------------------------------------------------------------- phases
def phase_data() -> None:
    df, prod_sup, split = get_frames()
    if ETH_PATH.exists():
        eth = pd.read_pickle(ETH_PATH)
    else:
        eth = fetch_klines("ETH-USD", interval="1h", period="730d")
        eth.to_pickle(ETH_PATH)
    save_result("meta", {
        "rows_raw": len(df), "rows_sup": len(prod_sup), "split": split,
        "holdout_start": str(prod_sup.index[split]),
        "data_first": str(df.index[0]), "data_last": str(df.index[-1]),
        "eth_rows": len(eth), "eth_first": str(eth.index[0]),
        "eth_last": str(eth.index[-1]),
        "guard_before": _guard(),
        "cols98": COLS98, "design": DESIGN,
    })
    print(f"data: BTC {len(df)} rows ({df.index[0]} -> {df.index[-1]}), "
          f"sup {len(prod_sup)}, holdout from {prod_sup.index[split]}; "
          f"ETH {len(eth)} rows", flush=True)


def _fetch_zip_csv(url: str) -> pd.DataFrame | None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            blob = r.read()
    except Exception as e:
        print(f"  fetch failed {url.split('/')[-1]}: {e}", flush=True)
        return None
    z = zipfile.ZipFile(io.BytesIO(blob))
    name = z.namelist()[0]
    return pd.read_csv(io.BytesIO(z.read(name)))


def phase_external() -> None:
    """Download the Binance Vision archive pieces for the df span
    (cached under external/) and assemble the raw external inputs:
    funding (monthly zips), perp 1h klines (monthly zips), metrics
    (daily zips, threaded)."""
    df = get_df()
    EXT.mkdir(parents=True, exist_ok=True)
    start = (df.index[0] - pd.Timedelta(days=2)).normalize()
    end = df.index[-1].normalize()
    months = pd.period_range(start, end, freq="M")
    days = pd.date_range(start, end, freq="D")

    # funding (monthly)
    fpath = EXT / "funding.pkl"
    if fpath.exists():
        funding = pd.read_pickle(fpath)
    else:
        parts = []
        for m in months:
            u = (f"{VISION}/monthly/fundingRate/BTCUSDT/"
                 f"BTCUSDT-fundingRate-{m}.zip")
            d = _fetch_zip_csv(u)
            if d is not None:
                parts.append(d)
        f = pd.concat(parts, ignore_index=True)
        funding = (pd.to_datetime(f["calc_time"], unit="ms"),
                   f["last_funding_rate"].to_numpy(dtype=float))
        pd.to_pickle(funding, fpath)
    # perp klines (monthly)
    ppath = EXT / "perp_1h.pkl"
    if ppath.exists():
        perp = pd.read_pickle(ppath)
    else:
        parts = []
        for m in months:
            u = (f"{VISION}/monthly/klines/BTCUSDT/1h/"
                 f"BTCUSDT-1h-{m}.zip")
            d = _fetch_zip_csv(u)
            if d is not None:
                parts.append(d)
        p = pd.concat(parts, ignore_index=True)
        # Binance Vision switched kline timestamps from ms to us in
        # the 2025 files; normalize by magnitude or every 2025+
        # candle lands in a far-future year and aligns to nothing.
        ot = p["open_time"].astype("int64").to_numpy()
        ot = np.where(ot > 10**14, ot // 1000, ot)
        # .to_numpy() is load-bearing: Series values would ALIGN
        # on their RangeIndex against the new datetime index and
        # produce an all-NaN frame (the label-alignment trap).
        perp = pd.DataFrame({
            "close": p["close"].astype(float).to_numpy(),
            "volume": p["volume"].astype(float).to_numpy(),
            "taker_buy_volume": p["taker_buy_volume"].astype(float).to_numpy(),
        }, index=pd.to_datetime(ot, unit="ms"))
        perp = perp[~perp.index.duplicated(keep="last")].sort_index()
        perp.to_pickle(ppath)
    # metrics (daily, threaded)
    mpath = EXT / "metrics_5m.pkl"
    if mpath.exists():
        metrics = pd.read_pickle(mpath)
    else:
        def one(day):
            u = (f"{VISION}/daily/metrics/BTCUSDT/"
                 f"BTCUSDT-metrics-{day:%Y-%m-%d}.zip")
            return _fetch_zip_csv(u)
        parts, missing = [], []
        with ThreadPoolExecutor(max_workers=16) as ex:
            for day, d in zip(days, ex.map(one, days)):
                if d is None:
                    missing.append(str(day.date()))
                else:
                    parts.append(d)
        m = pd.concat(parts, ignore_index=True)
        m["create_time"] = pd.to_datetime(m["create_time"])
        metrics = m.sort_values("create_time").reset_index(drop=True)
        metrics.to_pickle(mpath)
        print(f"  metrics: {len(metrics)} rows, missing days: {missing}",
              flush=True)
    # ---- tail stitch (Amendment 1, pre-registered before grading):
    # Binance Vision lags ~2 days at the live edge. Fill funding
    # settlements and perp candles after Binance's last timestamps
    # from OKX BTC-USDT-SWAP (public API); taker-buy share for the
    # stitched hours carries the last Binance value.
    import json as _json
    import urllib.request as _rq

    def _okx(url):
        req = _rq.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with _rq.urlopen(req, timeout=20) as r:
            return _json.loads(r.read())["data"]

    ft, fr = funding
    ft = pd.DatetimeIndex(ft)  # pickle may carry a Series
    funding = (ft, fr)
    bin_last_f = pd.Timestamp(ft[-1])
    okx_f = _okx("https://www.okx.com/api/v5/public/"
                 "funding-rate-history?instId=BTC-USDT-SWAP&limit=100")
    add_t, add_r = [], []
    for row in okx_f:
        ts = pd.to_datetime(int(row["fundingTime"]), unit="ms")
        if ts > bin_last_f:
            add_t.append(ts)
            add_r.append(float(row["fundingRate"]))
    if add_t:
        order = np.argsort(add_t)
        ft = ft.append(pd.DatetimeIndex([add_t[i] for i in order]))
        fr = np.concatenate([fr, [add_r[i] for i in order]])
        funding = (ft, fr)
        pd.to_pickle(funding, fpath)
    bin_last_p = perp.index[-1]
    okx_c = _okx("https://www.okx.com/api/v5/market/history-candles"
                 "?instId=BTC-USDT-SWAP&bar=1H&limit=100")
    last_share = float((perp["taker_buy_volume"] /
                        perp["volume"]).iloc[-1])
    add_rows = {}
    for row in okx_c:
        ts = pd.to_datetime(int(row[0]), unit="ms")
        if ts > bin_last_p:
            vol = float(row[6])  # volCcy: base-currency (BTC) volume
            add_rows[ts] = {"close": float(row[4]), "volume": vol,
                            "taker_buy_volume": last_share * vol}
    if add_rows:
        perp = pd.concat([perp, pd.DataFrame(add_rows).T]).sort_index()
        perp = perp.astype(float)
        perp.to_pickle(ppath)
    # Gap fill: the Vision Oct-2026 monthly file is missing 2026-10-28
    # entirely (720 rows for a 31-day month -- verified in the raw
    # archive). Fill every df hour the perp frame still lacks from
    # OKX history-candles, paged backwards with the `after` cursor.
    missing = df.index.difference(perp.index)
    gap_filled = 0
    if len(missing):
        want = set(pd.DatetimeIndex(missing))
        got: dict = {}
        cursor = None
        for _page in range(60):
            url = ("https://www.okx.com/api/v5/market/history-candles"
                   "?instId=BTC-USDT-SWAP&bar=1H&limit=100")
            if cursor is not None:
                url += f"&after={cursor}"
            rows = _okx(url)
            if not rows:
                break
            for row in rows:
                ts = pd.to_datetime(int(row[0]), unit="ms")
                if ts in want:
                    vol = float(row[6])
                    got[ts] = {"close": float(row[4]), "volume": vol,
                               "taker_buy_volume": last_share * vol}
            cursor = min(int(r[0]) for r in rows)
            if pd.to_datetime(cursor, unit="ms") <= min(want):
                break
        assert len(got) == len(want), (
            f"OKX gap fill incomplete: {len(got)}/{len(want)} hours")
        perp = pd.concat([perp, pd.DataFrame(got).T]).sort_index()
        perp = perp.astype(float)
        perp.to_pickle(ppath)
        gap_filled = len(got)
    metrics_stale_h = float((df.index[-1] + pd.Timedelta(hours=1)
                             - metrics["create_time"].iloc[-1])
                            / pd.Timedelta(hours=1))
    cover = {
        "funding_settlements": int(len(fr)),
        "funding_first": str(ft[0]), "funding_last": str(ft[-1]),
        "perp_rows": int(len(perp)),
        "perp_first": str(perp.index[0]), "perp_last": str(perp.index[-1]),
        "metrics_rows": int(len(metrics)),
        "metrics_first": str(metrics["create_time"].iloc[0]),
        "metrics_last": str(metrics["create_time"].iloc[-1]),
        "tail_funding_from_okx": int(len(add_t)),
        "tail_perp_hours_from_okx": int(len(add_rows)),
        "gap_perp_hours_from_okx": int(gap_filled),
        "metrics_stale_hours_at_data_end": metrics_stale_h,
    }
    save_result("external", cover)
    print(f"external: {cover}", flush=True)


def _external_inputs():
    funding = pd.read_pickle(EXT / "funding.pkl")
    metrics = pd.read_pickle(EXT / "metrics_5m.pkl")
    perp = pd.read_pickle(EXT / "perp_1h.pkl")
    eth = pd.read_pickle(ETH_PATH)
    return funding, metrics, perp, eth["close"]


def phase_frames() -> None:
    df, prod_sup, split = get_frames()
    funding, metrics, perp, eth_close = _external_inputs()
    t0 = time.time()
    frame, cols, evidence = next_try_features.build_next_try_frame(
        df, funding, metrics, perp, eth_close)
    print(f"  frame built in {time.time() - t0:.0f}s; "
          f"evidence={evidence}", flush=True)
    assert cols == COLS98 + next_try_features.NEW_COLUMNS
    hold_idx = prod_sup.index[split:]
    Xhold = frame.reindex(hold_idx)[cols]
    assert np.isfinite(Xhold.to_numpy(dtype=float)).all(), \
        "non-finite next-try feature inside the holdout"
    train_rows = frame.reindex(prod_sup.index[:split]).dropna(subset=cols)
    assert len(train_rows) > 5_000, f"train too small: {len(train_rows)}"
    vals = train_rows[cols].to_numpy(dtype=float)
    assert vals.min() >= 0.0 and vals.max() <= 100.0

    # Causality: truncation rebuilds at two cuts. The truncated
    # builder sees only external observations < idx[K]+1h -- what a
    # builder at time K could see -- and every new column must match
    # the full build EXACTLY on every shared row.
    ft, fr = funding
    trunc_ev = {}
    for K in (9000, int(df.index.get_indexer([hold_idx[0]])[0]) + 300):
        end = df.index[K] + pd.Timedelta(hours=1)
        keep_f = _ns_mask(ft, end)
        frame_t, _, _ = next_try_features.build_next_try_frame(
            df.iloc[:K + 1], (ft[keep_f], fr[keep_f]),
            metrics[metrics["create_time"] < end],
            perp[perp.index < end], eth_close[eth_close.index < end])
        worst = 0.0
        for c in next_try_features.NEW_COLUMNS:
            a = frame[c].iloc[:K + 1].to_numpy(dtype=float)
            b = frame_t[c].to_numpy(dtype=float)
            both = np.isfinite(a) & np.isfinite(b)
            assert np.array_equal(np.isnan(a), np.isnan(b)), (c, K)
            if both.any():
                worst = max(worst, float(np.abs(a[both] - b[both]).max()))
        assert worst <= 1e-9, f"causality FAIL at K={K}: {worst}"
        trunc_ev[str(K)] = {"max_abs_diff_new_cols": worst}
        print(f"  causality truncation K={K}: PASS (all 22 new columns, "
              f"all shared rows, max diff {worst})", flush=True)

    frame.to_pickle(FRAME_PATH)
    prod_sup.to_pickle(SUP_PATH)
    cov = evidence["new_column_coverage_from_p0"]
    assert min(cov.values()) >= 0.95, f"coverage below 95%: {cov}"
    save_result("frames", {
        "train_rows": int(len(train_rows)),
        "holdout_rows": int(len(Xhold)),
        "frame_evidence": evidence,
        "causality_truncation": trunc_ev,
        "train_first": str(train_rows.index[0]),
        "train_last": str(train_rows.index[-1]),
    })
    print(f"frames: train {len(train_rows)} rows, holdout "
          f"{len(Xhold)} finite; coverage min "
          f"{min(cov.values()):.3f}", flush=True)


def _ns_mask(times, end) -> np.ndarray:
    return pd.DatetimeIndex(times) < end


def _train_Xy():
    """Common train rows (finite on the full 120 frame) + labels,
    shared by screening and final training so every candidate is
    judged on identical rows."""
    frame = pd.read_pickle(FRAME_PATH)
    prod_sup = pd.read_pickle(SUP_PATH)
    split = len(prod_sup) - HOLDOUT
    cols = COLS98 + next_try_features.NEW_COLUMNS
    train_idx = prod_sup.index[:split]
    X = frame.reindex(train_idx)[cols].dropna()
    y = prod_sup.reindex(X.index)["target"].astype(int)
    assert (X.index < prod_sup.index[split]).all()
    prod = prod_sup.reindex(X.index)
    Xprod, _ = feature_matrix(prod)
    return X, y, Xprod


def phase_screen() -> None:
    """Frame screening on TRAIN rows only: fast proxy ensemble
    (ExtraTrees + LightGBM), 3-fold TimeSeriesSplit, mean accuracy.
    The holdout is not touched in this phase."""
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import TimeSeriesSplit

    X, y, Xprod = _train_Xy()
    nt = next_try_features
    frames = {
        "a_control_98": (X[COLS98], COLS98),
        "b_info_112": (X[COLS98 + nt.INFO_COLUMNS],
                       COLS98 + nt.INFO_COLUMNS),
        "c_seqcal_106": (X[COLS98 + nt.SEQUENCE_COLUMNS
                           + nt.CALENDAR_COLUMNS],
                         COLS98 + nt.SEQUENCE_COLUMNS
                         + nt.CALENDAR_COLUMNS),
        "d_full_120": (X, list(X.columns)),
        "e_info_prod_35": (pd.concat(
            [Xprod.reset_index(drop=True),
             X[nt.INFO_COLUMNS].reset_index(drop=True)], axis=1),
            list(FEATURE_COLUMNS) + nt.INFO_COLUMNS),
    }
    scores = {}
    tss = TimeSeriesSplit(n_splits=3)
    for name, (Xf, cols) in frames.items():
        Xv = Xf.to_numpy(dtype=float)
        accs = []
        for tr, te in tss.split(Xv):
            preds = []
            for make in (
                lambda: ExtraTreesClassifier(
                    n_estimators=300, max_depth=None, n_jobs=-1,
                    random_state=42),
                lambda: LGBMClassifier(
                    n_estimators=300, learning_rate=0.05, max_depth=5,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                    random_state=42, verbose=-1),
            ):
                m = make()
                m.fit(Xv[tr], y.iloc[tr])
                preds.append(m.predict(Xv[te]))
            pred = (np.mean(preds, axis=0) >= 0.5).astype(int)
            accs.append(accuracy_score(y.iloc[te], pred))
        scores[name] = {"mean_acc": float(np.mean(accs)),
                        "fold_accs": [float(a) for a in accs],
                        "n_cols": len(cols)}
        print(f"  screen {name}: mean CV acc "
              f"{scores[name]['mean_acc']:.4f} folds "
              f"{[round(a, 4) for a in accs]}", flush=True)
    order = ["a_control_98", "b_info_112", "c_seqcal_106",
             "d_full_120", "e_info_prod_35"]
    best = max(order, key=lambda k: (scores[k]["mean_acc"],
                                     -scores[k]["n_cols"],
                                     -order.index(k)))
    save_result("screen", {"scores": scores, "selected": best,
                           "train_rows": int(len(X))})
    print(f"screen: selected frame = {best} "
          f"({scores[best]['mean_acc']:.4f} train-CV)", flush=True)


def _selected_cols():
    sel = results()["screen"]["selected"]
    nt = next_try_features
    return {
        "a_control_98": COLS98,
        "b_info_112": COLS98 + nt.INFO_COLUMNS,
        "c_seqcal_106": COLS98 + nt.SEQUENCE_COLUMNS + nt.CALENDAR_COLUMNS,
        "d_full_120": COLS98 + nt.NEW_COLUMNS,
        "e_info_prod_35": list(FEATURE_COLUMNS) + nt.INFO_COLUMNS,
    }[sel]


def _frame_X(frame, prod_sup, idx, cols):
    """X for an arbitrary column set mixing frame120 columns and
    prod21 columns (candidate e)."""
    parts = []
    fcols = [c for c in cols if c in frame.columns]
    pcols = [c for c in cols if c in FEATURE_COLUMNS]
    if fcols:
        parts.append(frame.reindex(idx)[fcols])
    if pcols:
        Xp, _ = feature_matrix(prod_sup.reindex(idx))
        parts.append(Xp[pcols])
    return pd.concat(parts, axis=1)[cols]


def phase_final() -> None:
    """Train the Pick recipe on the CV-selected frame, pick the
    threshold on train OOF votes only, then grade ONCE on the
    shared holdout vs both champions re-scored two ways."""
    from sklearn.metrics import accuracy_score, precision_score
    from sklearn.model_selection import TimeSeriesSplit

    frame = pd.read_pickle(FRAME_PATH)
    prod_sup = pd.read_pickle(SUP_PATH)
    split = len(prod_sup) - HOLDOUT
    cols = _selected_cols()
    X, y, _ = _train_Xy()
    Xtr = _frame_X(frame, prod_sup, X.index, cols)
    assert list(Xtr.index) == list(X.index)
    hold_idx = prod_sup.index[split:]
    yhold = prod_sup.iloc[split:]["target"].astype(int)
    Xhold = _frame_X(frame, prod_sup, hold_idx, cols)
    assert np.isfinite(Xhold.to_numpy(dtype=float)).all()

    # 5-fold walk-forward with OOF votes (train only).
    tss = TimeSeriesSplit(n_splits=5)
    oof = np.full(len(Xtr), np.nan)
    fold_accs, fold_precs = [], []
    for tr, te in tss.split(Xtr):
        pipe = build_pick_pipeline(random_state=42)
        pipe.fit(Xtr.iloc[tr], y.iloc[tr])
        oof[te] = pipe.predict_proba(Xtr.iloc[te])[:, 1]
        pred = (oof[te] >= 0.5).astype(int)
        fold_accs.append(accuracy_score(y.iloc[te], pred))
        fold_precs.append(precision_score(y.iloc[te], pred,
                                          zero_division=0))
    wf_acc = float(np.mean(fold_accs))
    wf_prec = float(np.mean(fold_precs))
    mask = np.isfinite(oof)
    acc50 = float(accuracy_score(y.iloc[mask], (oof[mask] >= 0.5)))
    best_theta, best_oof = 0.5, acc50
    for theta in np.arange(0.40, 0.601, 0.01):
        a = float(accuracy_score(y.iloc[mask], (oof[mask] >= theta)))
        if a > best_oof:
            best_oof, best_theta = a, float(theta)
    theta = best_theta if best_oof >= acc50 + 0.005 else 0.5
    print(f"  WF acc={wf_acc:.4f} prec={wf_prec:.4f}; OOF acc@0.50="
          f"{acc50:.4f}, best theta {best_theta:.2f} -> {best_oof:.4f}; "
          f"USING theta={theta:.2f}", flush=True)

    t0 = time.time()
    pipe = build_pick_pipeline(random_state=42)
    pipe.fit(Xtr, y)
    joblib.dump(pipe, STACKER_PATH)
    print(f"  final fit in {time.time() - t0:.0f}s", flush=True)
    proba = pipe.predict_proba(Xhold)[:, 1]
    pred = (proba >= theta).astype(int)
    pred50 = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(yhold, pred))
    acc_50 = float(accuracy_score(yhold, pred50))
    prec = float(precision_score(yhold, pred, zero_division=0))
    up = int(pred.sum())
    win_accs = []
    for k in range(3):
        seg = slice(k * 264, (k + 1) * 264)
        win_accs.append(float(accuracy_score(
            yhold.iloc[seg], pred[seg])))
    majority = float(max(yhold.mean(), 1 - yhold.mean()))

    # ---- champions, two independent ways each ----
    df, prod_sup_full, _ = get_frames()
    # Run 96 (prod21): way A = nightly champion_holdout_frame path;
    # way B = direct feature_matrix on the holdout.
    champ96 = joblib.load(MODELS / "stacker_champion.pkl")
    meta96 = json.loads((MODELS / "stacker_champion.json").read_text())
    assert meta96["run_id"] == 96 and meta96["feature_set"] == "prod21"
    Xa, tag = nightly.champion_holdout_frame(
        df, prod_sup_full.iloc[split:], meta96)
    assert tag == "prod21"
    Xb, _ = feature_matrix(prod_sup_full.iloc[split:])
    pa = champ96.predict_proba(Xa)[:, 1]
    pb = champ96.predict_proba(Xb)[:, 1]
    assert np.max(np.abs(pa - pb)) <= 1e-12
    acc96 = float(accuracy_score(yhold, (pa >= 0.5).astype(int)))
    prec96 = float(precision_score(yhold, (pa >= 0.5).astype(int),
                                   zero_division=0))
    up96 = int((pa >= 0.5).sum())
    # Run 73 (pick2_98): way A = the study's own frame98 columns on
    # the pick2 study artifact; way B = nightly champion path with
    # a reconstructed run-73 meta + the shadow seat artifact (the
    # same brain, seated) -- predictions must agree exactly.
    champ73 = joblib.load(ROOT / "data" / "muse_pick2" /
                          "stacker_pick2.pkl")
    X73a = frame.reindex(hold_idx)[COLS98]
    p73a = champ73.predict_proba(X73a)[:, 1]
    meta73 = {"feature_set": "pick2_98", "feature_columns": COLS98}
    X73b, tag73 = nightly.champion_holdout_frame(
        df, prod_sup_full.iloc[split:], meta73)
    assert tag73 == "pick2_98"
    seat73 = joblib.load(MODELS / "shadow_seat" / "stacker.pkl")
    p73b = seat73.predict_proba(X73b[COLS98])[:, 1]
    assert np.max(np.abs(p73a - p73b)) <= 1e-12, \
        f"run73 two-way proba diff {np.max(np.abs(p73a - p73b))}"
    assert (frame.reindex(hold_idx)[COLS98].to_numpy()
            == X73b[COLS98].to_numpy()).all()
    acc73 = float(accuracy_score(yhold, (p73a >= 0.5).astype(int)))
    prec73 = float(precision_score(yhold, (p73a >= 0.5).astype(int),
                                   zero_division=0))
    up73 = int((p73a >= 0.5).sum())

    # ---- control frame (ablation reporting only) ----
    Xc_tr = _frame_X(frame, prod_sup, X.index, COLS98)
    Xc_hold = _frame_X(frame, prod_sup, hold_idx, COLS98)
    cpipe = build_pick_pipeline(random_state=42)
    cpipe.fit(Xc_tr, y)
    pc = cpipe.predict_proba(Xc_hold)[:, 1]
    acc_c = float(accuracy_score(yhold, (pc >= 0.5).astype(int)))
    prec_c = float(precision_score(yhold, (pc >= 0.5).astype(int),
                                   zero_division=0))

    win = bool(acc > acc96 and acc > acc73)
    degenerate = bool(win and (up <= 7 or prec <= 0.05))
    eff_win = bool(win and not degenerate)
    votes = vote_histogram(proba)
    scaler = pipe.named_steps["scaler"]
    stack = pipe.named_steps["stack"]
    Xs = scaler.transform(Xhold)
    imps = {}
    for name, est in stack.named_estimators_.items():
        imp = np.asarray(est.feature_importances_, dtype=float)
        imps[name] = imp / imp.sum() if imp.sum() > 0 else imp
    agg_imp = np.mean([imps[k] for k in ("rf", "xgb", "lgbm", "et")],
                      axis=0)
    order = np.argsort(-agg_imp)
    top15 = [{"feature": cols[i], "importance": float(agg_imp[i])}
             for i in order[:15]]
    new_rank = {}
    for c in next_try_features.NEW_COLUMNS:
        if c in cols:
            new_rank[c] = int(np.where(order == cols.index(c))[0][0]) + 1
    save_result("final", {
        "selected_frame": results()["screen"]["selected"],
        "n_features": len(cols), "features": cols,
        "theta": theta, "theta_oof_acc": best_oof, "oof_acc_50": acc50,
        "acc": acc, "acc_at_50": acc_50, "precision": prec,
        "up_calls": up, "wf_acc": wf_acc, "wf_precision": wf_prec,
        "window_accs": win_accs, "majority_baseline": majority,
        "holdout_up_rate": float(yhold.mean()),
        "champion96": {"acc": acc96, "precision": prec96,
                       "up_calls": up96, "two_way_agree": True},
        "champion73": {"acc": acc73, "precision": prec73,
                       "up_calls": up73, "two_way_agree": True},
        "control98": {"acc": acc_c, "precision": prec_c},
        "win": win, "degenerate": degenerate, "effective_win": eff_win,
        "votes": votes, "top15": top15, "new_feature_ranks": new_rank,
        "meta_coef": {n: float(c) for n, c in zip(
            ("rf", "xgb", "lgbm", "et"),
            stack.final_estimator_.coef_[0])},
    })
    print(f"final: NEXT TRY acc={acc:.4f} (theta {theta:.2f}; @0.50 "
          f"{acc_50:.4f}) prec={prec:.4f} up={up}/792 wf={wf_acc:.4f} "
          f"windows={[round(a, 4) for a in win_accs]} | run96 "
          f"{acc96:.4f} | run73 {acc73:.4f} | control98 {acc_c:.4f} | "
          f"majority {majority:.4f} -> "
          f"{'WIN' if eff_win else 'NO WIN'}", flush=True)


def phase_register() -> None:
    res = results()
    guard_after = _guard()
    _assert_guard(res["meta"]["guard_before"], guard_after)
    fin = res["final"]
    win = bool(fin["effective_win"])
    c96, c73 = fin["champion96"], fin["champion73"]
    if win:
        reason = (
            f"Muse's Next Try stacker holdout acc {fin['acc']:.4f} "
            f"beat BOTH champions re-scored on the shared holdout "
            f"(run 96: {c96['acc']:.4f}; run 73 / Muse's Pick #2: "
            f"{c73['acc']:.4f}) — WOULD PROMOTE: held as PENDING KASS "
            f"APPROVAL under the promotion lock; champion kept "
            f"(study verdict only)")
    elif fin["win"] and fin["degenerate"]:
        reason = (
            f"Muse's Next Try stacker holdout acc {fin['acc']:.4f} "
            f"strictly beat both champions (run 96 {c96['acc']:.4f}, "
            f"run 73 {c73['acc']:.4f}) BUT anatomy is DEGENERATE "
            f"(up-calls {fin['up_calls']}/792, precision "
            f"{fin['precision']:.4f}) — counted as a LOSS under the "
            f"pre-registered run-63 rule; champion kept")
    else:
        reason = (
            f"Muse's Next Try stacker holdout acc {fin['acc']:.4f} "
            f"did not beat both champions re-scored on the shared "
            f"holdout (run 96 {c96['acc']:.4f}, run 73 "
            f"{c73['acc']:.4f}); champion kept (study verdict only)")
    metrics = {
        "challenger_acc": fin["acc"], "champion_acc": c96["acc"],
        "challenger_wf_acc": fin["wf_acc"],
        "challenger_wf_precision": fin["wf_precision"],
        "precision": fin["precision"], "folds": 5,
        "pending_approval": win, "reason": reason, "deep": False,
        "family": "muse_next_try",
        "votes": fin["votes"],
        "muse_next_try": {
            "label": "Muse's Next Try",
            "verdict": "PASS" if win else "FAIL",
            "design": DESIGN,
            "selected_frame": fin["selected_frame"],
            "features": fin["features"],
            "n_features": fin["n_features"],
            "theta": fin["theta"],
            "screen": res["screen"]["scores"],
            "champion73_acc": c73["acc"],
            "champion73_precision": c73["precision"],
            "control98_acc": fin["control98"]["acc"],
            "majority_baseline": fin["majority_baseline"],
            "window_accs": fin["window_accs"],
            "up_calls": fin["up_calls"],
            "new_feature_ranks": fin["new_feature_ranks"],
            "top_features": [t["feature"] for t in fin["top15"]],
        },
    }
    log_run("stacker", metrics, False)
    add_lab_note(
        "Muse's Next Try Stacker (family=muse_next_try)",
        f"{DESIGN}. Frame selected by train-CV only "
        f"({fin['selected_frame']}); threshold {fin['theta']:.2f} "
        f"from train OOF votes. Holdout acc {fin['acc']:.4f} vs "
        f"run 96 {c96['acc']:.4f} and run 73 (Muse's Pick #2) "
        f"{c73['acc']:.4f} re-scored on identical timestamps (two "
        f"ways each, agreeing); precision {fin['precision']:.4f}; "
        f"up-calls {fin['up_calls']}/792; walk-forward "
        f"{fin['wf_acc']:.4f}; control 98 eyes {fin['control98']['acc']:.4f}; "
        f"majority baseline {fin['majority_baseline']:.4f}. "
        f"{'WIN — PENDING KASS APPROVAL under the promotion lock.' if win else 'No win.'} "
        f"Nothing installed.")
    save_result("registered", {"win": win, "guard_after": guard_after})
    print(f"register: Muse's Next Try logged (promoted=0, "
          f"pending={win}); guard intact", flush=True)


PHASES = [("data", phase_data), ("external", phase_external),
          ("frames", phase_frames), ("screen", phase_screen),
          ("final", phase_final), ("register", phase_register)]


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    done = progress()
    print(f"=== Muse's Next Try study (phases done: {sorted(done)}) "
          f"===", flush=True)
    for name, fn in PHASES:
        if name in done:
            print(f"{name}: already done — skipped", flush=True)
            continue
        t0 = time.time()
        fn()
        mark(name)
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)
    print("=== Muse's Next Try study complete ===", flush=True)


if __name__ == "__main__":
    main()
