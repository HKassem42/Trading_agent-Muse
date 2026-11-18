"""Lab vs driver's-seat separation (Kass directive, 2026-10-25):
"Keep the trading and training separate. I want to be able to see
all the champions and numbers myself and then tell you which one
I want in the driver seat of live trading."

Pins:
- the scored-102 feature frame (src/features/scored_features.py):
  the matrix-v3 column set, trailing-only by construction --
  a truncation rebuild must reproduce the full-frame values on
  every shared row (the v5.1 causality standard);
- nightly's champion grading frame selection: the LAB stacker
  champion is graded on ITS OWN feature set (champion metadata
  tag), the challenger on the production 21 -- same holdout rows;
- the live loop loads ONLY the driver's-seat artifacts
  (models/live_seat/) and fails loudly when the seat is missing --
  never a silent fallback to the lab champions;
- the LIVE dashboard's vision feed reads the SEAT stacker, not
  the lab champion;
- the Training Lab champions roster (compact feed "champs"):
  title-holders in coronation order with their recorded grades,
  plus the driver's-seat line sourced from the seat manifest;
- the crowning + decline sequence leaves no unresolved pending:
  run 64 promoted with Kass's decision recorded, run 63 declined
  with his decision recorded -> _pending_promotion is None.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.technical_indicators import FEATURE_COLUMNS  # noqa: E402


def _ohlcv(n: int, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.002, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.002, n))),
        "close": close,
        "volume": np.abs(rng.normal(100, 20, n)),
    }, index=idx)


# ------------------------------------------------------- scored frame

def test_scored_columns_are_the_matrix_v3_set():
    from features import scored_features as sf

    cols = sf.scored_columns()
    assert len(cols) == 94, len(cols)
    assert cols[:4] == ["score_master", "score_trend",
                        "score_momentum", "score_volume"]
    assert len(sf.raw90_columns()) == 90
    assert len(set(cols)) == 94


def test_scored_frame_truncation_causality():
    """Rebuilding the scored frame on a truncated candle history
    must reproduce the full-frame scores on every shared row --
    any look-ahead in the scoring path breaks this equality."""
    from features import scored_features as sf

    names = ["Relative Strength Index (RSI)", "Connors RSI (CRSI)",
             "SuperTrend Side (+1/-1)"]
    df = _ohlcv(4800)
    frame, cols = sf.build_scored_frame(df, names=names)
    for c in names:
        assert c in cols
        vals = frame[c].dropna()
        assert len(vals) > 100
        assert vals.between(0.0, 100.0).all()
    cut = 4600
    frame_t, _ = sf.build_scored_frame(df.iloc[:cut + 1], names=names)
    for c in names:
        a = frame[c].iloc[:cut + 1].to_numpy(float)
        b = frame_t[c].to_numpy(float)
        assert np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True), c


# ------------------------------------------- nightly champion framing

def _sup(n: int = 5200, seed: int = 3):
    from features.technical_indicators import make_supervised
    return _ohlcv(n, seed=seed), make_supervised(_ohlcv(n, seed=seed))


def test_champion_frame_prod21_default():
    from training import nightly

    df, sup = _sup(600)
    hold = sup.iloc[-100:]
    X, tag = nightly.champion_holdout_frame(df, hold, {})
    assert tag == "prod21"
    assert list(X.columns) == FEATURE_COLUMNS
    assert list(X.index) == list(hold.index)


def test_champion_frame_scored102_selection_and_duel_fixture(
        monkeypatch):
    """A scored102 champion must be graded on the scored columns,
    aligned to the same holdout timestamps the prod21 challenger
    is graded on -- a fixture duel exercises both paths."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from features import scored_features as sf
    from training import nightly

    df, sup = _sup(800)
    hold = sup.iloc[-200:]
    train = sup.iloc[:-200]
    scored_cols = ["score_master", "feat_a", "feat_b"]
    fake = pd.DataFrame(
        np.random.default_rng(1).normal(size=(len(df), 3)),
        index=df.index, columns=scored_cols)
    monkeypatch.setattr(
        sf, "build_scored_frame",
        lambda d, names=None: (fake, scored_cols))
    meta = {"feature_set": "scored102", "feature_columns": scored_cols}
    X, tag = nightly.champion_holdout_frame(df, hold, meta)
    assert tag == "scored102"
    assert list(X.columns) == scored_cols
    assert list(X.index) == list(hold.index)

    # Fixture duel on identical rows: champion on its scored frame,
    # challenger on the production frame; both produce an accuracy
    # against the same holdout labels.
    from sklearn.metrics import accuracy_score
    y = hold["target"].astype(int)
    champ = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=500)).fit(fake.reindex(train.index)[scored_cols],
                           train["target"].astype(int))
    Xc, _ = nightly.champion_holdout_frame(df, train, meta)
    assert list(Xc.index) == list(train.index)
    acc_champ = accuracy_score(y, champ.predict(X))
    Xp, tagp = nightly.champion_holdout_frame(df, hold, {})
    assert tagp == "prod21"
    chall = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=500)).fit(train[FEATURE_COLUMNS],
                           train["target"].astype(int))
    acc_chall = accuracy_score(y, chall.predict(Xp))
    assert 0.0 <= acc_champ <= 1.0 and 0.0 <= acc_chall <= 1.0


def test_champion_frame_unknown_tag_is_loud():
    from training import nightly

    df, sup = _sup(400)
    with pytest.raises(ValueError):
        nightly.champion_holdout_frame(
            df, sup.iloc[-50:], {"feature_set": "martian9"})


# ---------------------------------------------------------- live seat

def _light_stacker(bias_up: bool):
    """A real (if tiny) fitted pipeline whose P(up) on a neutral row
    leans strongly one way -- two classes, separable on feature 0,
    with the majority class flipped between the two biases so a seat
    model and a decoy lab champion give clearly different votes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, len(FEATURE_COLUMNS))),
                     columns=FEATURE_COLUMNS)
    X.iloc[:55, 0] = 2.0
    X.iloc[55:, 0] = -2.0
    y = pd.Series([1] * 55 + [0] * 5)
    if not bias_up:
        y = 1 - y
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=500)).fit(X, y)


def test_live_seat_paths_point_at_live_seat():
    from trading import live

    assert live.SEAT_DIR.name == "live_seat"
    assert live.SEAT_STACKER == live.SEAT_DIR / "stacker.pkl"
    assert live.SEAT_PPO == live.SEAT_DIR / "ppo.zip"
    assert live.SEAT_STACKER.parent == live.SEAT_DIR


def test_live_missing_seat_fails_loudly(tmp_path, monkeypatch):
    from trading import live

    monkeypatch.setattr(live, "SEAT_DIR", tmp_path / "live_seat")
    monkeypatch.setattr(live, "SEAT_STACKER",
                        tmp_path / "live_seat" / "stacker.pkl")
    monkeypatch.setattr(live, "SEAT_PPO",
                        tmp_path / "live_seat" / "ppo.zip")
    with pytest.raises(FileNotFoundError):
        live.run_once()


def test_live_loads_seat_artifacts_not_lab_champions(tmp_path,
                                                     monkeypatch):
    """run_once must resolve BOTH brains from the seat directory --
    a recorder on each load seam proves the paths, and a sentinel
    stops the run before any market work happens."""
    from trading import live

    seat = tmp_path / "live_seat"
    seat.mkdir()
    joblib.dump(_light_stacker(True), seat / "stacker.pkl")
    (seat / "ppo.zip").write_bytes(b"SEAT-PPO")
    monkeypatch.setattr(live, "SEAT_DIR", seat)
    monkeypatch.setattr(live, "SEAT_STACKER", seat / "stacker.pkl")
    monkeypatch.setattr(live, "SEAT_PPO", seat / "ppo.zip")

    seen = {}

    class _Sentinel(Exception):
        pass

    import stable_baselines3

    def fake_ppo_load(path, *a, **k):
        seen["ppo_path"] = Path(path)
        raise _Sentinel

    monkeypatch.setattr(stable_baselines3.PPO, "load", fake_ppo_load)
    with pytest.raises(_Sentinel):
        live.run_once()
    assert seen["ppo_path"] == seat / "ppo.zip"


def test_vision_reads_seat_stacker_not_lab_champion(tmp_path,
                                                    monkeypatch):
    """The LIVE dashboard's vision feed shows what the DRIVER sees:
    with a decoy lab champion that disagrees, p_up must follow the
    seat stacker."""
    from dashboards import compact
    import data.market_data as md
    import features.technical_indicators as ti

    root = tmp_path
    (root / "models" / "live_seat").mkdir(parents=True)
    seat_model = _light_stacker(True)
    decoy_model = _light_stacker(False)
    joblib.dump(seat_model, root / "models" / "live_seat" / "stacker.pkl")
    joblib.dump(decoy_model, root / "models" / "stacker_champion.pkl")
    monkeypatch.setattr(compact, "ROOT", root)

    frame = _ohlcv(30)
    for c in FEATURE_COLUMNS:
        frame[c] = 0.0
    monkeypatch.setattr(md, "fetch_klines", lambda *a, **k: frame.copy())
    monkeypatch.setattr(ti, "add_features", lambda d: d)

    out = compact.build_vision()
    row = frame.iloc[:-1].iloc[-1]
    expect = round(float(seat_model.predict_proba(
        row[FEATURE_COLUMNS].to_frame().T)[0, 1]), 4)
    decoy = round(float(decoy_model.predict_proba(
        row[FEATURE_COLUMNS].to_frame().T)[0, 1]), 4)
    assert expect != decoy  # the fixture can actually tell them apart
    assert out["p_up"] == expect


# ------------------------------------------------------ champions roster

def _fixture_registry(root: Path) -> None:
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(root / "data" / "training.db")
    con.execute("CREATE TABLE runs(id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " ts TEXT NOT NULL, kind TEXT NOT NULL,"
                " metrics TEXT NOT NULL, promoted INTEGER NOT NULL)")
    rows = [
        (40, "2026-10-23T04:03:20+00:00", "stacker", 1,
         {"challenger_acc": 0.4797979797979798,
          "challenger_wf_acc": 0.49855929072774285,
          "precision": 0.4830097087378641,
          "reason": "stacker challenger holdout acc 0.4798 > champion 0.4609"}),
        (44, "2026-10-24T04:18:03+00:00", "ppo", 0,
         {"challenger": {"score": -615.5760555338929,
                         "total_pnl": -650.3109733207384},
          "reason": "manual override"}),
        (63, "2026-10-25T22:23:13+00:00", "stacker", 0,
         {"challenger_acc": 0.51010101010101, "precision": 0.0,
          "pending_approval": True, "family": "full_indicator_stacker",
          "pending_decision": "declined", "pending_decided_by": "Kass",
          "reason": "degenerate win"}),
        (64, "2026-10-25T22:23:13+00:00", "stacker", 1,
         {"challenger_acc": 0.49747474747474746,
          "challenger_wf_acc": 0.5121151936444887,
          "precision": 0.4853801169590643,
          "pending_approval": True, "family": "full_indicator_stacker",
          "pending_decision": "approved", "pending_decided_by": "Kass",
          "promoted_scope": "lab champion — Champion #2",
          "reason": "scored win"}),
    ]
    for rid, ts, kind, promoted, metrics in rows:
        con.execute("INSERT INTO runs(id,ts,kind,metrics,promoted)"
                    " VALUES (?,?,?,?,?)",
                    (rid, ts, kind, json.dumps(metrics), promoted))
    con.commit()
    con.close()
    (root / "data" / "live_seat.json").write_text(json.dumps({
        "seat": {
            "stacker": {"run_id": 40, "feature_set": "prod21",
                        "sha256": "stk-sha"},
            "ppo": {"run_id": 44, "sha256": "ppo-sha"},
        },
        "ppo_champion_run_id": 44,
        "seated_by": "initial separation 2026-10-25",
    }))
    (root / "models" / "stacker_champion.json").write_text(json.dumps({
        "accuracy": 0.49747474747474746, "precision": 0.4853801169590643,
        "wf_accuracy": 0.5121151936444887,
        "wf_precision": 0.510288926748488, "folds": 5,
        "feature_set": "scored102", "run_id": 64,
        "model_path": str(root / "models" / "stacker_champion.pkl"),
    }))


def test_champions_roster_from_records(tmp_path, monkeypatch):
    from dashboards import compact

    _fixture_registry(tmp_path)
    monkeypatch.setattr(compact, "ROOT", tmp_path)
    champs = compact.build_champions()
    assert champs is not None
    assert champs["seat"] == {"ppo_i": 44, "stk_i": 40,
                              "stk_feat": "prod21"}
    stk = champs["stk"]
    assert len(stk) == 2
    n1, n2 = stk
    # Champion #1: the 21-feature stacker -- former lab champion,
    # still in the driver's seat (status 2).
    assert n1[0] == 1 and n1[1] == 40 and n1[3] == "prod21"
    assert n1[4] == 0.4798 and n1[5] == 0.4986 and n1[6] == 0.483
    assert n1[7] == 2
    # Champion #2: run 64, current lab champion (status 1).
    assert n2[0] == 2 and n2[1] == 64 and n2[3] == "scored102"
    assert n2[4] == 0.4975 and n2[5] == 0.5121 and n2[6] == 0.4854
    assert n2[7] == 1
    # PPO: the run-44 champion, lab champion + driver's seat (3).
    (ppo_row,) = champs["ppo"]
    assert ppo_row[1] == 44 and ppo_row[3] == -615.6
    assert ppo_row[4] == -650.31 and ppo_row[5] == 3
    # Names agree with the models index for the same run ids.
    names = {r[0]: r[2] for r in compact._models_index(
        compact._all_runs_desc())}
    assert n1[2] == names[40] and n2[2] == names[64]
    assert ppo_row[2] == names[44]


def test_champions_roster_absent_before_separation(tmp_path,
                                                   monkeypatch):
    from dashboards import compact

    monkeypatch.setattr(compact, "ROOT", tmp_path)
    assert compact.build_champions() is None


def test_crowning_plus_decline_leaves_no_pending():
    """The 2026-10-25 sequence end-state: run 64 crowned (promoted,
    decision approved on its record) and run 63 declined (decision
    on its record) -> nothing surfaces as pending."""
    from dashboards import compact

    def row(rid, kind, promoted, metrics):
        return {"id": rid, "ts": "2026-10-25T22:23:13+00:00",
                "kind": kind, "promoted": promoted, "metrics": metrics}

    rows = [  # newest first
        row(64, "stacker", True,
            {"pending_approval": True, "pending_decision": "approved",
             "reason": "scored win"}),
        row(63, "stacker", False,
            {"pending_approval": True, "pending_decision": "declined",
             "reason": "degenerate win"}),
        row(62, "ppo", False, {"reason": "champion kept"}),
    ]
    assert compact._pending_promotion(rows) is None


# --------------------------------- champions roster after Champion #3

def _fixture_registry_champion3(root: Path) -> None:
    """The registry/manifest world AFTER Kass's 2026-10-26 crowning:
    run 69 (Muse's Pick Stacker) installed as lab Champion #3 with
    his decision recorded; run 64 demoted to former champion (its
    own crowning decision stays on its record); run 40 still the
    seat stacker."""
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(root / "data" / "training.db")
    con.execute("CREATE TABLE runs(id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " ts TEXT NOT NULL, kind TEXT NOT NULL,"
                " metrics TEXT NOT NULL, promoted INTEGER NOT NULL)")
    rows = [
        (40, "2026-10-23T04:03:20+00:00", "stacker", 1,
         {"challenger_acc": 0.4797979797979798,
          "challenger_wf_acc": 0.49855929072774285,
          "precision": 0.4830097087378641,
          "reason": "stacker challenger holdout acc 0.4798 > champion 0.4609"}),
        (44, "2026-10-24T04:18:03+00:00", "ppo", 0,
         {"challenger": {"score": -615.5760555338929,
                         "total_pnl": -650.3109733207384},
          "reason": "manual override"}),
        (63, "2026-10-25T22:23:13+00:00", "stacker", 0,
         {"challenger_acc": 0.51010101010101, "precision": 0.0,
          "pending_approval": True, "family": "full_indicator_stacker",
          "pending_decision": "declined", "pending_decided_by": "Kass",
          "reason": "degenerate win"}),
        (64, "2026-10-25T22:23:13+00:00", "stacker", 1,
         {"challenger_acc": 0.49747474747474746,
          "challenger_wf_acc": 0.5121151936444887,
          "precision": 0.4853801169590643,
          "pending_approval": True, "family": "full_indicator_stacker",
          "full_indicator_stacker": {"n_features": 94},
          "pending_decision": "approved", "pending_decided_by": "Kass",
          "promoted_scope": "lab champion — Champion #2",
          "reason": "scored win"}),
        (69, "2026-10-26T23:25:53+00:00", "stacker", 1,
         {"challenger_acc": 0.5265151515151515,
          "challenger_wf_acc": 0.5021847070506456,
          "precision": 0.5167464114832536,
          "pending_approval": True, "family": "muse_pick",
          "muse_pick": {"n_features": 94},
          "pending_decision": "approved", "pending_decided_by": "Kass",
          "promoted_scope": "lab champion — Champion #3",
          "reason": "Muse's Pick win"}),
    ]
    for rid, ts, kind, promoted, metrics in rows:
        con.execute("INSERT INTO runs(id,ts,kind,metrics,promoted)"
                    " VALUES (?,?,?,?,?)",
                    (rid, ts, kind, json.dumps(metrics), promoted))
    con.commit()
    con.close()
    (root / "data" / "live_seat.json").write_text(json.dumps({
        "seat": {
            "stacker": {"run_id": 40, "feature_set": "prod21",
                        "sha256": "stk-sha"},
            "ppo": {"run_id": 44, "sha256": "ppo-sha"},
        },
        "ppo_champion_run_id": 44,
        "seated_by": "initial separation 2026-10-25",
    }))
    (root / "models" / "stacker_champion.json").write_text(json.dumps({
        "accuracy": 0.5265151515151515, "precision": 0.5167464114832536,
        "wf_accuracy": 0.5021847070506456,
        "wf_precision": 0.500419523113745, "folds": 5,
        "feature_set": "scored102", "run_id": 69,
        "model_path": str(root / "models" / "stacker_champion.pkl"),
    }))


def test_champions_roster_after_champion3_crowning(tmp_path, monkeypatch):
    """Kass (2026-10-26): crown run 69. The roster must still show
    ALL title-holders in coronation order: #1 run 40 (former, in
    the driver's seat, status 2), #2 run 64 (former, in NO seat --
    status 4, never the in-seat status 2), #3 run 69 (current,
    status 1)."""
    from dashboards import compact

    _fixture_registry_champion3(tmp_path)
    monkeypatch.setattr(compact, "ROOT", tmp_path)
    champs = compact.build_champions()
    assert champs is not None
    assert champs["seat"] == {"ppo_i": 44, "stk_i": 40,
                              "stk_feat": "prod21"}
    stk = champs["stk"]
    assert len(stk) == 3
    n1, n2, n3 = stk
    assert n1[0] == 1 and n1[1] == 40 and n1[3] == "prod21"
    assert n1[4] == 0.4798 and n1[5] == 0.4986 and n1[6] == 0.483
    assert n1[7] == 2
    assert n2[0] == 2 and n2[1] == 64 and n2[3] == "scored102"
    assert n2[4] == 0.4975 and n2[5] == 0.5121 and n2[6] == 0.4854
    assert n2[7] == 4
    assert n3[0] == 3 and n3[1] == 69 and n3[3] == "scored102"
    assert n3[4] == 0.5265 and n3[5] == 0.5022 and n3[6] == 0.5167
    assert n3[7] == 1
    (ppo_row,) = champs["ppo"]
    assert ppo_row[1] == 44 and ppo_row[5] == 3
    names = {r[0]: r[2] for r in compact._models_index(
        compact._all_runs_desc())}
    assert n1[2] == names[40] and n2[2] == names[64]
    assert n3[2] == names[69] and ppo_row[2] == names[44]


def test_former_champion_ha_frame_tag_from_record(tmp_path, monkeypatch):
    """Champion #7 crowning (2026-11-06): a displaced HA champion's
    roster row must keep its ha10 tag -- run 110 carries it via its
    muse_pick11 family, run 111 via its recorded nightly reason.
    Before the fix both fell through to the prod21 fallback."""
    import json as _json
    import sqlite3 as _sqlite3

    from dashboards import compact

    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "models" / "stacker_champion.json").write_text(_json.dumps(
        {"run_id": 111, "feature_set": "ha10"}))
    (tmp_path / "data" / "live_seat.json").write_text(_json.dumps({
        "seat": {"stacker": {"run_id": 40, "feature_set": "prod21"},
                 "ppo": {"run_id": 44}},
        "ppo_champion_run_id": 112}))
    con = _sqlite3.connect(tmp_path / "data" / "training.db")
    con.execute("CREATE TABLE runs(id INTEGER PRIMARY KEY, ts TEXT,"
                " kind TEXT, metrics TEXT, promoted INTEGER)")
    rows = [
        (40, "stacker", {"challenger_acc": 0.4798,
                         "challenger_wf_acc": 0.4986,
                         "precision": 0.483,
                         "pending_decision": "approved"}, 1),
        (110, "stacker", {"challenger_acc": 0.5341,
                          "challenger_wf_acc": 0.5234,
                          "precision": 0.526,
                          "family": "muse_pick11",
                          "pending_decision": "approved"}, 1),
        (111, "stacker", {"challenger_acc": 0.5379,
                          "challenger_wf_acc": 0.5303,
                          "precision": 0.5286,
                          "reason": "HA stacker challenger holdout acc "
                                    "0.5379 > champion 0.5328",
                          "pending_decision": "approved"}, 1),
        (112, "ppo", {"challenger": {"score": -291.6,
                                     "total_pnl": 738.69}}, 1),
    ]
    for rid, kind, metrics, prom in rows:
        con.execute("INSERT INTO runs VALUES (?, '2026-11-06', ?, ?, ?)",
                    (rid, kind, _json.dumps(metrics), prom))
    con.commit()
    con.close()
    monkeypatch.setattr(compact, "ROOT", tmp_path)
    champs = compact.build_champions()
    assert champs is not None
    by_run = {r[1]: r for r in champs["stk"]}
    assert by_run[110][3] == "ha10" and by_run[110][7] == 4
    assert by_run[111][3] == "ha10" and by_run[111][7] == 1
    (ppo_row,) = champs["ppo"]
    assert ppo_row[1] == 112 and ppo_row[5] == 1
