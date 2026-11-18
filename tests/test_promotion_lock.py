"""Promotion lock (Kass directive 2026-10-25): NO challenger -- PPO or
stacker, nightly or weekly deep -- advances to champion or reaches the
live trading dashboard without Kass's explicit agreement. Training and
grading are unchanged; a passing verdict is recorded as PENDING KASS
APPROVAL (registry flag + lab note) and the champion files are NOT
touched. Installs happen only on a run with the lock lifted.

These tests drive nightly.main() end to end with the model/data seams
stubbed (fetch, stacker builder, PPO trainer, duel evaluator) while the
real promotion machinery -- registry writes, lab notes, file
replacement -- runs against a tmp MODELS/DATA tree. The adversarial
core: with the lock ON, challengers that WIN everything must leave
every champion byte identical; with the lock OFF (a Kass-approved
install run), the same verdicts must install.

Run: .venv/bin/python -m pytest tests/test_promotion_lock.py -q
"""
import hashlib
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

from features.technical_indicators import make_supervised, feature_matrix  # noqa: E402
from models import stacker  # noqa: E402
from training import nightly  # noqa: E402


# ------------------------------------------------------------------ seams

def _frame(n=2600, seed=3):
    """Synthetic 1h OHLCV whose next-candle direction is learnable:
    move signs persist with p=0.88, so a stacker fit on the real labels
    scores ~0.85 holdout accuracy while one fit on shuffled labels
    sits at ~0.50."""
    rng = np.random.default_rng(seed)
    signs = np.ones(n)
    s = 1.0
    for t in range(n):
        if rng.random() < 0.12:
            s = -s
        signs[t] = s
    rets = signs * rng.uniform(0.001, 0.006, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, n))
    vol = rng.uniform(10, 100, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def _light_pipeline():
    """Cheap stand-in for the stacking pipeline: the lock tests grade
    promotion mechanics, not the stacker recipe, but the model must
    really learn the synthetic persistence (and really fail on
    shuffled labels) so the duel outcomes are data-driven."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=60, max_depth=6,
                                      n_jobs=-1, random_state=42)),
    ])


def _win(i, score, pnl, bench):
    return {"score": score, "total_pnl": pnl, "bench_pnl": bench,
            "max_drawdown": 100.0, "mean_drawdown": 50.0,
            "mean_reward": 0.1, "episodes": 10,
            "round_trips_per_day": 2.0,
            "action_mix": {"hold": 0.5, "buy": 0.2, "sell": 0.2,
                           "short": 0.1}}


def _eval_result(windows):
    return {"windows": windows,
            "agg_score": float(np.mean([w["score"] for w in windows])),
            "total_pnl": float(sum(w["total_pnl"] for w in windows)),
            "mean_reward": float(np.mean([w["mean_reward"]
                                          for w in windows]))}


CHAMP_RES = _eval_result([_win(0, -5.0, -200.0, 300.0),
                          _win(1, 0.0, -100.0, 300.0),
                          _win(2, 5.0, -200.0, 400.0)])
PASS_RES = _eval_result([_win(0, 10.0, 1000.0, 300.0),
                         _win(1, 20.0, 2000.0, 300.0),
                         _win(2, 30.0, 2000.0, 400.0)])
FAIL_RES = _eval_result([_win(0, -10.0, -900.0, 300.0),
                         _win(1, -20.0, -800.0, 300.0),
                         _win(2, -30.0, -800.0, 400.0)])


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _runs(data_dir: Path):
    con = sqlite3.connect(data_dir / "training.db")
    con.row_factory = sqlite3.Row
    out = [{"id": r["id"], "kind": r["kind"], "promoted": r["promoted"],
            "metrics": json.loads(r["metrics"])}
           for r in con.execute("SELECT * FROM runs ORDER BY id")]
    con.close()
    return out


def _setup(tmp_path, monkeypatch, *, stacker_challenger_good: bool,
           ppo_outcome: str, locked: bool, ppo_champion: bool = True):
    """Wire a full nightly.main() run into tmp MODELS/DATA with the
    seams stubbed. Returns (models_dir, data_dir, hashes_before)."""
    models = tmp_path / "models"
    data = tmp_path / "data"
    models.mkdir()
    data.mkdir()
    frame = _frame()

    monkeypatch.setattr(nightly, "MODELS", models)
    monkeypatch.setattr(nightly, "DATA", data)
    monkeypatch.setattr(nightly, "fetch_klines",
                        lambda *a, **k: frame.copy())
    monkeypatch.setattr(nightly, "promotion_locked", lambda: locked)
    monkeypatch.setattr(nightly, "crossfit_p_up",
                        lambda X, y, folds=4, **k: np.full(len(X), 0.5))
    monkeypatch.setattr(stacker, "build_pipeline",
                        lambda random_state=42: _light_pipeline())
    monkeypatch.setattr(stacker, "walk_forward_score",
                        lambda X, y, **k: {"accuracy": 0.5,
                                           "precision": 0.5, "folds": 5})

    # Champion stacker on disk: fit on the OPPOSITE label quality of
    # tonight's challenger, so the duel outcome is decided by the data.
    sup = make_supervised(frame.copy())
    split = len(sup) - 792
    Xtr, ytr = feature_matrix(sup.iloc[:split])
    rng = np.random.default_rng(7)
    y_champ = (ytr if not stacker_challenger_good
               else pd.Series(rng.permutation(ytr.to_numpy()),
                              index=ytr.index))
    champ_pipe = _light_pipeline().fit(Xtr, y_champ)
    joblib.dump(champ_pipe, models / "stacker_champion.pkl")
    (models / "stacker_champion.json").write_text(json.dumps(
        {"accuracy": 0.5, "precision": 0.5, "folds": 5,
         "model_path": str(models / "stacker_champion.pkl")}))

    def fake_train_and_save(X, y, out_path, random_state=42):
        yy = (y if stacker_challenger_good
              else pd.Series(rng.permutation(y.to_numpy()),
                             index=y.index))
        pipe = _light_pipeline().fit(X, yy)
        joblib.dump(pipe, out_path)
        return {"accuracy": 0.5, "precision": 0.5, "folds": 5,
                "model_path": str(out_path)}

    monkeypatch.setattr(stacker, "train_and_save", fake_train_and_save)

    # Since 2026-11-05 the nightly stacker challenger is the HA
    # recipe (nightly.train_ha_challenger); stub that seam with the
    # same controlled good/permuted-label semantics, fitted on the
    # HA frame it is handed.
    def fake_train_ha_challenger(X, y, out_path):
        yy = (y if stacker_challenger_good
              else pd.Series(rng.permutation(y.to_numpy()),
                             index=y.index))
        pipe = _light_pipeline().fit(X, yy)
        joblib.dump(pipe, out_path)
        return {"accuracy": 0.5, "precision": 0.5, "folds": 5,
                "model_path": str(out_path)}

    monkeypatch.setattr(nightly, "train_ha_challenger",
                        fake_train_ha_challenger)

    if ppo_champion:
        (models / "ppo_champion.zip").write_bytes(b"CHAMPION-PPO-BYTES")

    def fake_train_ppo(**kwargs):
        out = models / "ppo_challenger.zip"
        out.write_bytes(b"CHALLENGER-PPO-BYTES")
        (models / "ppo_challenger_curve.json").write_text(
            json.dumps({"timesteps": kwargs.get("timesteps", 0),
                        "curve": [[0, 0.0]]}))
        return out

    import models.ppo_train as ppo_train_mod
    monkeypatch.setattr(ppo_train_mod, "train", fake_train_ppo)

    def fake_evaluate_windows(path, windows, p_ups, **kwargs):
        if Path(path).name == "ppo_champion.zip":
            return CHAMP_RES
        return PASS_RES if ppo_outcome == "pass" else FAIL_RES

    monkeypatch.setattr(nightly, "evaluate_windows",
                        fake_evaluate_windows)

    hashes = {name: _sha(models / name)
              for name in ("ppo_champion.zip", "stacker_champion.pkl",
                           "stacker_champion.json")
              if (models / name).exists()}
    return models, data, hashes


# ------------------------------------------------------------------ tests

def test_lock_holds_winning_challengers(tmp_path, monkeypatch):
    """Locked: stacker wins its duel AND the PPO challenger passes
    formula + floor. NOTHING installs: champion bytes identical,
    challenger files unmoved, both runs recorded promoted=0 with
    pending_approval set, lab notes carry the pending marker."""
    models, data, before = _setup(
        tmp_path, monkeypatch, stacker_challenger_good=True,
        ppo_outcome="pass", locked=True)
    nightly.main()
    for name, h in before.items():
        assert _sha(models / name) == h, f"{name} changed under the lock"
    assert (models / "stacker_challenger.pkl").exists()
    assert (models / "ppo_challenger.zip").read_bytes() == \
        b"CHALLENGER-PPO-BYTES"
    runs = _runs(data)
    stk = next(r for r in runs if r["kind"] == "stacker")
    ppo = next(r for r in runs if r["kind"] == "ppo")
    assert stk["promoted"] == 0 and ppo["promoted"] == 0
    assert stk["metrics"]["pending_approval"] is True
    assert ppo["metrics"]["pending_approval"] is True
    notes = (data / "lab_notes.json").read_text()
    assert "PENDING KASS APPROVAL" in notes


def test_lock_failing_challengers_behave_as_before(tmp_path, monkeypatch):
    """Locked + losing challengers: the pre-lock behavior exactly --
    champion kept, promoted=0, NO pending flag, files untouched."""
    models, data, before = _setup(
        tmp_path, monkeypatch, stacker_challenger_good=False,
        ppo_outcome="fail", locked=True)
    nightly.main()
    for name, h in before.items():
        assert _sha(models / name) == h
    runs = _runs(data)
    stk = next(r for r in runs if r["kind"] == "stacker")
    ppo = next(r for r in runs if r["kind"] == "ppo")
    assert stk["promoted"] == 0 and ppo["promoted"] == 0
    assert not stk["metrics"].get("pending_approval")
    assert not ppo["metrics"].get("pending_approval")
    assert "did not beat" in stk["metrics"]["reason"]
    assert "champion kept" in ppo["metrics"]["reason"]


def test_lock_off_installs_winners(tmp_path, monkeypatch):
    """Lock lifted (a Kass-approved install run): the same winning
    verdicts install -- champion files are replaced by the
    challengers, runs record promoted=1, no pending flag."""
    models, data, before = _setup(
        tmp_path, monkeypatch, stacker_challenger_good=True,
        ppo_outcome="pass", locked=False)
    nightly.main()
    assert _sha(models / "stacker_champion.pkl") != \
        before["stacker_champion.pkl"]
    assert (models / "ppo_champion.zip").read_bytes() == \
        b"CHALLENGER-PPO-BYTES"
    assert not (models / "ppo_challenger.zip").exists()  # moved, not copied
    runs = _runs(data)
    stk = next(r for r in runs if r["kind"] == "stacker")
    ppo = next(r for r in runs if r["kind"] == "ppo")
    assert stk["promoted"] == 1 and ppo["promoted"] == 1
    assert not stk["metrics"].get("pending_approval")
    assert not ppo["metrics"].get("pending_approval")


def test_lock_holds_first_challenger_when_no_champion(tmp_path,
                                                      monkeypatch):
    """The no-champion bootstrap path is NOT a lock loophole: the
    first challenger is recorded pending, and no champion file is
    created."""
    models, data, _ = _setup(
        tmp_path, monkeypatch, stacker_challenger_good=False,
        ppo_outcome="pass", locked=True, ppo_champion=False)
    nightly.main()
    assert not (models / "ppo_champion.zip").exists()
    ppo = next(r for r in _runs(data) if r["kind"] == "ppo")
    assert ppo["promoted"] == 0
    assert ppo["metrics"]["pending_approval"] is True


def test_promotion_locked_default_and_env(monkeypatch):
    """The lock defaults ON and lifts only on TAM_PROMOTION_LOCK=0."""
    monkeypatch.delenv("TAM_PROMOTION_LOCK", raising=False)
    assert nightly.promotion_locked() is True
    monkeypatch.setenv("TAM_PROMOTION_LOCK", "0")
    assert nightly.promotion_locked() is False
    monkeypatch.setenv("TAM_PROMOTION_LOCK", "1")
    assert nightly.promotion_locked() is True


# ------------------------------------------------------- gate feed surface

def _run_row(rid, kind, promoted, metrics):
    return {"id": rid, "ts": "2026-10-25T00:00:00+00:00", "kind": kind,
            "promoted": promoted, "metrics": metrics}


def test_pending_promotion_derivation():
    """compact._pending_promotion: the newest unresolved pending
    record surfaces; a later recorded promotion of the same kind
    resolves it; kinds resolve independently."""
    from src.dashboards import compact

    pend = {"pending_approval": True, "reason": "won — WOULD PROMOTE"}
    plain = {"reason": "champion kept"}
    rows = [  # newest first, as _all_runs_desc returns
        _run_row(63, "ppo", False, pend),
        _run_row(62, "stacker", False, plain),
        _run_row(61, "ppo", True, plain),
    ]
    got = compact._pending_promotion(rows)
    assert got is not None and got["i"] == 63 and got["k"] == "ppo"

    resolved = [_run_row(64, "ppo", True, plain)] + rows
    assert compact._pending_promotion(resolved) is None

    # A stacker pending is not resolved by a PPO promotion.
    rows2 = [_run_row(65, "ppo", True, plain),
             _run_row(64, "stacker", False, pend),
             _run_row(63, "ppo", False, plain)]
    got2 = compact._pending_promotion(rows2)
    assert got2 is not None and got2["i"] == 64 and got2["k"] == "stacker"

    assert compact._pending_promotion(
        [_run_row(1, "ppo", False, plain)]) is None
    assert compact._pending_promotion([]) is None


def test_pending_promotion_declined_decision_resolves():
    """A pending record carrying Kass's recorded decision
    (metrics.pending_decision, e.g. "declined") no longer surfaces:
    the decision resolves the pending while the pending_approval
    flag itself stays in the registry as history. An undecided
    pending of another kind still surfaces."""
    from src.dashboards import compact

    pend = {"pending_approval": True, "reason": "won — WOULD PROMOTE"}
    declined = {**pend, "pending_decision": "declined",
                "pending_decided_by": "Kass",
                "pending_decision_reason": "keep the live test clean"}
    plain = {"reason": "champion kept"}
    rows = [  # newest first
        _run_row(62, "ppo", False, declined),
        _run_row(61, "stacker", False, pend),
        _run_row(60, "ppo", False, plain),
    ]
    # The declined PPO pending is resolved; the stacker pending,
    # still undecided, surfaces instead.
    got = compact._pending_promotion(rows)
    assert got is not None and got["i"] == 61 and got["k"] == "stacker"

    # Every pending decided -> nothing surfaces.
    rows[1] = _run_row(61, "stacker", False, declined)
    assert compact._pending_promotion(rows) is None


def test_gate_feed_carries_pending_key():
    """The gate feed always carries the pending field (null when no
    challenger awaits Kass), so the dashboards can render the banner
    the moment a run pends."""
    from src.dashboards import compact

    g = compact.build_gate()
    assert "pending" in g
    assert g["pending"] is None or (
        set(g["pending"]) == {"i", "k", "t", "why"}
        and g["pending"]["k"] in ("ppo", "stacker"))


def test_lab_notebook_never_truncates_history(tmp_path, monkeypatch):
    """The lab notebook is the paper trail the dashboards' name/version
    derivations read by exact timestamp. add_lab_note must never
    silently evict old notes: its 30-note cap rotated out the
    2026-10-21 "Features v5.1 migration" note, and run 16 (the v5.1
    stacker) lost its only "5.1" attestation -- the models index
    renamed it "Stacker v5" (2026-10-25, found when the scored-eyes
    registration append crossed the cap)."""
    monkeypatch.setattr(nightly, "DATA", tmp_path)
    for i in range(35):
        nightly.add_lab_note(f"note {i}", "detail")
    notes = json.loads((tmp_path / "lab_notes.json").read_text())
    assert len(notes) == 35
    assert notes[0]["title"] == "note 0"
    assert notes[-1]["title"] == "note 34"
