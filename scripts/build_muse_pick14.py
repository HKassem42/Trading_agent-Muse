"""Muse's Pick #14 — "The Committee" — LAB study (2026-11-18,
Kass's dealer's-choice order: "run your next muse pick on
whatever you want... let's try to beat everything we have ever
done").

Pre-registered in refs/muse-pick-14-preregistration-2026-11-18.md
BEFORE any candidate was fitted. The candidate is the exact
Pick recipe (build_pick_pipeline) on the exact ha10 frame, with
the single seed 42 replaced by a committee of 7 fixed seeds
[42, 7, 123, 2024, 555, 9001, 31337] whose meta-probabilities
are averaged (models.muse_pick14). Graded on Picks #9-#13's
IDENTICAL exam (last 792 supervised rows from 2026-10-03 19:00
UTC, majority 0.5088).

Setup stage (before any candidate fit): the rebuilt ha10 frame
must equal Pick #10's saved matrix exactly, Pick #11's artifact
must reproduce 0.5341, run 126 must re-score 0.5581 and run 123
must re-score 0.5518 on the exam — otherwise the study aborts.

Bar (frozen): PASS only if exam acc > run 126 re-scored
(0.5581) AND purged walk-forward acc >= 0.5234 (Pick #11's
recorded WF; 5 folds, 24-row embargo, full committee refit per
fold). On PASS the run registers pending_approval (WOULD
PROMOTE) and the PPO phase (scripts/build_muse_pick14_ppo.py)
runs; the single lab note is then appended by that script so
it can cover the PPO outcome too. On FAIL the note is appended
here and nothing changes. Staged and resumable; guards checked
before registration. LAB ONLY — no champion file, seat, book,
or live file is ever touched.

Usage: .venv/bin/python scripts/build_muse_pick14.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.muse_pick7 import anatomy, purged_wf_splits  # noqa: E402
from models.muse_pick10 import HA_COLUMNS, build_ha_matrix  # noqa: E402
from models.muse_pick14 import (SEEDS, CommitteeStacker,  # noqa: E402
                                fit_committee, fit_seed_pipeline)
from training.nightly import add_lab_note, log_run, vote_histogram  # noqa: E402

warnings.filterwarnings("ignore")

MODELS = ROOT / "models"
LAB9 = ROOT / "data" / "muse_pick9"
LAB10 = ROOT / "data" / "muse_pick10"
LAB11 = ROOT / "data" / "muse_pick11"
LAB = ROOT / "data" / "muse_pick14"
PROGRESS = LAB / "progress.json"
HOLDOUT = 792
FAMILY = "muse_pick14"
LABEL = "Stacker — The Committee (7-Seed Average)"
DESIGN = ("Muse's Pick #14 'The Committee': the standard Pick "
          "recipe (RF+XGB+LGBM+ET -> LR, SMOTE) on the 10 HA "
          "candle values, run at 7 fixed seeds [42, 7, 123, "
          "2024, 555, 9001, 31337] with the seeds' "
          "meta-probabilities averaged — the single-seed recipe "
          "replaced by a committee to average out the seed "
          "noise that decides the nightly exams by 2-4 "
          "candles — graded on Picks #9-#13's identical exam "
          "with a purged 5-fold walk-forward")
PICK11_ACC = 0.5340909090909091
WF_BAR = 0.5234  # Pick #11's recorded WF (0.5234264432029796)
GUARD_FILES = {
    "lab_ppo": MODELS / "ppo_champion.zip",
    "lab_stacker": MODELS / "stacker_champion.pkl",
    "lab_stacker_meta": MODELS / "stacker_champion.json",
    "seat_stacker": MODELS / "live_seat" / "stacker.pkl",
    "seat_ppo": MODELS / "live_seat" / "ppo.zip",
    "shadow_stacker": MODELS / "shadow_seat" / "stacker.pkl",
    "shadow_ppo": MODELS / "shadow_seat" / "ppo.zip",
    "gated_stacker": MODELS / "gated_seat" / "stacker.pkl",
    "champ_stacker": MODELS / "champ_seat" / "stacker.pkl",
    "champ_ppo": MODELS / "champ_seat" / "ppo.zip",
    "run73_artifact": ROOT / "data" / "muse_pick2" / "stacker_pick2.pkl",
}


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _guard():
    return {k: _sha(p) for k, p in GUARD_FILES.items()}


def _done() -> list:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text()).get("done", [])
    return []


def _mark(stage: str) -> None:
    done = _done()
    if stage not in done:
        done.append(stage)
    LAB.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps({"done": done}, indent=1))


def _say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _frames():
    raw = pd.read_pickle(LAB9 / "raw_ohlcv.pkl")
    sup = pd.read_pickle(LAB9 / "prod_sup.pkl")
    X = build_ha_matrix(raw).loc[sup.index]
    y = sup["target"].astype(int).to_numpy()
    return X, y, sup


def _score(pkl_path, X, y, split) -> dict:
    pipe = joblib.load(pkl_path)
    proba = pipe.predict_proba(X.iloc[split:])[:, 1]
    return anatomy(proba, y[split:])


# ------------------------------------------------------------- stages
def stage_setup() -> dict:
    meta9 = json.loads((LAB9 / "meta.json").read_text())
    assert meta9["holdout_start"] == "2026-10-03 19:00:00"
    split = meta9["split"]
    X, y, _ = _frames()
    assert list(X.columns) == HA_COLUMNS
    assert X.shape == (17050, 10), X.shape
    assert not X.isna().any().any(), "ha10 frame has NaNs"
    # Frame identity: rebuilt ha10 == Pick #10's saved matrix.
    saved10 = pd.read_pickle(LAB10 / "ha_matrix.pkl")
    pd.testing.assert_frame_equal(X, saved10)
    # Pick #11's artifact must reproduce its print on this exam.
    m110 = _score(LAB11 / "pick11_stacker.pkl", X, y, split)
    assert abs(m110["acc"] - PICK11_ACC) < 1e-12, m110
    # Champion re-scores on the identical exam rows.
    champ_now = json.loads((MODELS / "stacker_champion.json")
                           .read_text())
    assert champ_now["run_id"] == 126, champ_now["run_id"]
    m126 = _score(MODELS / "stacker_champion.pkl", X, y, split)
    assert abs(m126["acc"] - 0.5580808080808081) < 1e-9, m126
    arch = sorted((MODELS / "archive").glob(
        "*_pre_champion13/stacker_champion.pkl"))
    if len(arch) == 1:
        m123 = _score(arch[0], X, y, split)
        assert abs(m123["acc"] - 0.5518) < 5e-4, m123
        champ123_source = "live re-score of the archived artifact"
    else:
        # DEVIATION (reported to the parent agent): run 123's
        # archived artifact was destroyed by archive_champions'
        # newest-5 retention between Pick #13 (2026-11-12, whose
        # setup asserted this same glob == 1) and the 2026-11-15
        # retrain; no copy survives in the tree or the GitHub
        # full-backup. The exam itself is proven intact by the
        # three exact checks above (frame identity vs Pick
        # #10's saved matrix, run 110 reproducing 0.5341 to
        # 1e-12, run 126 re-scoring 0.5581 to 1e-9), and the
        # frozen bar never references run 123 — its re-score is
        # carried from Pick #13's verified setup record on this
        # byte-identical exam.
        m123 = json.loads((ROOT / "data" / "muse_pick13"
                           / "meta.json").read_text()
                          )["champ123_rescore"]
        assert abs(m123["acc"] - 0.5518) < 5e-4, m123
        champ123_source = ("pick13 recorded re-score (artifact "
                           "pruned by archive retention)")
    meta = {"guard_before": _guard(), "split": split,
            "exam_majority": meta9["exam_majority"],
            "holdout_start": meta9["holdout_start"],
            "rows_sup": int(len(y)),
            "pick11_acc": PICK11_ACC, "pick11_rescore": m110,
            "champ126_rescore": m126, "champ123_rescore": m123,
            "champ123_rescore_source": champ123_source}
    (LAB / "meta.json").write_text(json.dumps(meta, indent=1))
    _say(f"setup: ha10 frame {X.shape} on the identical exam "
         f"(majority {meta['exam_majority']:.4f}); frame identity "
         f"OK (run 110 reproduces {m110['acc']:.4f}); re-scores: "
         f"run 126 {m126['acc']:.4f} (prec {m126['precision']:.4f}, "
         f"up {m126['up_calls']}), run 123 {m123['acc']:.4f}")
    return meta


def stage_fit_seeds(meta) -> dict:
    X, y, _ = _frames()
    split = meta["split"]
    out = {}
    for s in SEEDS:
        t0 = time.time()
        pipe = fit_seed_pipeline(X.iloc[:split], y[:split], s)
        proba = pipe.predict_proba(X.iloc[split:])[:, 1]
        np.save(LAB / f"proba_seed_{s}.npy", proba)
        joblib.dump(pipe, LAB / f"seed_{s}_stacker.pkl")
        m = anatomy(proba, y[split:])
        out[str(s)] = m
        _say(f"seed {s}: exam acc {m['acc']:.4f} prec "
             f"{m['precision']:.4f} up {m['up_calls']}/{HOLDOUT} "
             f"({time.time() - t0:.0f}s)")
    (LAB / "seeds.json").write_text(json.dumps(out, indent=1))
    accs = [out[str(s)]["acc"] for s in SEEDS]
    _say(f"seeds: min {min(accs):.4f} max {max(accs):.4f} "
         f"mean {float(np.mean(accs)):.4f} — the noise band")
    return out


def stage_committee(meta) -> dict:
    X, y, _ = _frames()
    split = meta["split"]
    pipes = [joblib.load(LAB / f"seed_{s}_stacker.pkl") for s in SEEDS]
    committee = CommitteeStacker(pipes, seeds=list(SEEDS),
                                 feature_columns=list(X.columns))
    joblib.dump(committee, LAB / "pick14_stacker.pkl")
    proba = committee.predict_proba(X.iloc[split:])[:, 1]
    seed_probas = np.vstack(
        [np.load(LAB / f"proba_seed_{s}.npy") for s in SEEDS])
    assert np.allclose(proba, seed_probas.mean(axis=0),
                       rtol=0, atol=1e-12), \
        "committee proba != mean of seed probas"
    np.save(LAB / "proba_pick14.npy", proba)
    m = anatomy(proba, y[split:])
    _say(f"committee: exam acc {m['acc']:.4f} prec "
         f"{m['precision']:.4f} up {m['up_calls']}/{HOLDOUT} vs "
         f"bar: run 126 re-scored "
         f"{meta['champ126_rescore']['acc']:.4f}")
    return m


def stage_wf(meta) -> dict:
    X, y, _ = _frames()
    split = meta["split"]
    Xtr = X.iloc[:split]
    ytr = y[:split]
    folds = purged_wf_splits(split, horizon=24, n_splits=5)
    fold_accs, fold_majs, proba_all = [], [], np.full(split, np.nan)
    for k, (tr, va) in enumerate(folds):
        t0 = time.time()
        committee = fit_committee(Xtr.iloc[tr], ytr[tr])
        p = committee.predict_proba(Xtr.iloc[va])[:, 1]
        proba_all[va] = p
        m = anatomy(p, ytr[va])
        maj = float(max(ytr[va].mean(), 1 - ytr[va].mean()))
        fold_accs.append(m["acc"])
        fold_majs.append(maj)
        _say(f"wf fold {k}: acc {m['acc']:.4f} vs majority "
             f"{maj:.4f} ({time.time() - t0:.0f}s)")
    pooled = anatomy(proba_all[np.isfinite(proba_all)],
                     ytr[np.isfinite(proba_all)])
    out = {"fold_accs": fold_accs, "fold_majorities": fold_majs,
           "wf_acc": float(np.mean(fold_accs)),
           "wf_majority": float(np.mean(fold_majs)),
           "wf_precision_pooled": pooled["precision"]}
    (LAB / "wf.json").write_text(json.dumps(out, indent=1))
    _say(f"wf: mean acc {out['wf_acc']:.4f} vs majority "
         f"{out['wf_majority']:.4f} (bar {WF_BAR:.4f})")
    return out


def _note_text(meta, m, wf, seed_accs, pending) -> str:
    a126 = meta["champ126_rescore"]["acc"]
    a123 = meta["champ123_rescore"]["acc"]
    return (
        f"{DESIGN}. Setup verified on the identical Picks "
        f"#9-#13 exam (frame identity OK; run 110 reproduces "
        f"{PICK11_ACC:.4f}; run 126 re-scored {a126:.4f}; run "
        f"123 re-scored {a123:.4f} "
        f"[{meta.get('champ123_rescore_source', 'live re-score')}]; "
        f"majority {meta['exam_majority']:.4f}). Per-seed exam accs: "
        f"{ {s: round(a, 4) for s, a in seed_accs.items()} } "
        f"(band {min(seed_accs.values()):.4f}-"
        f"{max(seed_accs.values()):.4f}). Committee exam acc "
        f"{m['acc']:.4f}, precision {m['precision']:.4f}, "
        f"up-calls {m['up_calls']}/{HOLDOUT}; purged "
        f"walk-forward {wf['wf_acc']:.4f} vs the {WF_BAR:.4f} "
        f"bar (folds {['%.4f' % a for a in wf['fold_accs']]}). "
        + ("PASS under the frozen bar — WOULD PROMOTE; crowning "
           "ruled on by Muse under Kass's 2026-11-12 standing "
           "delegation. "
           if pending else
           "FAIL under the frozen bar — Champion 126 keeps "
           "retraining; nothing changes. ")
        + "Nothing installed by the study itself.")


def stage_register(meta, cand, wf) -> dict:
    _, y, _ = _frames()
    split = meta["split"]
    proba = np.load(LAB / "proba_pick14.npy")
    m = anatomy(proba, y[split:])  # grade recomputed, second way
    assert abs(m["acc"] - cand["acc"]) < 1e-12
    a126 = meta["champ126_rescore"]["acc"]
    a123 = meta["champ123_rescore"]["acc"]
    seeds_meta = json.loads((LAB / "seeds.json").read_text())
    seed_accs = {s: seeds_meta[str(s)]["acc"] for s in SEEDS}
    exam_ok = m["acc"] > a126
    wf_ok = wf["wf_acc"] >= WF_BAR
    pending = bool(exam_ok and wf_ok)
    pred = (proba >= 0.5).astype(int)
    ey = y[split:]
    w = len(ey) // 3
    window_accs = [float((pred[k * w:(k + 1) * w]
                          == ey[k * w:(k + 1) * w]).mean())
                   for k in range(3)]
    guard_after = _guard()
    assert guard_after == meta["guard_before"], "GUARD TRIPPED"
    base = (f"Muse's Pick #14 committee holdout acc {m['acc']:.4f} "
            f"on the identical exam (run 126 re-scored {a126:.4f}; "
            f"run 123 re-scored {a123:.4f}; Pick #11 "
            f"{PICK11_ACC:.4f}; majority "
            f"{meta['exam_majority']:.4f}); per-seed accs "
            f"{[round(seed_accs[s], 4) for s in SEEDS]}")
    if pending:
        reason = (f"{base} with purged walk-forward "
                  f"{wf['wf_acc']:.4f} >= the {WF_BAR:.4f} bar — "
                  f"WOULD PROMOTE: held as PENDING APPROVAL; "
                  f"crowning ruled on by Muse under Kass's "
                  f"2026-11-12 standing delegation")
    elif not exam_ok:
        reason = (f"{base} — did not beat the re-scored champion "
                  f"on the exam; FAIL under the pre-registered "
                  f"bar. Champion 126 keeps retraining; nothing "
                  f"changes")
    else:
        reason = (f"{base} — beat the re-scored champion on the "
                  f"exam BUT its purged walk-forward "
                  f"{wf['wf_acc']:.4f} fell under the {WF_BAR:.4f} "
                  f"bar; FAIL under the pre-registered bar — the "
                  f"committee may not buy exam points by giving "
                  f"back walk-forward. Champion kept")
    metrics = {
        "challenger_acc": m["acc"], "champion_acc": a126,
        "challenger_wf_acc": wf["wf_acc"],
        "challenger_wf_precision": wf["wf_precision_pooled"],
        "precision": m["precision"], "folds": 5,
        "pending_approval": pending,
        "reason": reason, "deep": False, "family": FAMILY,
        "feature_set": "ha10ens",
        "votes": vote_histogram(proba),
        FAMILY: {
            "label": LABEL, "verdict": "PASS" if pending else "FAIL",
            "design": DESIGN,
            "champ126_rescore_acc": a126,
            "champ123_rescore_acc": a123,
            "pick11_acc": PICK11_ACC,
            "majority_baseline": meta["exam_majority"],
            "exam_rows": int(len(ey)), "horizon": 1,
            "window_accs": window_accs,
            "up_calls": m["up_calls"],
            "wf_majority": wf["wf_majority"],
            "wf_fold_accs": wf["fold_accs"],
            "seeds": SEEDS,
            "seed_accs": {str(s): seed_accs[s] for s in SEEDS},
            "wf_bar": WF_BAR,
        },
    }
    log_run("stacker", metrics, False)
    import sqlite3
    with sqlite3.connect(ROOT / "data" / "training.db") as con:
        run_id = con.execute(
            "SELECT MAX(id) FROM runs").fetchone()[0]
    note_appended = False
    if not pending:
        # FAIL is final here: the single study note is appended
        # now. On PASS the note is deferred to the PPO script so
        # one note covers the whole study (pre-registration).
        add_lab_note(f"Muse's Pick #14 (family={FAMILY})",
                     _note_text(meta, m, wf, seed_accs, pending))
        note_appended = True
    out = {"pick": m, "wf": wf, "pending": pending, "meta": meta,
           "run_id": run_id, "note_appended": note_appended,
           "seed_accs": {str(s): seed_accs[s] for s in SEEDS}}
    (LAB / "results.json").write_text(json.dumps(out, indent=1))
    _say(f"registered run {run_id}: acc {m['acc']:.4f} — "
         f"{'PASS (pending approval)' if pending else 'FAIL'}")
    return out


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    done = _done()
    if "setup" not in done:
        meta = stage_setup()
        _mark("setup")
    else:
        meta = json.loads((LAB / "meta.json").read_text())
        _say("setup: resumed from checkpoint")
    if "fit_seeds" not in done:
        stage_fit_seeds(meta)
        _mark("fit_seeds")
    else:
        _say("seeds: resumed from checkpoint")
    if "committee" not in done:
        cand = stage_committee(meta)
        _mark("committee")
    else:
        proba = np.load(LAB / "proba_pick14.npy")
        _, y, _ = _frames()
        cand = anatomy(proba, y[meta["split"]:])
        _say("committee: resumed from saved probabilities")
    if "wf" not in done:
        wf = stage_wf(meta)
        _mark("wf")
    else:
        wf = json.loads((LAB / "wf.json").read_text())
        _say("wf: resumed from checkpoint")
    if "register" not in done:
        stage_register(meta, cand, wf)
        _mark("register")
    _say("DONE")


if __name__ == "__main__":
    main()
