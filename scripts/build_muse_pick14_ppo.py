"""Muse's Pick #14 — PPO phase — LAB study (2026-11-18).

Runs ONLY when the Pick #14 stacker passed its frozen bar
(scripts/build_muse_pick14.py, results.json pending=true);
pre-registered in refs/muse-pick-14-preregistration-2026-11-18.md.

The standard nightly PPO phase with the committee standing in
for the single stacker, mirroring src/training/nightly.py's
PPO section step for step:

- fresh 730d frame; the last 792 supervised candles are the
  holdout (3 chronological 264-candle duel windows); nothing
  trains on them;
- a FRESH committee (7 fixed seeds, models.muse_pick14) is fit
  on the training region's ha10 matrix; the challenger PPO's
  training votes are CROSS-FITTED committee votes
  (crossfit_committee_p_up: TimeSeriesSplit 4 folds, each fold's
  votes from a committee fit only on earlier rows, leading
  segment neutral 0.5 — the nightly crossfit_p_up semantics);
- PPO challenger: train_ppo, warm-start from run 112,
  60,000 steps, ent_coef 0.05, training region only;
- duel: the champion pair (PPO run 112 reading Stacker run
  126's votes on the champion's own frame) vs the challenger
  (the new PPO reading the fresh committee's votes), graded by
  nightly.evaluate_windows with next-candle-open fills,
  score = P&L - 0.5 x maxDD;
- the full gate: formula (aggregate beats champion AND >= 2/3
  windows) AND gate_decision's floor (non-hold >= 10%, duel
  P&L > $0, P&L > buy-and-hold).

A full gate pass marks the PPO run pending_approval — the
study STOPS there: no champion file is installed, no crowning
is executed, Book #5 is not re-pinned, no seat is touched; the
crowning ruling belongs to the parent agent. A gate failure is
recorded NOT PROMOTED as usual. This script appends the
study's single lab note (deferred from the stacker script so
one note covers design, setup, per-seed results, committee
result, bar verdict, and this PPO outcome). Guards (champion /
seat SHA-256) are asserted unchanged at the end. LAB ONLY.

Usage: .venv/bin/python scripts/build_muse_pick14_ppo.py
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

from data.market_data import fetch_klines  # noqa: E402
from features.technical_indicators import make_supervised  # noqa: E402
from models.muse_pick10 import HA_COLUMNS, build_ha_matrix  # noqa: E402
from models.muse_pick14 import (crossfit_committee_p_up,  # noqa: E402
                                fit_committee)
from models.ppo_train import train as train_ppo  # noqa: E402
from training.nightly import (_agg, _champion_meta,  # noqa: E402
                              add_lab_note, champion_holdout_frame,
                              evaluate_windows, gate_decision,
                              log_run)

warnings.filterwarnings("ignore")

MODELS = ROOT / "models"
DATA = ROOT / "data"
LAB = ROOT / "data" / "muse_pick14"
FAMILY = "muse_pick14"
PPO_LABEL = "PPO — Committee Votes"
PPO_DESIGN = ("Muse's Pick #14 PPO phase: the standard nightly "
              "PPO challenger (warm-start run 112, 60k steps, "
              "training region only) trained and graded on the "
              "7-seed committee's votes — cross-fitted committee "
              "votes in training, a fresh committee fit on the "
              "training region for the duel — against the "
              "sitting champion pair (PPO run 112 + Stacker "
              "run 126), full gate (formula + floor)")
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


def _say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    results = json.loads((LAB / "results.json").read_text())
    assert results["pending"], \
        "stacker did not pass its bar — no PPO phase (pre-reg)"
    meta = results["meta"]
    guard_before = _guard()
    assert guard_before == meta["guard_before"], \
        "guard state moved since the stacker phase"
    sidecar = MODELS / "ppo_challenger_progress.json"
    assert not sidecar.exists(), \
        f"stale PPO progress sidecar present ({sidecar}); " \
        f"refusing to train over an unknown resume state"

    # -- the nightly frame, fetched the nightly way --
    df = None
    for attempt in range(3):
        try:
            df = fetch_klines("BTC-USD", interval="1h",
                              period="730d")
            break
        except Exception as e:
            _say(f"fetch attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                raise
            time.sleep(30 * (attempt + 1))
    sup = make_supervised(df)
    split = len(sup) - 792
    train_sup, hold_sup = sup.iloc[:split], sup.iloc[split:]
    assert train_sup.index.max() < hold_sup.index.min()
    windows_sup = [hold_sup.iloc[k * 264:(k + 1) * 264]
                   for k in range(3)]
    ha_all = build_ha_matrix(df)
    Htr = ha_all.reindex(train_sup.index)[HA_COLUMNS]
    Hhold = ha_all.reindex(hold_sup.index)[HA_COLUMNS]
    assert Htr.index.equals(train_sup.index)
    assert Hhold.index.equals(hold_sup.index)
    ytr = train_sup["target"]
    _say(f"frame: {len(sup)} supervised rows, train "
         f"{len(train_sup)}, holdout {len(hold_sup)}")

    # -- the fresh committee for this frame --
    t0 = time.time()
    committee = fit_committee(Htr, ytr)
    joblib.dump(committee, LAB / "committee_ppo_phase.pkl")
    _say(f"committee fit on the training region "
         f"({time.time() - t0:.0f}s)")
    chall_votes_all = committee.predict_proba(Hhold)[:, 1]
    p_up_windows_chall = [chall_votes_all[k * 264:(k + 1) * 264]
                          for k in range(3)]

    # -- champion pair votes: run 126 on its own frame --
    champ_meta = _champion_meta()
    champ_Xhold, _tag = champion_holdout_frame(df, hold_sup,
                                               champ_meta)
    stacker_model = joblib.load(MODELS / "stacker_champion.pkl")
    champ_votes_all = stacker_model.predict_proba(champ_Xhold)[:, 1]
    p_up_windows_champ = [champ_votes_all[k * 264:(k + 1) * 264]
                          for k in range(3)]

    # -- cross-fitted committee training votes --
    t0 = time.time()
    p_up_train = crossfit_committee_p_up(Htr, ytr, folds=4)
    np.save(LAB / "ppo_phase_crossfit_votes.npy", p_up_train)
    _say(f"cross-fitted committee votes ready "
         f"({time.time() - t0:.0f}s)")

    # -- train the challenger, the standard nightly way --
    challenger_path = train_ppo(
        timesteps=60_000, warm_start=True,
        champion_path=MODELS / "ppo_champion.zip",
        stacker_path=MODELS / "stacker_champion.pkl",
        db_path=DATA / "train_tmp.db",
        ent_coef=0.05, sup=train_sup, p_up=p_up_train)
    import shutil
    shutil.copy2(challenger_path, LAB / "ppo_challenger.zip")
    _say(f"challenger trained -> {challenger_path}")

    # -- the duel --
    champ_res = evaluate_windows(MODELS / "ppo_champion.zip",
                                 windows_sup, p_up_windows_champ)
    chall_res = evaluate_windows(challenger_path, windows_sup,
                                 p_up_windows_chall)
    champ, chall = _agg(champ_res), _agg(chall_res)
    wins = 0
    if champ_res and chall_res:
        wins = sum(1 for cw, hw in zip(champ_res["windows"],
                                       chall_res["windows"])
                   if hw["score"] > cw["score"])
    formula_pass = chall_res is not None and (
        champ_res is None or (chall["score"] > champ["score"]
                              and wins >= 2))
    ppo_verdict, floor_fail = gate_decision(chall, formula_pass)
    _say(f"duel: challenger agg={chall['score']:.2f} "
         f"(pnl={chall['total_pnl']:.2f}) champion "
         f"agg={champ['score']:.2f} (pnl={champ['total_pnl']:.2f}) "
         f"windows won={wins}/3 bench={chall['bench_pnl']:.2f} "
         f"verdict_promote={ppo_verdict} floor_fail={floor_fail}")
    if chall_res is None:
        reason = ("Muse's Pick #14 PPO challenger failed "
                  "evaluation; champion kept")
    elif floor_fail is not None:
        reason = (f"Muse's Pick #14 PPO challenger passed the "
                  f"relative gate (aggregate {chall['score']:.0f} "
                  f"vs champion {champ['score']:.0f}, won {wins}/3 "
                  f"windows) but was not promoted — {floor_fail}; "
                  f"champion kept")
    elif ppo_verdict:
        reason = (f"Muse's Pick #14 PPO challenger won {wins}/3 "
                  f"windows, aggregate score {chall['score']:.0f} "
                  f"vs champion {champ['score']:.0f} (next-candle "
                  f"fills; buy-and-hold over the same windows: "
                  f"${chall['bench_pnl']:+,.0f}) (gate pass incl. "
                  f"two-part floor) — WOULD PROMOTE: held as "
                  f"PENDING APPROVAL; crowning ruled on by Muse "
                  f"under Kass's 2026-11-12 standing delegation")
    else:
        reason = (f"champion kept: Muse's Pick #14 PPO "
                  f"challenger won {wins}/3 windows, aggregate "
                  f"{chall['score']:.0f} vs champion "
                  f"{champ['score']:.0f}")
    pending = bool(ppo_verdict)
    reward_curve = None
    telemetry = None
    curve_path = MODELS / "ppo_challenger_curve.json"
    if curve_path.exists():
        try:
            side = json.loads(curve_path.read_text())
            reward_curve = side.get("curve")
            telemetry = side.get("telemetry")
        except Exception:
            reward_curve = None
            telemetry = None
    log_run("ppo", {"challenger": chall, "champion": champ,
                    "reason": reason, "deep": False,
                    "timesteps": 60_000,
                    "pending_approval": pending,
                    "reward_curve": reward_curve,
                    "telemetry": telemetry,
                    "family": FAMILY, "feature_set": "ha10ens",
                    FAMILY: {
                        "label": PPO_LABEL,
                        "verdict": "PASS" if pending else "FAIL",
                        "design": PPO_DESIGN,
                        "stacker_run": results["run_id"],
                        "windows_won": wins,
                    }}, False)
    import sqlite3
    with sqlite3.connect(DATA / "training.db") as con:
        ppo_run_id = con.execute(
            "SELECT MAX(id) FROM runs").fetchone()[0]

    # -- the study's single lab note (deferred from the stacker
    #    script so it covers the PPO outcome too) --
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_muse_pick14 import _note_text
    m = results["pick"]
    wf = results["wf"]
    seed_accs = {int(k): v for k, v in
                 results["seed_accs"].items()}
    note = _note_text(meta, m, wf, seed_accs, True)
    note += (
        f" PPO phase: challenger (committee votes) won "
        f"{wins}/3 windows, aggregate {chall['score']:.0f} vs "
        f"champion {champ['score']:.0f}, duel P&L "
        f"${chall['total_pnl']:+,.0f} vs champion "
        f"${champ['total_pnl']:+,.0f}, buy-and-hold "
        f"${chall['bench_pnl']:+,.0f} — "
        + ("FULL GATE PASS: both runs held as PENDING APPROVAL "
           "for Muse's crowning ruling; nothing installed, no "
           "seat touched."
           if pending else
           "gate NOT passed: PPO recorded NOT PROMOTED; the "
           "stacker run stands pending on its own verdict."))
    add_lab_note(f"Muse's Pick #14 (family={FAMILY})", note)

    guard_after = _guard()
    assert guard_after == guard_before, "GUARD TRIPPED"
    out = {"ppo_run_id": ppo_run_id, "pending": pending,
           "wins": wins, "challenger": chall, "champion": champ,
           "floor_fail": floor_fail}
    (LAB / "ppo_results.json").write_text(json.dumps(out, indent=1))
    _say(f"registered PPO run {ppo_run_id}: "
         f"{'PASS (pending approval)' if pending else 'NOT PROMOTED'}")
    _say("DONE")


if __name__ == "__main__":
    main()
