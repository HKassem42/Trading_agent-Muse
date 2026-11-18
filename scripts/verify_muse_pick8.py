"""Independent verifier for Muse's Pick #8 — grade #2.

Recomputes the Pick #8 exam metrics from the SAVED probabilities
and the raw close series in a fresh process (no objects shared
with the build), then checks them against the run registered in
training.db. Exit code 0 only if every number agrees.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.muse_pick7 import anatomy, exam_split  # noqa: E402
from models.muse_picks456 import horizon_labels  # noqa: E402

LAB = ROOT / "data" / "muse_pick8"
LAB1 = ROOT / "data" / "muse_next_try"


def main() -> int:
    raw = pd.read_pickle(LAB1 / "raw_ohlcv.pkl")
    prod_sup = pd.read_pickle(LAB1 / "prod_sup.pkl")
    y24 = horizon_labels(raw["close"], 24).reindex(prod_sup.index)
    split = len(prod_sup) - 792
    _, exam_pos = exam_split(prod_sup.index, y24, split, horizon=24)
    exam_y = y24.to_numpy(dtype=float)[exam_pos].astype(int)
    proba = np.load(LAB / "proba_pick8.npy")
    assert len(proba) == len(exam_y), "proba/exam length mismatch"
    m = anatomy(proba, exam_y)
    con = sqlite3.connect(ROOT / "data" / "training.db")
    row = con.execute(
        "SELECT id, metrics FROM runs WHERE json_extract(metrics, "
        "'$.family') = 'muse_pick8' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row is not None, "no muse_pick8 run registered"
    run_id, metrics = row[0], json.loads(row[1])
    fam = metrics["muse_pick8"]
    checks = {
        "run_id": run_id,
        "acc_recomputed": m["acc"],
        "acc_registered": metrics["challenger_acc"],
        "acc_agree": abs(m["acc"] - metrics["challenger_acc"]) < 1e-9,
        "precision_agree": abs(m["precision"] - metrics["precision"]) < 1e-9,
        "up_calls_agree": m["up_calls"] == fam["up_calls"],
        "exam_rows_agree": len(exam_y) == fam["exam_rows"],
        "pick7_candidate_acc": fam["pick7_candidate_acc"],
        "wf_acc": metrics["challenger_wf_acc"],
        "wf_majority": fam["wf_majority"],
        "verdict": fam["verdict"],
        "pending_approval": metrics["pending_approval"],
    }
    print(json.dumps(checks, indent=1))
    ok = all(checks[k] for k in ("acc_agree", "precision_agree",
                                 "up_calls_agree", "exam_rows_agree"))
    print("VERIFIED" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
