"""Independent verifier for Muse's Pick #9 — grade #2.

Recomputes the Pick #9 exam metrics from the SAVED probabilities
and the saved supervised frame in a fresh process (no objects
shared with the build), then checks them against the run
registered in training.db. Exit code 0 only if every number
agrees.
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

from models.muse_pick7 import anatomy  # noqa: E402
from models.muse_pick9 import WINDOW, sequence_rows  # noqa: E402

LAB = ROOT / "data" / "muse_pick9"


def main() -> int:
    meta = json.loads((LAB / "meta.json").read_text())
    prod_sup = pd.read_pickle(LAB / "prod_sup.pkl")
    y = prod_sup["target"].astype(int).to_numpy()
    split = meta["split"]
    rows = sequence_rows(len(y), WINDOW)
    exam_rows = np.array([r for r in rows if r >= split], dtype=int)
    exam_y = y[exam_rows]
    proba = np.load(LAB / "proba_pick9.npy")
    assert len(proba) == len(exam_y), "proba/exam length mismatch"
    m = anatomy(proba, exam_y)
    con = sqlite3.connect(ROOT / "data" / "training.db")
    row = con.execute(
        "SELECT id, metrics FROM runs WHERE json_extract(metrics, "
        "'$.family') = 'muse_pick9' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row is not None, "no muse_pick9 run registered"
    run_id, metrics = row[0], json.loads(row[1])
    fam = metrics["muse_pick9"]
    checks = {
        "run_id": run_id,
        "acc_recomputed": m["acc"],
        "acc_registered": metrics["challenger_acc"],
        "acc_agree": abs(m["acc"] - metrics["challenger_acc"]) < 1e-9,
        "precision_agree": abs(m["precision"] - metrics["precision"]) < 1e-9,
        "up_calls_agree": m["up_calls"] == fam["up_calls"],
        "exam_rows_agree": len(exam_y) == fam["exam_rows"],
        "champ96_acc": fam["champ96_acc"],
        "champ73_acc": fam["champ73_acc"],
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
