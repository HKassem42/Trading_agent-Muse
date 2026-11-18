"""HA-sequence challenger family v1 (2026-10-22, Kass green-lit).

Trading is PARKED: lab-only. Never runs the live loop, never writes the
paper ledger or live state, never promotes. Champion artifacts are read
(graded), never written. All artifacts under data/ha_seq/.

Pipeline:
  setup     fetch Coinbase BTC-USD 1h (730d), cache raw + supervised
            frames; record integrity snapshot (champion/paper mtimes).
  features  Heikin-Ashi per-candle channels on the RAW frame (true HA
            history), cached.
  train_main  CNN+LSTM on the production train region (last 792 sup
            rows are the holdout, never trained on); chronological
            validation slice from the END of the train region drives
            early stopping; stats fit on fit-region candles only.
            Epoch-checkpointed (the runtime kills processes).
  grade_main  classifier accuracy/precision on the 792-row holdout,
            majority-class baseline alongside, probabilities saved.
  train_mid   second angle: same recipe fit on the first ~60% of rows,
            graded on the next slice (rows 60%-70%).
  duel      the classifier wrapped in the threshold policy (a genuine
            SB3 model zip) graded by training.nightly.evaluate_windows
            on the standard 3x264 gate windows, champion PPO re-graded
            on the SAME windows with the current stacker champion's
            votes; buy-and-hold rides along in the scorer.
  register  training.db run (kind "ppo", family=ha_seq, promoted=0) in
            the nightly challenger/champion shape + lab note, so the
            Gate Tracker displays the duel.
  verify    champion/paper mtimes unchanged; parked state untouched.

Usage: .venv/bin/python scripts/build_ha_seq.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.technical_indicators import FEATURE_COLUMNS, make_supervised  # noqa: E402
from models import ha_seq  # noqa: E402
from models.ha_policy import build_rule_model  # noqa: E402

MODELS = ROOT / "models"
LAB = ROOT / "data" / "ha_seq"
PROGRESS_PATH = LAB / "progress.json"
RESULTS_PATH = LAB / "results.json"
RAW_PATH = LAB / "ohlcv.pkl"
SUP_PATH = LAB / "sup.pkl"
FEAT_PATH = LAB / "feat.pkl"

SEED = 42
HOLDOUT = 792
WINDOW = 264
VAL_FRAC = 0.15          # chronological tail of the fit region
MAX_EPOCHS = 30
PATIENCE = 6
BATCH = 256
LR = 1e-3


# ---------------------------------------------------------------- state
def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return default
    return default


def _save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def progress() -> set:
    return set(_load_json(PROGRESS_PATH, {}).get("done", []))


def mark(phase: str) -> None:
    _save_json(PROGRESS_PATH, {"done": sorted(progress() | {phase})})


def results() -> dict:
    return _load_json(RESULTS_PATH, {})


def save_result(key: str, value) -> None:
    res = results()
    res[key] = value
    _save_json(RESULTS_PATH, res)


def _mtimes() -> dict:
    out = {}
    for p in (MODELS / "ppo_champion.zip", MODELS / "stacker_champion.pkl",
              MODELS / "stacker_champion.json", ROOT / "data" / "paper.db",
              ROOT / "data" / "live_state.json"):
        out[p.name] = p.stat().st_mtime if p.exists() else None
    return out


# ---------------------------------------------------------------- data
def get_raw() -> pd.DataFrame:
    if RAW_PATH.exists():
        return pd.read_pickle(RAW_PATH)
    from data.market_data import fetch_klines
    df = None
    for attempt in range(3):
        try:
            df = fetch_klines("BTC-USD", interval="1h", period="730d")
            break
        except Exception as e:
            print(f"fetch attempt {attempt + 1} failed: {e}", flush=True)
            if attempt == 2:
                raise
            time.sleep(30 * (attempt + 1))
    LAB.mkdir(parents=True, exist_ok=True)
    df.to_pickle(RAW_PATH)
    return df


def get_sup(raw: pd.DataFrame) -> pd.DataFrame:
    if SUP_PATH.exists():
        return pd.read_pickle(SUP_PATH)
    sup = make_supervised(raw)
    LAB.mkdir(parents=True, exist_ok=True)
    sup.to_pickle(SUP_PATH)
    return sup


def get_feat(raw: pd.DataFrame) -> pd.DataFrame:
    if FEAT_PATH.exists():
        return pd.read_pickle(FEAT_PATH)
    feat = ha_seq.candle_features(raw)
    LAB.mkdir(parents=True, exist_ok=True)
    feat.to_pickle(FEAT_PATH)
    return feat


def split_frames(sup: pd.DataFrame):
    split = len(sup) - HOLDOUT
    train_sup, hold_sup = sup.iloc[:split], sup.iloc[split:]
    assert train_sup.index.max() < hold_sup.index.min(), "train/holdout overlap"
    windows = [hold_sup.iloc[k * WINDOW:(k + 1) * WINDOW] for k in range(3)]
    return train_sup, hold_sup, windows


# ---------------------------------------------------------------- phases
def phase_setup() -> None:
    raw = get_raw()
    sup = get_sup(raw)
    train_sup, hold_sup, _ = split_frames(sup)
    pos = raw.index.get_indexer(sup.index)
    assert (pos >= 0).all(), "sup index not a subset of raw index"
    checks = {
        "train_holdout_disjoint": bool(
            train_sup.index.max() < hold_sup.index.min()),
        "sup_subset_of_raw": bool((pos >= 0).all()),
        "first_sup_row_deep_history": bool(pos[0] >= ha_seq.SEQ_LEN),
        "holdout_target_present": bool(hold_sup["target"].notna().all()),
    }
    save_result("meta", {
        "raw_rows": int(len(raw)), "sup_rows": int(len(sup)),
        "train_rows": int(len(train_sup)), "holdout_rows": int(len(hold_sup)),
        "raw_start": str(raw.index.min()), "raw_end": str(raw.index.max()),
        "sup_start": str(sup.index.min()), "sup_end": str(sup.index.max()),
        "holdout_start": str(hold_sup.index.min()),
        "holdout_up_rate": float(hold_sup["target"].mean()),
        "train_up_rate": float(train_sup["target"].mean()),
        "checks": checks, "integrity_start": _mtimes(),
    })
    print(f"setup: raw {len(raw)} rows, sup {len(sup)}, train "
          f"{len(train_sup)}, holdout {len(hold_sup)}, checks={checks}",
          flush=True)
    assert all(checks.values()), f"self-checks failed: {checks}"


def phase_features() -> None:
    raw = get_raw()
    feat = get_feat(raw)
    warm = int(feat[ha_seq.CHANNELS].isna().any(axis=1).sum())
    desc = {c: {"min": float(feat[c].min()), "max": float(feat[c].max())}
            for c in ha_seq.CHANNELS}
    save_result("features", {"warmup_nan_rows": warm, "ranges": desc})
    print(f"features: {feat.shape}, nan rows {warm}", flush=True)


def _sequences_for(feat, stats, raw_pos, row_slice):
    """(X, y, rows) for sup rows in row_slice: sequences end at each
    row's raw position; rows without a valid sequence are dropped."""
    Xall, valid = ha_seq.build_all_sequences(feat, stats)
    rows = np.arange(row_slice.start, row_slice.stop)
    q = raw_pos[rows]
    keep = valid[q]
    rows, q = rows[keep], q[keep]
    return Xall[q], rows


def _train_classifier(tag: str, fit_rows: slice, val_rows: slice,
                      eval_rows: slice | None) -> dict:
    """One full classifier run: stats from fit-region candles only,
    epoch-checkpointed training with early stopping on val loss."""
    import torch
    from torch import nn

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    raw, sup, feat = get_raw(), get_sup(get_raw()), get_feat(get_raw())
    raw_pos = raw.index.get_indexer(sup.index)
    y = sup["target"].to_numpy(dtype=np.float32)

    # Stats over candles up to and including the last fit candle.
    stats_upto = int(raw_pos[fit_rows.stop - 1]) + 1
    stats = ha_seq.fit_stats(feat, upto=stats_upto)
    Xfit, rfit = _sequences_for(feat, stats, raw_pos, fit_rows)
    Xval, rval = _sequences_for(feat, stats, raw_pos, val_rows)
    yfit, yval = y[rfit], y[rval]
    print(f"{tag}: fit {len(yfit)} seqs, val {len(yval)} seqs, "
          f"stats_upto={stats_upto}", flush=True)
    assert len(yfit) > 1000 and len(yval) > 100

    model = ha_seq.build_model()
    n_params = ha_seq.count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    Xfit_t = torch.from_numpy(Xfit)
    yfit_t = torch.from_numpy(yfit)
    Xval_t = torch.from_numpy(Xval)
    yval_t = torch.from_numpy(yval)

    ckpt_path = LAB / f"{tag}_ckpt.pt"
    best = {"val_loss": float("inf"), "epoch": -1}
    history: list[dict] = []
    start_epoch = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        best = ck["best"]
        history = ck["history"]
        start_epoch = ck["epoch"] + 1
        if best.get("state") is not None:
            pass  # best state kept inside ckpt; restored at the end
        print(f"{tag}: resuming from epoch {start_epoch} "
              f"(best val_loss {best['val_loss']:.4f})", flush=True)

    def evaluate(X_t, y_t):
        model.eval()
        losses, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for i in range(0, len(y_t), 1024):
                xb, yb = X_t[i:i + 1024], y_t[i:i + 1024]
                logits = model(xb)
                losses += float(loss_fn(logits, yb)) * len(yb)
                correct += int(((logits > 0).float() == yb).sum())
                n += len(yb)
        return losses / n, correct / n

    stale = 0
    for epoch in range(start_epoch, MAX_EPOCHS):
        t0 = time.time()
        model.train()
        gen = torch.Generator().manual_seed(SEED + epoch)
        perm = torch.randperm(len(yfit_t), generator=gen)
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xfit_t[idx]), yfit_t[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        vloss, vacc = evaluate(Xval_t, yval_t)
        rec = {"epoch": epoch, "train_loss": tot / len(yfit_t),
               "val_loss": vloss, "val_acc": vacc,
               "secs": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"{tag} epoch {epoch}: train_loss {rec['train_loss']:.4f} "
              f"val_loss {vloss:.4f} val_acc {vacc:.4f} "
              f"({rec['secs']:.0f}s)", flush=True)
        if vloss < best["val_loss"] - 1e-4:
            best = {"val_loss": vloss, "epoch": epoch,
                    "state": {k: v.clone() for k, v in
                              model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "opt": opt.state_dict(), "best": best,
                    "history": history}, ckpt_path)
        if stale >= PATIENCE:
            print(f"{tag}: early stop at epoch {epoch} "
                  f"(best {best['epoch']})", flush=True)
            break

    model.load_state_dict(best["state"])
    out_path = LAB / f"{tag}_model.pt"
    torch.save({"state": best["state"], "stats": stats,
                "seq_len": ha_seq.SEQ_LEN, "channels": ha_seq.CHANNELS,
                "best_epoch": best["epoch"]}, out_path)
    rec = {"n_params": n_params, "best_epoch": best["epoch"],
           "best_val_loss": best["val_loss"],
           "epochs_run": len(history), "history_tail": history[-3:],
           "fit_n": int(len(yfit)), "val_n": int(len(yval)),
           "fit_rows": [int(fit_rows.start), int(fit_rows.stop)],
           "val_rows": [int(val_rows.start), int(val_rows.stop)],
           "stats_upto": stats_upto}
    if eval_rows is not None:
        Xev, rev = _sequences_for(feat, stats, raw_pos, eval_rows)
        yev = y[rev]
        model.eval()
        probs = []
        with torch.no_grad():
            Xt = torch.from_numpy(Xev)
            for i in range(0, len(Xt), 1024):
                probs.append(torch.sigmoid(model(Xt[i:i + 1024])).numpy())
        p = np.concatenate(probs)
        pred = (p >= 0.5).astype(int)
        from sklearn.metrics import accuracy_score, precision_score
        rec["eval"] = {
            "acc": float(accuracy_score(yev, pred)),
            "prec": float(precision_score(yev, pred, zero_division=0)),
            "n": int(len(yev)),
            "up_rate": float(yev.mean()),
            "majority_baseline": float(max(yev.mean(), 1 - yev.mean())),
            "rows": [int(eval_rows.start), int(eval_rows.stop)],
            "mean_prob": float(p.mean()), "prob_std": float(p.std()),
        }
        print(f"{tag} eval: acc {rec['eval']['acc']:.4f} "
              f"prec {rec['eval']['prec']:.4f} n={len(yev)} "
              f"baseline {rec['eval']['majority_baseline']:.4f}", flush=True)
    return rec


def phase_train_main() -> None:
    sup = get_sup(get_raw())
    split = len(sup) - HOLDOUT
    val_start = split - int(VAL_FRAC * split)
    rec = _train_classifier("main", slice(0, val_start),
                            slice(val_start, split), None)
    save_result("train_main", rec)


def phase_grade_main() -> None:
    import torch
    raw, sup, feat = get_raw(), get_sup(get_raw()), get_feat(get_raw())
    raw_pos = raw.index.get_indexer(sup.index)
    saved = torch.load(LAB / "main_model.pt", weights_only=False)
    model = ha_seq.build_model()
    model.load_state_dict(saved["state"])
    model.eval()
    Xall, valid = ha_seq.build_all_sequences(feat, saved["stats"])
    y = sup["target"].to_numpy(dtype=float)
    # Inference runs over RAW rows (Xall is raw-length); map to sup rows
    # through raw_pos before touching any sup-aligned array.
    probs_raw = np.full(len(raw), np.nan)
    Xt = torch.from_numpy(Xall)
    with torch.no_grad():
        for i in range(0, len(raw), 1024):
            chunk = torch.sigmoid(model(Xt[i:i + 1024])).numpy()
            probs_raw[i:i + len(chunk)] = chunk
    probs = probs_raw[raw_pos]
    probs[~valid[raw_pos]] = np.nan
    np.savez(LAB / "probs.npz", probs=probs)
    split = len(sup) - HOLDOUT
    hp, hy = probs[split:], y[split:]
    assert not np.isnan(hp).any(), "holdout rows missing sequences"
    pred = (hp >= 0.5).astype(int)
    from sklearn.metrics import accuracy_score, precision_score
    rec = {
        "holdout_acc": float(accuracy_score(hy, pred)),
        "holdout_prec": float(precision_score(hy, pred, zero_division=0)),
        "holdout_n": int(len(hy)),
        "holdout_up_rate": float(hy.mean()),
        "majority_baseline": float(max(hy.mean(), 1 - hy.mean())),
        "mean_prob": float(hp.mean()), "prob_std": float(hp.std()),
        "prob_min": float(hp.min()), "prob_max": float(hp.max()),
        "frac_above_055": float((hp >= 0.55).mean()),
        "frac_below_045": float((hp <= 0.45).mean()),
    }
    save_result("grade_main", rec)
    print(f"grade_main: holdout acc {rec['holdout_acc']:.4f} "
          f"prec {rec['holdout_prec']:.4f} n={rec['holdout_n']} "
          f"baseline {rec['majority_baseline']:.4f}", flush=True)


def phase_train_mid() -> None:
    sup = get_sup(get_raw())
    n = len(sup)
    region_end = int(0.60 * n)
    fit_end = int(region_end * (1 - VAL_FRAC))
    rec = _train_classifier("mid", slice(0, fit_end),
                            slice(fit_end, region_end),
                            slice(region_end, int(0.70 * n)))
    save_result("train_mid", rec)


def _agg(res) -> dict:
    """Flatten a 3-window evaluate_windows result into the nightly
    run-record shape (mirrors training.nightly._agg)."""
    if res is None:
        return {"mean_reward": -999.0, "total_pnl": 0.0,
                "max_drawdown": 0.0, "score": -1e9, "runnable": False}
    ws = res["windows"]
    return {"score": res["agg_score"], "total_pnl": res["total_pnl"],
            "mean_reward": res["mean_reward"],
            "max_drawdown": float(max(w["max_drawdown"] for w in ws)),
            "bench_pnl": float(sum(w["bench_pnl"] for w in ws)),
            "action_mix": ws[-1]["action_mix"],
            "windows": [{"score": round(w["score"], 2),
                         "pnl": round(w["total_pnl"], 2),
                         "bench": round(w["bench_pnl"], 2)} for w in ws],
            "runnable": True}


def phase_duel() -> None:
    import joblib
    from training.nightly import evaluate_windows
    raw, sup = get_raw(), get_sup(get_raw())
    _, _, windows = split_frames(sup)
    probs = np.load(LAB / "probs.npz")["probs"]
    split = len(sup) - HOLDOUT
    ha_p = [probs[split + k * WINDOW: split + (k + 1) * WINDOW]
            for k in range(3)]
    assert all(not np.isnan(p).any() for p in ha_p)
    stacker_model = joblib.load(MODELS / "stacker_champion.pkl")
    champ_p = [stacker_model.predict_proba(w[FEATURE_COLUMNS])[:, 1]
               for w in windows]

    rule = build_rule_model()
    zip_path = LAB / "ha_rule_model"
    rule.save(zip_path)  # SB3 appends .zip
    zip_file = LAB / "ha_rule_model.zip"
    assert zip_file.exists()

    champ_res = evaluate_windows(MODELS / "ppo_champion.zip",
                                 windows, champ_p)
    ha_res = evaluate_windows(zip_file, windows, ha_p)
    assert champ_res is not None, "champion failed to grade"
    assert ha_res is not None, "HA policy failed to grade"
    wins = sum(1 for cw, hw in zip(champ_res["windows"], ha_res["windows"])
               if hw["score"] > cw["score"])
    bench_ha = float(sum(w["bench_pnl"] for w in ha_res["windows"]))
    bench_ch = float(sum(w["bench_pnl"] for w in champ_res["windows"]))
    assert abs(bench_ha - bench_ch) < 1e-6, "bench mismatch across arms"
    agg_re = float(np.mean([w["score"] for w in ha_res["windows"]]))
    assert abs(agg_re - ha_res["agg_score"]) < 1e-9
    # Activity guard (added after the v1 run exposed the hole): a
    # challenger that never trades scores 0.0 and "beats" any losing
    # champion without contesting a single window. v1's classifier
    # never output a probability outside [0.468, 0.547] on the holdout,
    # so the 0.55/0.45 policy held 100% and formally "passed" 3/3 --
    # a vacuous pass. A real pass requires actual round trips.
    traded = any(w["round_trips_per_day"] > 0 for w in ha_res["windows"])
    gate_pass = bool(ha_res["agg_score"] > champ_res["agg_score"]
                     and wins >= 2 and traded)
    rec = {
        "challenger": _agg(ha_res), "champion": _agg(champ_res),
        "windows_won": wins, "gate_pass": gate_pass, "traded": traded,
        "bench_pnl_total": bench_ha,
        "agg_recomputed": agg_re,
        "ha_action_mix_by_window": [w["action_mix"]
                                    for w in ha_res["windows"]],
        "champ_action_mix_by_window": [w["action_mix"]
                                       for w in champ_res["windows"]],
    }
    save_result("duel", rec)
    print(f"duel: HA agg={ha_res['agg_score']:,.2f} "
          f"pnl={ha_res['total_pnl']:+,.2f} | champ "
          f"agg={champ_res['agg_score']:,.2f} "
          f"pnl={champ_res['total_pnl']:+,.2f} | bench "
          f"{bench_ha:+,.2f} | windows won {wins}/3 | "
          f"GATE {'PASS' if gate_pass else 'FAIL'}", flush=True)


def phase_register() -> None:
    from training.nightly import add_lab_note, log_run
    duel = results()["duel"]
    gm = results()["grade_main"]
    tm = results()["train_main"]
    ch, cp = duel["challenger"], duel["champion"]
    verdict = ("GATE PASS" if duel["gate_pass"] else "gate FAIL")
    reason = (f"HA-seq challenger (family=ha_seq): {verdict} -- won "
              f"{duel['windows_won']}/3 windows, aggregate "
              f"{ch['score']:.0f} vs champion {cp['score']:.0f} "
              f"(next-candle fills; buy-and-hold over the same windows: "
              f"${duel['bench_pnl_total']:+,.0f}); classifier holdout "
              f"acc {gm['holdout_acc']:.4f} on n={gm['holdout_n']}")
    metrics = {
        "challenger": ch, "champion": cp, "reason": reason,
        "family": "ha_seq", "deep": False, "timesteps": 0,
        "reward_curve": None,
        "ha_seq": {
            "classifier_holdout_acc": gm["holdout_acc"],
            "classifier_holdout_prec": gm["holdout_prec"],
            "classifier_holdout_n": gm["holdout_n"],
            "majority_baseline": gm["majority_baseline"],
            "thresholds": {"long_p": 0.55, "short_p": 0.45},
            "n_params": tm["n_params"], "best_epoch": tm["best_epoch"],
            "gate_pass": duel["gate_pass"],
            "windows_won": duel["windows_won"],
        },
    }
    log_run("ppo", metrics, False)  # lab challenger: never promoted here
    import sqlite3
    with sqlite3.connect(ROOT / "data" / "training.db") as con:
        run_id = con.execute("SELECT MAX(id) FROM runs").fetchone()[0]
    add_lab_note("HA-sequence challenger (family=ha_seq)", reason)
    save_result("register", {"run_id": int(run_id), "reason": reason})
    print(f"register: run {run_id} logged (promoted=0)", flush=True)


def phase_verify() -> None:
    start = results()["meta"]["integrity_start"]
    now = _mtimes()
    same = all(start[k] == now[k] for k in start)
    save_result("verify", {"mtimes_unchanged": bool(same),
                           "start": start, "end": now})
    print(f"verify: champion/paper mtimes unchanged = {same}", flush=True)
    assert same, f"INTEGRITY: mtimes changed {start} -> {now}"


PHASES = [
    ("setup", phase_setup),
    ("features", phase_features),
    ("train_main", phase_train_main),
    ("grade_main", phase_grade_main),
    ("train_mid", phase_train_mid),
    ("duel", phase_duel),
    ("register", phase_register),
    ("verify", phase_verify),
]


def main() -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    done = progress()
    print(f"=== HA-seq v1 build (phases done: {sorted(done) or 'none'}) ===",
          flush=True)
    for name, fn in PHASES:
        if name in done:
            print(f"{name}: already done -- skipped", flush=True)
            continue
        t0 = time.time()
        fn()
        mark(name)
        print(f"--- phase {name} done in {time.time() - t0:.0f}s ---",
              flush=True)
    print("=== HA-seq v1 build complete ===", flush=True)


if __name__ == "__main__":
    main()
