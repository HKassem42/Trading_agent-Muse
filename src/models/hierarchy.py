"""Two-step decision hierarchy (2026-10-24, Drawing-Board Option C).

Executes refs/hierarchy-plan-2026-10-24.md EXACTLY as pre-registered.
The plan is the contract; this module is machinery only.

Kass's idea: the flat 3-class models drown in HOLD and spend their
capacity separating BUY from SHORT on rows where nothing is
tradeable. Split the decision:

  Stage 1 (gate)      binary TRADE / NO-TRADE. TRADE = the matrix
                      label is BUY or SHORT (|next-bar return|
                      cleared the +-0.30% fee band); NO-TRADE =
                      label HOLD.
  Stage 2 (direction) binary UP / DOWN, trained ONLY on true
                      TRADE rows (UP = BUY, DOWN = SHORT).

Combined policy (fixed): gate fires when P(TRADE) >= 0.50; the
direction is stage 2's argmax (exact tie -> UP, the project's
LONG-first convention). NO-TRADE -> HOLD, TRADE+UP -> BUY,
TRADE+DOWN -> SHORT. The end-to-end prediction is the only
verdict-bearing object; stage metrics are diagnostics.

Arms: "focus" (gbm_focus.FOCUS_FEATURES, 11 columns) and "all"
(gbm_focus.matrix_feature_columns, 60 columns). Learner: LightGBM
binary with gbm_focus.LGBM_PARAMS' shared values verbatim
(objective adapted to "binary", num_class dropped -- the mechanical
change a binary stage requires), class weights = inverse class
frequency within each stage's own training rows
(gbm_focus.balanced_weights).

For the duel the end-to-end labels ride the env's vote channel as
signed conviction +1 / -1 / 0 encoded p_up = (c + 1) / 2 -- the
composite screen's rule-arm encoding -- through the same saved
rule-policy zip (models.gbm_policy) and the same duel mirror
(models.composite_screen.evaluate_windows_composite).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models import gbm_focus as gf

TARGET_COL = gf.TARGET_COL
CLASSES = gf.CLASSES  # ["BUY", "HOLD", "SHORT"]
GATE_THRESHOLD = 0.50

ARMS = ("focus", "all")
ARM_LABELS = {"focus": "Focus 11", "all": "All Features"}
ARM_DESIGNS = {
    "focus": "two-step hierarchy on Kass's focus-11 features: "
             "stage-1 binary gate (TRADE vs NO-TRADE) + stage-2 "
             "binary direction (UP/DOWN, TRADE rows only), LightGBM, "
             "class-weighted per stage",
    "all": "two-step hierarchy on all 60 matrix feature columns: "
           "stage-1 binary gate (TRADE vs NO-TRADE) + stage-2 "
           "binary direction (UP/DOWN, TRADE rows only), LightGBM, "
           "class-weighted per stage",
}

# gbm_focus.LGBM_PARAMS with the mechanically required binary
# adaptation (objective binary, no num_class); every shared value
# (300 trees, lr 0.05, 31 leaves, min-child 40, subsample 0.9/1,
# colsample 0.9, lambda 1.0, seed 42) is verbatim.
BINARY_PARAMS = {k: v for k, v in gf.LGBM_PARAMS.items()
                 if k not in ("objective", "num_class")}
BINARY_PARAMS["objective"] = "binary"

__all__ = ["TARGET_COL", "CLASSES", "GATE_THRESHOLD", "ARMS",
           "ARM_LABELS", "ARM_DESIGNS", "BINARY_PARAMS",
           "arm_features", "gate_target", "direction_target",
           "HierarchyModel", "fit_hierarchy", "end_to_end_labels",
           "hierarchy_conviction", "walk_forward_arm",
           "rule1_walk_forward", "rule2_duel", "rule3_robustness",
           "apply_verdict"]


# ------------------------------------------------------------------ data
def arm_features(df: pd.DataFrame, arm: str) -> list[str]:
    if arm == "focus":
        return list(gf.FOCUS_FEATURES)
    if arm == "all":
        return gf.matrix_feature_columns(list(df.columns))
    raise ValueError(f"unknown arm {arm!r}")


# ---------------------------------------------------------------- targets
def gate_target(y3) -> np.ndarray:
    """1 = TRADE (label BUY or SHORT), 0 = NO-TRADE (label HOLD)."""
    return (np.asarray(y3) != "HOLD").astype(int)


def direction_target(y3) -> np.ndarray:
    """1 = UP (label BUY), 0 = DOWN (label SHORT). Defined only on
    TRADE rows; HOLD rows map to 0 here and must be masked out by
    the caller (fit_hierarchy does)."""
    return (np.asarray(y3) == "BUY").astype(int)


# ---------------------------------------------------------------- model
class _BinaryPredictor:
    """One LightGBM binary stage; predict_proba_pos returns P(class 1)
    regardless of the model's internal class ordering."""

    def __init__(self, model):
        self.model = model
        self._pos = list(model.classes_).index(1)

    def predict_proba_pos(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.model.predict_proba(X)[:, self._pos]


def _fit_binary(X: np.ndarray, y01: np.ndarray) -> _BinaryPredictor:
    from lightgbm import LGBMClassifier

    y01 = np.asarray(y01)
    assert set(np.unique(y01).tolist()) == {0, 1}, \
        "binary stage needs both classes present"
    model = LGBMClassifier(**BINARY_PARAMS)
    model.fit(X, y01, sample_weight=gf.balanced_weights(y01))
    return _BinaryPredictor(model)


class HierarchyModel:
    """The two fitted stages plus the fixed combination rule."""

    def __init__(self, gate: _BinaryPredictor,
                 direction: _BinaryPredictor):
        self.gate = gate
        self.direction = direction

    def predict_proba_trade(self, X) -> np.ndarray:
        return self.gate.predict_proba_pos(X)

    def predict_proba_up(self, X) -> np.ndarray:
        return self.direction.predict_proba_pos(X)

    def predict(self, X) -> np.ndarray:
        return end_to_end_labels(self.predict_proba_trade(X),
                                 self.predict_proba_up(X))


def end_to_end_labels(p_trade, p_up) -> np.ndarray:
    """The fixed combination rule: fire at P(TRADE) >= 0.50;
    direction by stage-2 argmax with an exact tie going UP
    (p_up >= 0.50 -> BUY)."""
    p_trade = np.asarray(p_trade, dtype=float)
    p_up = np.asarray(p_up, dtype=float)
    out = np.full(p_trade.shape, "HOLD", dtype=object)
    fire = p_trade >= GATE_THRESHOLD
    out[fire & (p_up >= 0.5)] = "BUY"
    out[fire & (p_up < 0.5)] = "SHORT"
    return out


def hierarchy_conviction(model: HierarchyModel, X) -> np.ndarray:
    """End-to-end labels as duel conviction: BUY +1, SHORT -1,
    HOLD 0 (the composite screen's rule-arm encoding)."""
    lab = model.predict(X)
    c = np.zeros(lab.shape, dtype=float)
    c[lab == "BUY"] = 1.0
    c[lab == "SHORT"] = -1.0
    return c


# ---------------------------------------------------------------- fitting
def fit_hierarchy(X, y3, ts=None) -> tuple[HierarchyModel, dict]:
    """Fit both stages on the rows given -- and ONLY those rows.
    Stage 2 consumes the subset that is truly TRADE. The audit
    records exactly what was consumed (counts, class counts, and,
    when `ts` is given, the timestamp bounds of each stage's input)
    so a caller can poison other rows and prove the poison never
    reached a fit input."""
    X = np.asarray(X, dtype=float)
    y3 = np.asarray(y3)
    g = gate_target(y3)
    gate = _fit_binary(X, g)
    tr = g == 1
    assert tr.sum() > 0, "no TRADE rows to train stage 2"
    direction = _fit_binary(X[tr], direction_target(y3[tr]))
    audit = {
        "n_fed": int(len(y3)),
        "gate_class_counts": {0: int((g == 0).sum()),
                              1: int((g == 1).sum())},
        "n_trade_fed": int(tr.sum()),
        "direction_class_counts": {
            0: int((direction_target(y3[tr]) == 0).sum()),
            1: int((direction_target(y3[tr]) == 1).sum())},
        "fit_max_abs": float(np.abs(X).max()),
    }
    if ts is not None:
        ts = pd.DatetimeIndex(ts)
        audit["fed_ts_min"] = ts.min()
        audit["fed_ts_max"] = ts.max()
        audit["trade_fed_ts_max"] = ts[tr].max()
    return HierarchyModel(gate, direction), audit


# ------------------------------------------------------------- walk-forward
def _stage_diagnostics(y3_va, p_trade, p_up, pred) -> dict:
    """Stage diagnostics over one graded block (never
    verdict-bearing): where does the hierarchy lose -- gate or
    direction?"""
    from sklearn.metrics import balanced_accuracy_score

    y3_va = np.asarray(y3_va)
    g_true = gate_target(y3_va)
    fired = np.asarray(p_trade) >= GATE_THRESHOLD
    dir_up = np.asarray(p_up) >= 0.5
    dir_true_up = direction_target(y3_va) == 1
    out = {
        "gate_balanced_accuracy": float(balanced_accuracy_score(
            g_true, fired.astype(int))) if len(np.unique(g_true)) > 1
            else float("nan"),
        "trade_base_rate": float(g_true.mean()),
        "gate_fire_rate": float(fired.mean()),
        "activity_share": float((pred != "HOLD").mean()),
        "n_true_trade": int(g_true.sum()),
        "n_fired_true_trade": int((fired & (g_true == 1)).sum()),
    }
    tt = g_true == 1
    out["stage2_acc_on_true_trade"] = (
        float((dir_up[tt] == dir_true_up[tt]).mean()) if tt.any()
        else float("nan"))
    ft = fired & tt
    out["stage2_acc_on_fired_true_trade"] = (
        float((dir_up[ft] == dir_true_up[ft]).mean()) if ft.any()
        else float("nan"))
    return out


def _fold_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "pred_distribution": {c: float((pred == c).mean())
                              for c in CLASSES},
    }


def walk_forward_arm(df_pre: pd.DataFrame, arm: str) -> dict:
    """TimeSeriesSplit(5) walk-forward over the pre-duel frame.
    Each fold fits both stages on the fold's training rows only
    (stage 2 on those rows' TRADE subset). Returns per-fold
    end-to-end metrics + stage diagnostics, and pooled OOF metrics
    over the graded rows (the union of the 5 validation blocks)."""
    from sklearn.model_selection import TimeSeriesSplit

    cols = arm_features(df_pre, arm)
    X = df_pre[cols].to_numpy(dtype=float)
    y = df_pre[TARGET_COL].to_numpy()
    ts = df_pre.index
    tss = TimeSeriesSplit(n_splits=5)
    folds, audits = [], []
    oof_pred = np.full(len(df_pre), None, dtype=object)
    oof_pt = np.full(len(df_pre), np.nan)
    oof_pu = np.full(len(df_pre), np.nan)
    for k, (tr_idx, va_idx) in enumerate(tss.split(X)):
        assert tr_idx.max() < va_idx.min(), "fold is not chronological"
        model, audit = fit_hierarchy(X[tr_idx], y[tr_idx],
                                     ts=ts[tr_idx])
        audits.append(audit)
        pt = model.predict_proba_trade(X[va_idx])
        pu = model.predict_proba_up(X[va_idx])
        pred = end_to_end_labels(pt, pu)
        oof_pred[va_idx] = pred
        oof_pt[va_idx] = pt
        oof_pu[va_idx] = pu
        m = _fold_metrics(y[va_idx], pred)
        m["fold"] = k
        m["diagnostics"] = _stage_diagnostics(y[va_idx], pt, pu, pred)
        folds.append(m)
    graded = oof_pred != None  # noqa: E711 -- object array sentinel
    pooled = _fold_metrics(y[graded], oof_pred[graded])
    pooled["majority_baseline"] = float(
        max((y[graded] == c).mean() for c in CLASSES))
    pooled["n_graded"] = int(graded.sum())
    pooled["diagnostics"] = _stage_diagnostics(
        y[graded], oof_pt[graded], oof_pu[graded], oof_pred[graded])
    return {"arm": arm, "features": cols, "folds": folds,
            "pooled": pooled, "fold_audits": audits}


# ---------------------------------------------------------------- verdict
def rule1_walk_forward(wf: dict) -> bool:
    """Pooled end-to-end balanced accuracy > 1/3 AND >= 4/5 folds
    above 1/3 AND pooled accuracy >= the majority baseline -- the
    operationalization the composite and regime plans fixed."""
    p = wf["pooled"]
    fold_bals = [f["balanced_accuracy"] for f in wf["folds"]]
    return bool(p["balanced_accuracy"] > 1.0 / 3.0
                and sum(1 for b in fold_bals if b > 1.0 / 3.0) >= 4
                and p["accuracy"] >= p["majority_baseline"])


def rule2_duel(duel: dict) -> bool:
    """Duel: total P&L > $0 AND total P&L > buy-and-hold AND
    non-hold activity share >= 10% (fixed in the plan)."""
    return bool(duel["total_pnl"] > 0.0
                and duel["total_pnl"] > duel["bench_pnl"]
                and duel["non_hold_share"] >= 0.10)


def rule3_robustness(robust: dict) -> bool:
    return bool(robust["total_pnl"] >= 0.0)


def apply_verdict(wf: dict, duel: dict, robust: dict) -> dict:
    """All three pre-registered rules or NULL."""
    r1 = rule1_walk_forward(wf)
    r2 = rule2_duel(duel)
    r3 = rule3_robustness(robust)
    return {
        "verdict": "PASS" if (r1 and r2 and r3) else "FAIL",
        "rule1_walk_forward": r1,
        "rule2_duel": r2,
        "rule3_robustness": r3,
        "fold_balanced_accuracies":
            [f["balanced_accuracy"] for f in wf["folds"]],
    }
