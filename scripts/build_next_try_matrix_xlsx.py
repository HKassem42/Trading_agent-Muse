"""Build Muse_Next_Try_Matrix.xlsx -- the design + 120-column
feature matrix + clean-era leaderboard (+ results when graded).

Saved to the goal's Library files folder and the project's refs/.
Run anytime; the Results sheet fills in from
data/muse_next_try/results.json when the study has graded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Font, PatternFill  # noqa: E402

from features import next_try_features as nt  # noqa: E402
from features import pick2_features, scored_features  # noqa: E402

NAVY = PatternFill("solid", fgColor="0D1524")
GREEN = PatternFill("solid", fgColor="3DDC84")
TITLE = Font(bold=True, size=14, color="FFFFFF")
HEAD = Font(bold=True, color="FFFFFF")
BOLD = Font(bold=True)

OUTS = [
    ROOT / "refs" / "Muse_Next_Try_Matrix.xlsx",
    Path.home() / "workspace" / "goals" /
    "tradingagent-muse-learning-trading-bot" / "files" /
    "Muse_Next_Try_Matrix.xlsx",
]

INFO_SPEC = [
    ("Funding Rate (last settled)", "Binance Vision fundingRate (monthly)",
     "Last settled funding rate, settlement <= t",
     "Trailing-4320 percentile score"),
    ("Funding Rate 24h Mean", "Binance Vision fundingRate (monthly)",
     "Mean of settlements in (t-24h, t]", "Trailing-4320 percentile score"),
    ("Open Interest Level", "Binance Vision metrics (daily, 5m)",
     "Last sum_open_interest < t+1h", "Trailing-4320 percentile score"),
    ("Open Interest Change 1h %", "Binance Vision metrics (daily, 5m)",
     "OI(last < t+1h) / OI(last < t) - 1", "Trailing-4320 percentile score"),
    ("Open Interest Change 24h %", "Binance Vision metrics (daily, 5m)",
     "OI(last < t+1h) / OI(last < t+1h-24h) - 1",
     "Trailing-4320 percentile score"),
    ("Long/Short Ratio (all accounts)", "Binance Vision metrics (daily, 5m)",
     "count_long_short_ratio, last obs < t+1h",
     "Trailing-4320 percentile score"),
    ("Top Trader Account L/S Ratio", "Binance Vision metrics (daily, 5m)",
     "count_toptrader_long_short_ratio, last obs < t+1h",
     "Trailing-4320 percentile score"),
    ("Top Trader Position L/S Ratio", "Binance Vision metrics (daily, 5m)",
     "sum_toptrader_long_short_ratio, last obs < t+1h",
     "Trailing-4320 percentile score"),
    ("Taker L/S Volume Ratio (1h)", "Binance Vision metrics (daily, 5m)",
     "Mean of sum_taker_long_short_vol_ratio inside candle t",
     "Trailing-4320 percentile score"),
    ("Perp Basis %", "Binance Vision perp 1h klines + Coinbase spot",
     "perp close_t / spot close_t - 1", "Trailing-4320 percentile score"),
    ("Perp Taker Buy Share", "Binance Vision perp 1h klines",
     "taker_buy_volume / volume, candle t",
     "Trailing-4320 percentile score"),
    ("Perp/Spot Volume Ratio", "Binance Vision perp 1h + Coinbase spot",
     "perp volume_t / spot volume_t", "Trailing-4320 percentile score"),
    ("ETH Return 1h %", "Coinbase ETH-USD 1h (project adapter)",
     "ETH close_t / close_{t-1} - 1", "Trailing-4320 percentile score"),
    ("ETH/BTC Ratio Change 24h %", "Coinbase ETH-USD + BTC-USD 1h",
     "(ETH/BTC)_t / (ETH/BTC)_{t-24} - 1",
     "Trailing-4320 percentile score"),
]
SEQ_SPEC = [
    ("Master Score Lag 6h", "Pick #2 frame (scored-102 composite)",
     "score_master at t-6h", "Already a 0..100 score"),
    ("Master Score Lag 24h", "Pick #2 frame (scored-102 composite)",
     "score_master at t-24h", "Already a 0..100 score"),
    ("Master Score Change 24h", "Pick #2 frame (scored-102 composite)",
     "(score_master_t - score_master_{t-24} + 100) / 2",
     "Rescaled to 0..100"),
    ("Prior Candle Return %", "Coinbase BTC-USD 1h",
     "log return of candle t-1", "Trailing-4320 percentile score"),
]
CAL_SPEC = [
    ("Hour of Day Sin", "Candle timestamp (UTC)", "sin(2*pi*hour/24)",
     "(sin+1)*50 -> 0..100"),
    ("Hour of Day Cos", "Candle timestamp (UTC)", "cos(2*pi*hour/24)",
     "(cos+1)*50 -> 0..100"),
    ("Day of Week Sin", "Candle timestamp (UTC)", "sin(2*pi*weekday/7)",
     "(sin+1)*50 -> 0..100"),
    ("Day of Week Cos", "Candle timestamp (UTC)", "cos(2*pi*weekday/7)",
     "(cos+1)*50 -> 0..100"),
]


def _head(ws, row, cols):
    for j, c in enumerate(cols, start=1):
        cell = ws.cell(row=row, column=j, value=c)
        cell.font = HEAD
        cell.fill = NAVY


def main() -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Design"
    ws["A1"] = "Muse's Next Try — Design"
    ws["A1"].font = Font(bold=True, size=16)
    rows = [
        ("Ordered by", "Kass, 2026-11-03 ~23:56 ET — 'pick my next model "
         "from start to finish... beat your previous try... Label it "
         "as ur next try. GO!'"),
        ("Previous try", "Muse's Pick #2 (run 73): holdout 0.5354, "
         "precision 0.5250, walk-forward 0.5177 — the best clean-era "
         "stacker Muse has produced."),
        ("Hypothesis", "Every prior null read BTC's own candles. The "
         "untested lever is NEW information: derivatives positioning "
         "(funding / open interest / long-short ratios / taker flow / "
         "basis), ETH cross-asset moves, the Master Score's own "
         "trajectory, and calendar cyclicals."),
        ("Frame", "Pick #2's 98 scored eyes + 22 new causal columns = "
         "120 (see Feature Matrix sheet)."),
        ("Recipe", "Muse's Pick recipe: StandardScaler -> SMOTE -> "
         "RandomForest + XGBoost + LightGBM + ExtraTrees -> "
         "LogisticRegression."),
        ("Selection", "Frame chosen by train-only CV screening (fast "
         "ExtraTrees+LightGBM proxy, 3-fold TimeSeriesSplit). Decision "
         "threshold chosen on train out-of-fold votes only, used only "
         "if it beats 0.50 by >= 0.005 OOF accuracy. The 792-candle "
         "holdout is touched once, at final grading."),
        ("Causality rule", "A feature at row t uses only external "
         "observations timestamped < t+1h (through candle t's close — "
         "the same footing as the spot features). Funding: last "
         "settlement <= t. Pinned by truncation-rebuild tests over "
         "ALL rows."),
        ("Verdict rule", "WIN = holdout accuracy strictly beats BOTH "
         "champions re-scored on the identical holdout (run 96 prod21 "
         "and run 73 pick2_98, two independent ways each) with "
         "non-degenerate anatomy (up-calls > 7, precision > 0.05). "
         "A win is PENDING KASS APPROVAL under the promotion lock; "
         "nothing installs."),
        ("Pre-registration", "refs/muse-next-try-preregistration-"
         "2026-11-04.md (frozen before any grading)."),
    ]
    r = 3
    for k, v in rows:
        ws.cell(row=r, column=1, value=k).font = BOLD
        ws.cell(row=r, column=2, value=v)
        r += 1
    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 110

    ws = wb.create_sheet("Feature Matrix")
    ws["A1"] = "Muse's Next Try — 120-column feature matrix"
    ws["A1"].font = Font(bold=True, size=14)
    _head(ws, 3, ["#", "Column", "Family", "Source", "Definition",
                  "Transform"])
    r = 4
    base = list(scored_features.scored_columns()) + list(
        pick2_features.NEW_COLUMNS)
    for i, c in enumerate(base, start=1):
        fam = ("Scored-102 base" if i <= 94 else "Pick #2 moments")
        ws.cell(row=r, column=1, value=i)
        ws.cell(row=r, column=2, value=c)
        ws.cell(row=r, column=3, value=fam)
        ws.cell(row=r, column=4, value="Matrix-v3 scored frame / "
                "pick2_features")
        ws.cell(row=r, column=5, value="Trailing-4320 percentile score "
                "of the indicator (Champion #4 frame)")
        ws.cell(row=r, column=6, value="0..100 score")
        r += 1
    for name, src, defn, tr in INFO_SPEC:
        ws.cell(row=r, column=1, value=r - 3)
        ws.cell(row=r, column=2, value=name)
        ws.cell(row=r, column=3, value="NEW — derivatives/cross-asset")
        ws.cell(row=r, column=4, value=src)
        ws.cell(row=r, column=5, value=defn)
        ws.cell(row=r, column=6, value=tr)
        r += 1
    for name, src, defn, tr in SEQ_SPEC:
        ws.cell(row=r, column=1, value=r - 3)
        ws.cell(row=r, column=2, value=name)
        ws.cell(row=r, column=3, value="NEW — sequence")
        ws.cell(row=r, column=4, value=src)
        ws.cell(row=r, column=5, value=defn)
        ws.cell(row=r, column=6, value=tr)
        r += 1
    for name, src, defn, tr in CAL_SPEC:
        ws.cell(row=r, column=1, value=r - 3)
        ws.cell(row=r, column=2, value=name)
        ws.cell(row=r, column=3, value="NEW — calendar")
        ws.cell(row=r, column=4, value=src)
        ws.cell(row=r, column=5, value=defn)
        ws.cell(row=r, column=6, value=tr)
        r += 1
    for col, w in zip("ABCDEF", (5, 44, 26, 42, 52, 30)):
        ws.column_dimensions[col].width = w

    ws = wb.create_sheet("Leaderboard (clean era)")
    ws["A1"] = "Clean-era stacker leaderboard — the bar to beat " \
               "(pre-fix runs 1-15 excluded: look-ahead bug)"
    ws["A1"].font = Font(bold=True, size=13)
    _head(ws, 3, ["Run", "Model", "Holdout acc", "Precision",
                  "Walk-forward acc", "Status"])
    board = [
        (73, "Muse's Pick #2 (scored-102 + Hurst/moments)",
         0.5354, 0.5250, 0.5177, "Champion #4 (former); Book #2 eyes"),
        (78, "Nightly challenger (prod21)", 0.5354, 0.5297, 0.4968,
         "Declined by Muse under Kass's delegation"),
        (96, "Nightly challenger (prod21)", 0.5341, 0.5298, 0.5006,
         "Lab Stacker Champion #5 (current)"),
        (80, "Nightly challenger (prod21)", 0.5303, 0.5213, 0.5005,
         "BACK BURNER (Kass)"),
        (69, "Muse's Pick #1 (scored-102 + ExtraTrees)",
         0.5265, 0.5167, 0.5022, "Champion #3 (former)"),
        (94, "Weekly deep challenger (prod21)", 0.5253, 0.5209, 0.5070,
         "Did not beat champion"),
        (98, "Nightly challenger (prod21)", 0.5152, 0.5080, 0.5124,
         "Did not beat champion"),
    ]
    r = 4
    for row in board:
        for j, v in enumerate(row, start=1):
            ws.cell(row=r, column=j, value=v)
        r += 1
    for col, w in zip("ABCDEF", (6, 46, 13, 11, 17, 44)):
        ws.column_dimensions[col].width = w

    ws = wb.create_sheet("Results")
    ws["A1"] = "Muse's Next Try — Results"
    ws["A1"].font = Font(bold=True, size=14)
    res_path = ROOT / "data" / "muse_next_try" / "results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    fin = res.get("final")
    r = 3
    if not fin:
        ws.cell(row=r, column=1,
                value="Grading has not completed yet — this sheet "
                      "fills in from data/muse_next_try/results.json")
    else:
        rows = [
            ("Verdict", "WIN — pending Kass approval"
             if fin["effective_win"] else "NO WIN"),
            ("Selected frame (train-CV only)", fin["selected_frame"]),
            ("Features", fin["n_features"]),
            ("Decision threshold (train OOF)", fin["theta"]),
            ("Holdout accuracy", fin["acc"]),
            ("Holdout accuracy @ 0.50", fin["acc_at_50"]),
            ("Holdout precision", fin["precision"]),
            ("Up-calls (of 792)", fin["up_calls"]),
            ("Walk-forward accuracy", fin["wf_acc"]),
            ("Window accuracies (3x264)", str(fin["window_accs"])),
            ("Run 96 re-scored (same holdout)", fin["champion96"]["acc"]),
            ("Run 73 re-scored (same holdout)", fin["champion73"]["acc"]),
            ("Control (98 eyes) holdout", fin["control98"]["acc"]),
            ("Majority-class baseline", fin["majority_baseline"]),
            ("Screening (train CV)", json.dumps(
                {k: round(v["mean_acc"], 4)
                 for k, v in res["screen"]["scores"].items()})),
            ("Top features", ", ".join(
                t["feature"] for t in fin["top15"][:8])),
        ]
        res2_path = ROOT / "data" / "muse_next_try2" / "results.json"
        if res2_path.exists():
            f2 = json.loads(res2_path.read_text())
            rows += [
                ("— Try #2 (run 102) —", ""),
                ("Try #2 verdict", "WIN — pending Kass approval"
                 if f2["effective_win"] else "NO WIN"),
                ("Try #2 selection", "Pick recipe's own 3-fold train "
                 "walk-forward: control 98 = 0.5039 (selected), "
                 "full 120 = 0.5028, info 112 = 0.4937"),
                ("Try #2 frame", f2["selected_frame"]),
                ("Try #2 holdout accuracy", f2["acc"]),
                ("Try #2 holdout precision", f2["precision"]),
                ("Try #2 walk-forward (5-fold)", f2["wf_acc"]),
                ("Try #2 window accuracies", str(f2["window_accs"])),
            ]
        for k, v in rows:
            ws.cell(row=r, column=1, value=k).font = BOLD
            ws.cell(row=r, column=2, value=v)
            r += 1
    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 90

    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        wb.save(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
