"""Export Kass's HA inspection workbook from the study's HA frame.

Builds the frame exactly as the HA-Deep study does (run-82 persisted
fetch, src/training/ha_deep.build_ha_frame) and writes:
  sheet 1 "HA Candles"  — timestamp (UTC), raw O/H/L/C, HA O/H/L/C,
                          HA color (GREEN/RED/NEUTRAL), signed streak;
  sheet 2 "HA Features" — timestamp + the frozen 58 feature columns.
Formatting per Kass's readability standard (2026-10-29 restyle): 13pt
body, 16pt bold headers, wrapped headers, frozen panes. Export only —
no grading or registration behavior lives here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.ha_deep import FEATURE_COLUMNS_HA, build_ha_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path.home() / "workspace" / "your_files" / "BTC-USD_1h_HeikinAshi_Features.xlsx"

HEADER_FONT = Font(name="Calibri", size=16, bold=True, color="FFFFFF")
HEADER_FILL = PatternFill("solid", fgColor="1F3864")
BODY_FONT = Font(name="Calibri", size=13)
HEADER_ALIGN = Alignment(horizontal="center", vertical="center", wrap_text=True)
COLOR_WORD = {1: "GREEN", -1: "RED", 0: "NEUTRAL"}


def _style_header(ws, ncols: int) -> None:
    for col in range(1, ncols + 1):
        cell = ws.cell(row=1, column=col)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = HEADER_ALIGN
    ws.row_dimensions[1].height = 38


def main() -> None:
    df = pd.read_pickle(ROOT / "data" / "meta_label" / "df.pkl")
    frame, _seq, aux = build_ha_frame(df)
    assert list(frame.columns) == FEATURE_COLUMNS_HA
    ha = aux["ha"]
    colors = aux["colors"]
    streak = aux["streak"]
    n = len(df)
    stamps = [ts.to_pydatetime() for ts in df.index]

    wb = Workbook()

    # ---- sheet 1: HA candle series ----
    ws1 = wb.active
    ws1.title = "HA Candles"
    headers1 = ["Timestamp (UTC)", "Open", "High", "Low", "Close",
                "HA Open", "HA High", "HA Low", "HA Close",
                "HA Color", "HA Streak (signed)"]
    ws1.append(headers1)
    raw = df[["open", "high", "low", "close"]].to_numpy(dtype=float)
    hav = ha[["ha_open", "ha_high", "ha_low", "ha_close"]].to_numpy(dtype=float)
    for t in range(n):
        ws1.append([stamps[t], raw[t, 0], raw[t, 1], raw[t, 2], raw[t, 3],
                    hav[t, 0], hav[t, 1], hav[t, 2], hav[t, 3],
                    COLOR_WORD[int(colors[t])], int(streak[t])])
    _style_header(ws1, len(headers1))
    ws1.column_dimensions["A"].width = 21
    for col in range(2, 10):
        ws1.column_dimensions[get_column_letter(col)].width = 14
    ws1.column_dimensions["J"].width = 13
    ws1.column_dimensions["K"].width = 19
    for row in ws1.iter_rows(min_row=2, max_row=n + 1):
        for cell in row:
            cell.font = BODY_FONT
        row[0].number_format = "yyyy-mm-dd hh:mm"
        for cell in row[1:9]:
            cell.number_format = "#,##0.00"
    ws1.freeze_panes = "B2"

    # ---- sheet 2: the frozen 58-feature frame ----
    ws2 = wb.create_sheet("HA Features")
    headers2 = ["Timestamp (UTC)"] + FEATURE_COLUMNS_HA
    ws2.append(headers2)
    vals = frame.to_numpy(dtype=float)
    for t in range(n):
        row = [stamps[t]]
        row.extend(None if not np.isfinite(v) else float(v) for v in vals[t])
        ws2.append(row)
    _style_header(ws2, len(headers2))
    ws2.column_dimensions["A"].width = 21
    for col in range(2, len(headers2) + 1):
        ws2.column_dimensions[get_column_letter(col)].width = 16
    for row in ws2.iter_rows(min_row=2, max_row=n + 1):
        for cell in row:
            cell.font = BODY_FONT
        row[0].number_format = "yyyy-mm-dd hh:mm"
        for cell in row[1:]:
            cell.number_format = "0.000000"
    ws2.freeze_panes = "B2"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes); rows={n}, "
          f"sheet2 cols={len(headers2)}")


if __name__ == "__main__":
    main()
