"""Tests for the lab diagnosis battery helpers (scripts/diagnosis.py).

The ablation is only meaningful if the family grouping is an exact
partition of the 21 production features -- a feature silently dropped
or double-counted would fake a family verdict. These tests pin that
down, plus the variant construction and the second-angle aggregation.
"""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "diagnosis", ROOT / "scripts" / "diagnosis.py")
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)

from features.technical_indicators import FEATURE_COLUMNS  # noqa: E402


def test_families_partition_features_exactly():
    flat = [c for cols in diag.FAMILIES.values() for c in cols]
    assert sorted(flat) == sorted(FEATURE_COLUMNS)
    assert len(flat) == len(set(flat)) == len(FEATURE_COLUMNS)


def test_kass_five_indicators_covered():
    kass_five = {"ha_dir", "obv_slope", "adl_slope", "netvol_rel", "ewo"}
    covered = set(diag.FAMILIES["kass_indicators"]) | set(
        diag.FAMILIES["volume_flow"])
    assert kass_five <= covered


def test_variant_columns():
    variants = diag.variant_columns()
    assert variants["full"] == list(FEATURE_COLUMNS)
    for fam, cols in diag.FAMILIES.items():
        assert variants[f"only_{fam}"] == cols
        loo = variants[f"loo_{fam}"]
        assert sorted(loo + cols) == sorted(FEATURE_COLUMNS)
        assert not set(loo) & set(cols)


def test_chunk_plan():
    plan = diag.chunk_plan([])
    assert len(plan) == 12 and all(s == 25_000 for _, s in plan)
    assert sum(s for _, s in plan) == 300_000
    assert diag.chunk_plan([1, 2]) == plan[2:]
    assert diag.chunk_plan(list(range(1, 13))) == []
    assert diag.chunk_plan([], total=250_000, chunk=100_000) == [
        (1, 100_000), (2, 100_000), (3, 50_000)]


def test_aggregate_from_windows_is_plain_mean():
    ws = [{"score": 10.0}, {"score": -4.0}, {"score": 3.0}]
    assert diag.aggregate_from_windows(ws) == float(np.mean([10.0, -4.0, 3.0]))
