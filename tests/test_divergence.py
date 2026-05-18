"""Tests for :mod:`analysis.divergence`.

Synthesize small rep1/rep2 CSV pairs (one per RQ), call the comparison
helpers, and verify the delta arithmetic + within-band booleans + the
summarize() aggregate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.divergence import (
    compare_rq1,
    compare_rq2,
    compare_rq3,
    compare_rq4,
    main,
    summarize,
)


def _write_rq1_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_compare_rq1_inner_joins_on_config_tuple(tmp_path: Path) -> None:
    rep1 = tmp_path / "rq1_summary_rep1.csv"
    rep2 = tmp_path / "rq1_summary_rep2.csv"
    _write_rq1_csv(
        rep1,
        [
            {
                "p4_program": "l3_lpm",
                "packet_size_bytes": 256,
                "background_load_mbps": 1,
                "cold_idle_reference": False,
                "n_samples": 1000,
                "median_us": 120.0,
            },
            {
                "p4_program": "l3_lpm",
                "packet_size_bytes": 256,
                "background_load_mbps": 25,
                "cold_idle_reference": False,
                "n_samples": 1000,
                "median_us": 100.0,
            },
        ],
    )
    _write_rq1_csv(
        rep2,
        [
            {
                "p4_program": "l3_lpm",
                "packet_size_bytes": 256,
                "background_load_mbps": 1,
                "cold_idle_reference": False,
                "n_samples": 1000,
                "median_us": 130.0,
            },
            {
                "p4_program": "l3_lpm",
                "packet_size_bytes": 256,
                "background_load_mbps": 25,
                "cold_idle_reference": False,
                "n_samples": 1000,
                "median_us": 95.0,
            },
        ],
    )
    df = compare_rq1(rep1, rep2)
    assert len(df) == 2
    row_1m = df[df["background_load_mbps"] == 1].iloc[0]
    assert row_1m["delta_us"] == 10.0
    assert row_1m["delta_pct_us"] == pytest.approx(8.333, abs=0.01)
    assert not row_1m["within_5pct_us"]
    assert row_1m["within_10pct_us"]
    assert row_1m["within_20pct_us"]


def test_compare_rq2_arithmetic(tmp_path: Path) -> None:
    rep1 = tmp_path / "rq2_summary_rep1.csv"
    rep2 = tmp_path / "rq2_summary_rep2.csv"
    pd.DataFrame(
        [
            {
                "n_switches": 8,
                "n_entries_per_switch": 1000,
                "operation": "insert",
                "mode": "async",
                "median_s": 2.1,
            },
        ]
    ).to_csv(rep1, index=False)
    pd.DataFrame(
        [
            {
                "n_switches": 8,
                "n_entries_per_switch": 1000,
                "operation": "insert",
                "mode": "async",
                "median_s": 2.0,
            },
        ]
    ).to_csv(rep2, index=False)
    df = compare_rq2(rep1, rep2)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["delta_pct_s"] == pytest.approx(-4.762, abs=0.01)
    assert row["within_5pct_s"]


def test_compare_rq3_uses_abs_mean(tmp_path: Path) -> None:
    rep1 = tmp_path / "rq3_summary_rep1.csv"
    rep2 = tmp_path / "rq3_summary_rep2.csv"
    pd.DataFrame(
        [
            {"n_switches": 2, "background_load_mbps": 0, "abs_mean_us": 5000.0},
        ]
    ).to_csv(rep1, index=False)
    pd.DataFrame(
        [
            {"n_switches": 2, "background_load_mbps": 0, "abs_mean_us": 5200.0},
        ]
    ).to_csv(rep2, index=False)
    df = compare_rq3(rep1, rep2)
    assert len(df) == 1
    row = df.iloc[0]
    # 200/5000 = 4%
    assert row["within_5pct_us"]
    assert row["delta_pct_us"] == pytest.approx(4.0)


def test_compare_rq4_groups_by_full_key(tmp_path: Path) -> None:
    rep1 = tmp_path / "rq4_summary_rep1.csv"
    rep2 = tmp_path / "rq4_summary_rep2.csv"
    base = {
        "p4_program": "l3_lpm",
        "topology": "linear_n",
        "n_switches": 4,
        "background_load_mbps": 25,
        "source_workload_type": "resource_only",
        "metric": "cpu_percent_total",
    }
    pd.DataFrame([{**base, "mean": 50.0}]).to_csv(rep1, index=False)
    pd.DataFrame([{**base, "mean": 55.0}]).to_csv(rep2, index=False)
    df = compare_rq4(rep1, rep2)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["delta_pct_value"] == pytest.approx(10.0)
    assert not row["within_5pct_value"]
    assert row["within_10pct_value"]


def test_summarize_counts_bands_and_flags(tmp_path: Path) -> None:
    rep1 = tmp_path / "rq1_summary_rep1.csv"
    rep2 = tmp_path / "rq1_summary_rep2.csv"
    base = {
        "p4_program": "l3_lpm",
        "packet_size_bytes": 256,
        "cold_idle_reference": False,
    }
    pd.DataFrame(
        [
            {**base, "background_load_mbps": 1, "median_us": 100.0},
            {**base, "background_load_mbps": 25, "median_us": 100.0},
        ]
    ).to_csv(rep1, index=False)
    pd.DataFrame(
        [
            {**base, "background_load_mbps": 1, "median_us": 105.0},
            {**base, "background_load_mbps": 25, "median_us": 130.0},
        ]
    ).to_csv(rep2, index=False)
    df = compare_rq1(rep1, rep2)
    out = summarize({"rq1": df})
    stats = out["per_rq"]["rq1"]
    assert stats["n_configs"] == 2
    assert stats["pct_within_5"] == 50.0
    assert stats["pct_within_10"] == 50.0
    assert stats["pct_within_20"] == 50.0
    assert stats["max_abs_delta_pct"] == pytest.approx(30.0)
    assert len(out["flagged_configs"]) == 1
    assert out["flagged_configs"][0]["delta_pct"] == pytest.approx(30.0)


def test_main_writes_report_and_summary(tmp_path: Path) -> None:
    sd = tmp_path / "summaries"
    sd.mkdir()
    base = {
        "p4_program": "l3_lpm",
        "packet_size_bytes": 256,
        "background_load_mbps": 1,
        "cold_idle_reference": False,
    }
    pd.DataFrame([{**base, "median_us": 120.0}]).to_csv(sd / "rq1_summary_rep1.csv", index=False)
    pd.DataFrame([{**base, "median_us": 130.0}]).to_csv(sd / "rq1_summary_rep2.csv", index=False)
    rc = main(["--summary", str(sd)])
    assert rc == 0
    assert (sd / "divergence_report.csv").is_file()
    summary_path = sd / "divergence_summary.json"
    assert summary_path.is_file()
    blob = json.loads(summary_path.read_text())
    assert "rq1" in blob["per_rq"]
    assert blob["per_rq"]["rq1"]["n_configs"] == 1


def test_main_skips_missing_pairs(tmp_path: Path) -> None:
    """If only rep1 exists for a given RQ, it's silently skipped."""
    sd = tmp_path / "summaries"
    sd.mkdir()
    # Only RQ1 has a pair; RQ2/3/4 are missing.
    pd.DataFrame(
        [
            {
                "p4_program": "l3_lpm",
                "packet_size_bytes": 256,
                "background_load_mbps": 1,
                "cold_idle_reference": False,
                "median_us": 100.0,
            }
        ]
    ).to_csv(sd / "rq1_summary_rep1.csv", index=False)
    pd.DataFrame(
        [
            {
                "p4_program": "l3_lpm",
                "packet_size_bytes": 256,
                "background_load_mbps": 1,
                "cold_idle_reference": False,
                "median_us": 102.0,
            }
        ]
    ).to_csv(sd / "rq1_summary_rep2.csv", index=False)
    rc = main(["--summary", str(sd)])
    assert rc == 0
