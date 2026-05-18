"""Unit tests for :mod:`analysis.aggregate`.

Tests build a small in-memory list of JSONL-shaped records, call the
per-RQ aggregators directly, and assert the resulting DataFrame has
the expected columns and statistics. One end-to-end test glob-loads
the harness's actual ``data/raw/`` directory if any JSONL is present
and confirms the CLI produces non-empty CSVs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.aggregate import (
    METRIC_RQ1,
    METRIC_RQ2,
    METRIC_RQ3,
    aggregate_experiment_log,
    aggregate_rq1,
    aggregate_rq2,
    aggregate_rq3,
    aggregate_rq4,
    main,
)


def _rq1_record(prog: str, size: int, load: int, value: float, cold: bool = False) -> dict:
    return {
        "run_id": "test",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "rq": 1,
        "config": {
            "p4_program": prog,
            "packet_size_bytes": size,
            "background_load_mbps": load,
            "probe_layer": "l3",
            "cold_idle_reference": cold,
            "repetition": 0,
        },
        "metric": METRIC_RQ1,
        "value": value,
        "extras": {"sequence": 0, "ingress_ts_us": 0, "egress_ts_us": int(value)},
    }


def test_aggregate_rq1_groups_by_config_tuple() -> None:
    records = [
        _rq1_record("l3_lpm", 256, 25, 100.0),
        _rq1_record("l3_lpm", 256, 25, 110.0),
        _rq1_record("l3_lpm", 256, 25, 120.0),
        _rq1_record("l3_lpm", 256, 45, 200.0),
    ]
    df = aggregate_rq1(records)
    assert len(df) == 2  # two distinct (size, load) cells
    row_25 = df[df["background_load_mbps"] == 25].iloc[0]
    assert row_25["n_samples"] == 3
    assert row_25["median_us"] == 110.0
    assert row_25["p99_us"] >= 100.0


def test_aggregate_rq1_distinguishes_cold_idle() -> None:
    records = [
        _rq1_record("l3_lpm", 256, 0, 600.0, cold=True),
        _rq1_record("l3_lpm", 256, 0, 650.0, cold=True),
        _rq1_record("l3_lpm", 256, 1, 150.0, cold=False),
    ]
    df = aggregate_rq1(records)
    assert len(df) == 2
    cold = df[df["cold_idle_reference"]].iloc[0]
    warm = df[~df["cold_idle_reference"]].iloc[0]
    assert cold["median_us"] == 625.0
    assert warm["median_us"] == 150.0


def test_aggregate_rq2_collects_reps_and_eps() -> None:
    records = [
        {
            "run_id": "t",
            "timestamp_utc": "2026-01-01T00:00:00Z",
            "rq": 2,
            "config": {
                "n_switches": 2,
                "n_entries_per_switch": 100,
                "operation": "insert",
                "mode": "async",
                "repetition": 0,
            },
            "metric": METRIC_RQ2,
            "value": 0.05,
            "extras": {"success_count": 200, "failure_count": 0, "entries_per_second": 4000.0},
        },
        {
            "run_id": "t",
            "timestamp_utc": "2026-01-01T00:00:00Z",
            "rq": 2,
            "config": {
                "n_switches": 2,
                "n_entries_per_switch": 100,
                "operation": "insert",
                "mode": "async",
                "repetition": 1,
            },
            "metric": METRIC_RQ2,
            "value": 0.07,
            "extras": {"success_count": 200, "failure_count": 0, "entries_per_second": 2857.0},
        },
    ]
    df = aggregate_rq2(records)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["n_reps"] == 2
    assert row["median_s"] == pytest.approx(0.06)
    assert row["mean_s"] == pytest.approx(0.06)
    assert row["median_entries_per_sec"] == pytest.approx(3428.5)


def test_aggregate_rq3_includes_abs_stats_for_signed_drift() -> None:
    records = [
        {
            "run_id": "t",
            "timestamp_utc": "2026-01-01T00:00:00Z",
            "rq": 3,
            "config": {
                "n_switches": 2,
                "background_load_mbps": 0,
                "packet_size_bytes": 256,
                "repetition": 0,
            },
            "metric": METRIC_RQ3,
            "value": -3000.0,
            "extras": {"drift_us": [-3000]},
        },
        {
            "run_id": "t",
            "timestamp_utc": "2026-01-01T00:00:00Z",
            "rq": 3,
            "config": {
                "n_switches": 2,
                "background_load_mbps": 0,
                "packet_size_bytes": 256,
                "repetition": 0,
            },
            "metric": METRIC_RQ3,
            "value": -1000.0,
            "extras": {"drift_us": [-1000]},
        },
    ]
    df = aggregate_rq3(records)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["n_samples"] == 2
    assert row["mean_us"] == -2000.0
    # abs stats should not be negative
    assert row["abs_mean_us"] == 2000.0
    assert row["per_hop_n"] == 2


def test_aggregate_rq4_groups_by_metric_and_config() -> None:
    base_cfg = {
        "p4_program": "l3_lpm",
        "topology": "linear_n",
        "n_switches": 4,
        "background_load_mbps": 25,
        "source_workload_type": "resource_only",
        "repetition": 0,
        "sample_index": 0,
    }

    def rec(metric: str, value: float, idx: int) -> dict:
        cfg = dict(base_cfg, sample_index=idx)
        return {
            "run_id": "t",
            "timestamp_utc": "2026-01-01T00:00:00Z",
            "rq": 4,
            "config": cfg,
            "metric": metric,
            "value": value,
            "extras": {},
        }

    records = [
        rec("cpu_percent_total", 50.0, 0),
        rec("cpu_percent_total", 70.0, 1),
        rec("cpu_percent_total", 60.0, 2),
        rec("rss_per_bmv2_bytes", 100e6, 0),
        rec("rss_per_bmv2_bytes", 110e6, 1),
    ]
    df = aggregate_rq4(records)
    assert len(df) == 2  # two distinct (metric) entries for the same config
    cpu_row = df[df["metric"] == "cpu_percent_total"].iloc[0]
    assert cpu_row["mean"] == 60.0
    assert cpu_row["max"] == 70.0
    rss_row = df[df["metric"] == "rss_per_bmv2_bytes"].iloc[0]
    assert rss_row["n_samples"] == 2


def test_experiment_log_records_one_row_per_file(tmp_path: Path) -> None:
    f1 = tmp_path / "a.jsonl"
    f1.write_text(
        json.dumps(_rq1_record("l3_lpm", 256, 1, 100.0))
        + "\n"
        + json.dumps(_rq1_record("l3_lpm", 256, 1, 110.0))
        + "\n"
    )
    f2 = tmp_path / "b.jsonl"
    f2.write_text(json.dumps(_rq1_record("l3_lpm_int", 256, 25, 130.0)) + "\n")
    file_records = [(f1, list(_iter_jsonl(f1))), (f2, list(_iter_jsonl(f2)))]
    df = aggregate_experiment_log([f1, f2], file_records)
    assert len(df) == 2
    assert int(df[df["source_file"] == "a.jsonl"]["rq1_records"].iloc[0]) == 2
    assert int(df[df["source_file"] == "b.jsonl"]["rq1_records"].iloc[0]) == 1


def _iter_jsonl(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def test_main_end_to_end_writes_csvs(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    summary = tmp_path / "summary"
    raw.mkdir()
    # Plant a tiny RQ1+RQ2 JSONL.
    (raw / "test.jsonl").write_text(
        json.dumps(_rq1_record("l3_lpm", 256, 1, 100.0))
        + "\n"
        + json.dumps(_rq1_record("l3_lpm", 256, 1, 110.0))
        + "\n"
    )
    rc = main(["--raw", str(raw), "--summary", str(summary)])
    assert rc == 0
    for name in ("rq1_summary", "rq2_summary", "rq3_summary", "rq4_summary", "experiment_log"):
        assert (summary / f"{name}.csv").is_file(), f"missing {name}.csv"
    # RQ1 should have a row; the others may be empty but the file must exist.
    rq1_csv = (summary / "rq1_summary.csv").read_text()
    assert "l3_lpm" in rq1_csv


def test_main_label_flag_appends_suffix(tmp_path: Path) -> None:
    """``--label rep2`` writes ``rq1_summary_rep2.csv`` etc."""
    raw = tmp_path / "raw"
    summary = tmp_path / "summary"
    raw.mkdir()
    (raw / "test.jsonl").write_text(json.dumps(_rq1_record("l3_lpm", 256, 1, 100.0)) + "\n")
    rc = main(["--raw", str(raw), "--summary", str(summary), "--label", "rep2"])
    assert rc == 0
    for name in ("rq1_summary", "rq2_summary", "rq3_summary", "rq4_summary", "experiment_log"):
        assert (summary / f"{name}_rep2.csv").is_file(), f"missing {name}_rep2.csv"
        # Unsuffixed name must NOT exist when --label is set.
        assert not (summary / f"{name}.csv").is_file(), f"unsuffixed {name}.csv leaked"


def test_main_label_default_writes_unsuffixed(tmp_path: Path) -> None:
    """No ``--label`` arg → unsuffixed names (back-compat with pre-Phase-H)."""
    raw = tmp_path / "raw"
    summary = tmp_path / "summary"
    raw.mkdir()
    (raw / "test.jsonl").write_text(json.dumps(_rq1_record("l3_lpm", 256, 1, 100.0)) + "\n")
    rc = main(["--raw", str(raw), "--summary", str(summary)])
    assert rc == 0
    assert (summary / "rq1_summary.csv").is_file()
    assert not (summary / "rq1_summary_.csv").is_file()
    assert not (summary / "rq1_summary_rep2.csv").is_file()


def test_main_label_preserves_aggregation_results(tmp_path: Path) -> None:
    """Running with --label and without should produce identical content,
    only the filename differs. Aggregation logic must be label-agnostic."""
    raw = tmp_path / "raw"
    raw.mkdir()
    summary_a = tmp_path / "a"
    summary_b = tmp_path / "b"
    (raw / "test.jsonl").write_text(
        json.dumps(_rq1_record("l3_lpm", 256, 1, 100.0))
        + "\n"
        + json.dumps(_rq1_record("l3_lpm", 256, 25, 90.0))
        + "\n"
    )
    main(["--raw", str(raw), "--summary", str(summary_a)])
    main(["--raw", str(raw), "--summary", str(summary_b), "--label", "rep2"])
    plain = (summary_a / "rq1_summary.csv").read_text()
    labeled = (summary_b / "rq1_summary_rep2.csv").read_text()
    assert plain == labeled
