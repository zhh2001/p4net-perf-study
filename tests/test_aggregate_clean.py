"""Tests for strict two-level aggregation of the clean Concern 4 campaigns."""

from __future__ import annotations

import copy
import hashlib
import json
import statistics
from pathlib import Path

import pytest

from analysis.aggregate import METRIC_RQ1, METRIC_RQ2, METRIC_RQ3, METRIC_RQ4_SET
from analysis.aggregate_clean import (
    aggregate_clean_campaign,
    aggregate_clean_rq1_runs,
    aggregate_clean_rq2_runs,
    aggregate_clean_rq3_runs,
    aggregate_clean_rq4_runs,
    main,
    validate_clean_manifest,
    validate_manifest_source_hashes,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _rq1_records(*, run_id: str = "rq1-clean") -> list[dict]:
    records: list[dict] = []
    for repetition in range(5):
        values = [1.0, 2.0, 3.0] if repetition == 0 else [
            repetition * 10.0 + offset for offset in (1, 2, 3, 4)
        ]
        for offset, value in enumerate(values):
            records.append(
                {
                    "run_id": run_id,
                    "rq": 1,
                    "config": {
                        "p4_program": "l3_lpm",
                        "packet_size_bytes": 256,
                        "background_load_mbps": 25,
                        "probe_layer": "l3",
                        "cold_idle_reference": False,
                        "n_probes": 4,
                        "probe_interval_ms": 60.0,
                        "repetition": repetition,
                        "schedule_index": repetition,
                    },
                    "metric": METRIC_RQ1,
                    "value": value,
                    "extras": {"sequence": repetition * 4 + offset},
                }
            )
    return records


def _rq2_records() -> list[dict]:
    return [
        {
            "run_id": "rq2-clean",
            "rq": 2,
            "config": {
                "p4_program": "l3_lpm",
                "topology": "linear_n",
                "n_switches": 2,
                "n_entries_per_switch": 100,
                "operation": "insert",
                "mode": "async",
                "repetition": repetition,
                "schedule_index": repetition,
            },
            "metric": METRIC_RQ2,
            "value": 0.1 + repetition * 0.01,
            "extras": {
                "success_count": 200,
                "failure_count": 0,
                "entries_per_second": 200 / (0.1 + repetition * 0.01),
            },
        }
        for repetition in range(5)
    ]


def _rq3_records() -> list[dict]:
    records: list[dict] = []
    for repetition in range(5):
        for offset, value in enumerate((-2.0, -1.0, 1.0, 2.0)):
            records.append(
                {
                    "run_id": "rq3-clean",
                    "rq": 3,
                    "config": {
                        "p4_program": "l3_lpm_int_chain",
                        "topology": "linear_n",
                        "n_switches": 3,
                        "background_load_mbps": 25,
                        "packet_size_bytes": 256,
                        "n_probes": 4,
                        "probe_interval_ms": 60.0,
                        "repetition": repetition,
                        "schedule_index": repetition,
                    },
                    "metric": METRIC_RQ3,
                    "value": value + repetition,
                    "extras": {
                        "sequence": repetition * 4 + offset,
                        "hop_count": 3,
                        "switch_ids": [1, 2, 3],
                        "raw_ingress_us": [10, 20, 30],
                        "raw_egress_us": [15, 25, 35],
                        "boot_us": [100, 200, 300],
                        "aligned_ingress_us": [110, 220, 330],
                        "aligned_egress_us": [115, 225, 335],
                        "drift_us": [
                            value + repetition - 0.5,
                            value + repetition + 0.5,
                        ],
                    },
                }
            )
    return records


def _rq4_records() -> list[dict]:
    records: list[dict] = []
    for repetition in range(5):
        for metric in METRIC_RQ4_SET:
            for sample_index in range(3):
                timestamp_us = repetition * 1_000_000 + sample_index * 100_000
                extras: dict = {"timestamp_us": timestamp_us}
                if metric in {"cpu_percent_per_bmv2", "rss_per_bmv2_bytes"}:
                    extras["per_pid"] = {"123": repetition + sample_index + 1}
                elif metric == "net_io_pps_per_iface":
                    extras["per_iface"] = {
                        "s1-eth1": {"rx_pps": repetition + sample_index + 1}
                    }
                records.append(
                    {
                        "run_id": "rq4-clean",
                        "rq": 4,
                        "config": {
                            "p4_program": "l3_lpm",
                            "topology": "single_switch",
                            "n_switches": 1,
                            "background_load_mbps": 45,
                            "rate_mbps": 45,
                            "source_workload_type": "resource_only",
                            "source_rq": 4,
                            "duration_s": 60.0,
                            "resource_sample_interval_s": 0.1,
                            "repetition": repetition,
                            "schedule_index": repetition,
                            "sample_index": sample_index,
                        },
                        "metric": metric,
                        "value": float(repetition + sample_index + 1),
                        "extras": extras,
                    }
                )
    return records


def _rq1_manifest() -> dict:
    config = {
        "rq": 1,
        "workload_type": "latency_l3",
        "p4_program": "l3_lpm",
        "packet_size_bytes": 256,
        "background_load_mbps": 25,
        "n_probes": 4,
        "probe_interval_ms": 60.0,
        "repetitions": 5,
    }
    return {
        "schema_version": 1,
        "run_id": "rq1-clean",
        "campaign": {
            "campaign": {"name": "rq1-test", "shuffle_all_cells": True},
            "configs": [config],
        },
        "scheduled_cells": [
            {
                "schedule_index": repetition,
                "repetition": repetition,
                "config": config,
            }
            for repetition in range(5)
        ],
        "sha256": {
            "analysis/aggregate_clean.py": hashlib.sha256(
                (REPO_ROOT / "analysis" / "aggregate_clean.py").read_bytes()
            ).hexdigest()
        },
    }


def _write_test_completion(
    *, raw_path: Path, manifest_path: Path, completion_path: Path
) -> None:
    system_info_path = completion_path.parent / "system_info.json"
    system_info_path.write_text('{"bmv2_version": "test"}\n', encoding="utf-8")
    runner_log_path = completion_path.parent / "runner.log"
    runner_log_path.write_text("complete\n", encoding="utf-8")

    def record(path: Path) -> dict:
        return {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }

    completion = {
        "schema_version": 1,
        "run_id": "rq1-clean",
        "completed_utc": "2026-01-01T00:00:00Z",
        "status": "complete",
        "exit_code": 0,
        "scheduled_cell_count": 5,
        "attempted_cell_count": 5,
        "successful_cell_count": 5,
        "failure_count": 0,
        "raw_record_count": len(raw_path.read_text(encoding="utf-8").splitlines()),
        "files": {
            "raw_jsonl": record(raw_path),
            "system_info": record(system_info_path),
            "measurement_manifest": record(manifest_path),
            "runner_log": record(runner_log_path),
        },
    }
    completion_path.write_text(json.dumps(completion), encoding="utf-8")


def test_rq1_two_level_summary_preserves_run_boundaries_and_loss() -> None:
    run_summary, config_summary = aggregate_clean_campaign(_rq1_records(), 1)

    assert len(run_summary) == 5
    first = run_summary[run_summary["repetition"] == 0].iloc[0]
    assert first["probes_sent"] == 4
    assert first["probes_received"] == 3
    assert first["probes_lost"] == 1
    assert first["probe_loss_pct"] == 25.0
    assert first["median_us"] == 2.0
    assert first["iqr_us"] == 1.0
    assert first["p99_us"] == pytest.approx(2.98)
    assert "p999_us" not in run_summary.columns

    assert len(config_summary) == 1
    summary = config_summary.iloc[0]
    assert summary["n_reps"] == 5
    assert summary["probe_loss_pct_median"] == 0.0
    assert summary["probe_loss_pct_min"] == 0.0
    assert summary["probe_loss_pct_max"] == 25.0
    assert summary["median_us_min"] == 2.0
    assert summary["median_us_max"] == 42.5
    assert summary["median_us_median"] == 22.5
    pooled_values = [record["value"] for record in _rq1_records()]
    assert summary["median_us_median"] != statistics.median(pooled_values)


def test_clean_campaign_rejects_mixed_runs_missing_repetition_and_failure() -> None:
    mixed = _rq1_records()
    mixed[-1]["run_id"] = "another-run"
    with pytest.raises(ValueError, match="mixed campaign run_ids"):
        aggregate_clean_rq1_runs(mixed)

    missing = [
        record for record in _rq1_records() if record["config"]["repetition"] != 4
    ]
    with pytest.raises(ValueError, match="repetition mismatch"):
        aggregate_clean_campaign(missing, 1)

    failed = _rq1_records()
    failed.append(
        {
            "run_id": "rq1-clean",
            "rq": 1,
            "config": {"repetition": 2},
            "metric": "config_failure",
            "value": "RuntimeError: test",
            "extras": {},
        }
    )
    with pytest.raises(ValueError, match="config_failure"):
        aggregate_clean_rq1_runs(failed)


def test_rq1_rejects_duplicate_or_out_of_range_probe_sequences() -> None:
    duplicate = _rq1_records()
    duplicate[1]["extras"]["sequence"] = duplicate[0]["extras"]["sequence"]
    with pytest.raises(ValueError, match="duplicate probe sequences"):
        aggregate_clean_rq1_runs(duplicate)

    outside = _rq1_records()
    outside[0]["extras"]["sequence"] = 999
    with pytest.raises(ValueError, match="sequences outside"):
        aggregate_clean_rq1_runs(outside)


def test_rq2_reports_wall_time_throughput_and_rejects_operation_failures() -> None:
    run_summary, config_summary = aggregate_clean_campaign(_rq2_records(), 2)
    assert len(run_summary) == 5
    assert list(run_summary["wall_clock_s"]) == pytest.approx(
        [0.1, 0.11, 0.12, 0.13, 0.14]
    )
    summary = config_summary.iloc[0]
    assert summary["wall_clock_s_median"] == pytest.approx(0.12)
    assert summary["entries_per_second_min"] == pytest.approx(200 / 0.14)
    assert summary["entries_per_second_max"] == 2000.0

    failed = _rq2_records()
    failed[0]["extras"]["failure_count"] = 1
    with pytest.raises(ValueError, match="failure_count=1"):
        aggregate_clean_rq2_runs(failed)

    duplicate = _rq2_records()
    duplicate.append(copy.deepcopy(duplicate[0]))
    with pytest.raises(ValueError, match="duplicate RQ2"):
        aggregate_clean_rq2_runs(duplicate)

    partial = _rq2_records()
    partial[0]["extras"]["success_count"] = 199
    with pytest.raises(ValueError, match="expected 200"):
        aggregate_clean_rq2_runs(partial)


def test_rq3_reports_signed_absolute_and_per_hop_run_statistics() -> None:
    run_summary, config_summary = aggregate_clean_campaign(_rq3_records(), 3)
    assert len(run_summary) == 5
    first = run_summary[run_summary["repetition"] == 0].iloc[0]
    assert first["probes_sent"] == 4
    assert first["probes_received"] == 4
    assert first["probes_lost"] == 0
    assert first["probe_loss_pct"] == 0.0
    assert first["mean_us"] == 0.0
    assert first["median_us"] == 0.0
    assert first["iqr_us"] == 2.5
    assert first["abs_mean_us"] == 1.5
    assert first["abs_median_us"] == 1.5
    assert first["per_hop_n"] == 8
    assert first["per_hop_abs_mean_us"] == pytest.approx(1.5)
    assert len(config_summary) == 1
    assert config_summary.iloc[0]["abs_p99_us_max"] >= 2.0

    missing_hops = _rq3_records()
    missing_hops[0]["extras"]["drift_us"] = []
    with pytest.raises(ValueError, match="per-hop drift_us"):
        aggregate_clean_rq3_runs(missing_hops)


def test_rq4_summarizes_each_metric_inside_run_before_across_run() -> None:
    run_summary, config_summary = aggregate_clean_campaign(_rq4_records(), 4)
    assert len(run_summary) == 20
    assert len(config_summary) == len(METRIC_RQ4_SET)
    first = run_summary[
        (run_summary["metric"] == "cpu_percent_per_bmv2")
        & (run_summary["repetition"] == 0)
    ].iloc[0]
    assert first["n_samples"] == 3
    assert first["mean"] == 2.0
    assert first["median"] == 2.0
    assert first["iqr"] == 1.0
    assert first["p95"] == pytest.approx(2.9)
    assert first["max"] == 3.0

    metric_summary = config_summary[
        config_summary["metric"] == "cpu_percent_per_bmv2"
    ].iloc[0]
    assert metric_summary["n_reps"] == 5
    assert metric_summary["mean_median"] == 4.0
    assert metric_summary["max_min"] == 3.0
    assert metric_summary["max_max"] == 7.0


def test_rq4_rejects_missing_metric_and_misaligned_sample_indexes() -> None:
    missing_metric = [
        record
        for record in _rq4_records()
        if not (
            record["config"]["repetition"] == 2
            and record["metric"] == "rss_per_bmv2_bytes"
        )
    ]
    with pytest.raises(ValueError, match="metric mismatch"):
        aggregate_clean_rq4_runs(missing_metric)

    misaligned = _rq4_records()
    target = next(
        record
        for record in misaligned
        if record["config"]["repetition"] == 0
        and record["metric"] == "rss_per_bmv2_bytes"
        and record["config"]["sample_index"] == 2
    )
    target["config"]["sample_index"] = 3
    with pytest.raises(ValueError, match="sample indexes"):
        aggregate_clean_rq4_runs(misaligned)

    timestamp_mismatch = _rq4_records()
    timestamp_mismatch[0]["extras"]["timestamp_us"] += 1
    with pytest.raises(ValueError, match="different timestamps"):
        aggregate_clean_rq4_runs(timestamp_mismatch)


def test_manifest_validator_matches_run_id_randomized_schedule_and_raw_cells() -> None:
    run_summary = aggregate_clean_rq1_runs(_rq1_records())
    validate_clean_manifest(_rq1_manifest(), 1, run_summary)

    wrong_run = _rq1_manifest()
    wrong_run["run_id"] = "other"
    with pytest.raises(ValueError, match="run_id mismatch"):
        validate_clean_manifest(wrong_run, 1, run_summary)

    not_randomized = _rq1_manifest()
    not_randomized["campaign"]["campaign"]["shuffle_all_cells"] = False
    with pytest.raises(ValueError, match="shuffle_all_cells"):
        validate_clean_manifest(not_randomized, 1, run_summary)

    wrong_schedule = _rq1_manifest()
    wrong_schedule["scheduled_cells"][0]["schedule_index"] = 4
    with pytest.raises(ValueError, match="schedule indexes"):
        validate_clean_manifest(wrong_schedule, 1, run_summary)


def test_manifest_source_hash_validator_rejects_changed_source() -> None:
    manifest = _rq1_manifest()
    validate_manifest_source_hashes(manifest)

    manifest["sha256"]["analysis/aggregate_clean.py"] = "0" * 64
    with pytest.raises(ValueError, match="source hash mismatch"):
        validate_manifest_source_hashes(manifest)


def test_cli_requires_manifest_and_writes_separate_clean_csvs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_path = tmp_path / "rq1-clean.jsonl"
    raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in _rq1_records()),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_rq1_manifest()), encoding="utf-8")
    _write_test_completion(
        raw_path=raw_path,
        manifest_path=manifest_path,
        completion_path=tmp_path / "completion.json",
    )
    summary_dir = tmp_path / "summaries"

    result = main(
        [
            "--raw-file",
            str(raw_path),
            "--manifest",
            str(manifest_path),
            "--rq",
            "1",
            "--summary",
            str(summary_dir),
            "--label",
            "c4-test",
        ]
    )
    assert result == 0
    assert (summary_dir / "rq1_run_summary_c4-test.csv").is_file()
    assert (summary_dir / "rq1_summary_c4-test.csv").is_file()
    assert (summary_dir / "rq1_provenance_c4-test.json").is_file()
    assert "VALID: RQ1 clean campaign" in capsys.readouterr().out

    with pytest.raises(ValueError, match="manifest not found"):
        main(
            [
                "--raw-file",
                str(raw_path),
                "--rq",
                "1",
                "--summary",
                str(summary_dir),
            ]
        )
