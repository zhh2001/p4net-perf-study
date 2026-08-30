"""Concern 4 campaign scheduling, provenance, and raw-schema tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from runner.runner import (
    _build_execution_cells,
    _control_plane_config_payload,
    _int_config_payload,
    _latency_config_payload,
    _resource_config_payload,
    _validate_control_plane_result,
    _validate_resource_samples,
    _write_calibration_manifest,
    _write_campaign_completion,
    _write_campaign_manifest,
    main,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_config(name: str) -> dict:
    path = REPO_ROOT / "runner" / "configs" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_c4_configs_preserve_full_matrices_with_five_restarts() -> None:
    expected_counts = {1: 37, 2: 42, 3: 6, 4: 8}
    expected_cells = {1: 185, 2: 210, 3: 30, 4: 40}

    for rq, config_count in expected_counts.items():
        original = _load_config(f"rq{rq}_full.yaml")
        c4 = _load_config(f"rq{rq}_c4.yaml")
        assert c4["campaign"]["name"] == f"rq{rq}_c4_clean_restarts"
        assert c4["campaign"]["shuffle_all_cells"] is True
        for campaign_field in ("seed", "warmup_seconds", "cooldown_seconds"):
            assert c4["campaign"][campaign_field] == original["campaign"][campaign_field]
        assert len(c4["configs"]) == config_count
        assert sum(int(cfg["repetitions"]) for cfg in c4["configs"]) == expected_cells[rq]

        for old_cfg, c4_cfg in zip(original["configs"], c4["configs"], strict=True):
            assert c4_cfg == {**old_cfg, "repetitions": 5}


def test_c4_schedules_each_configuration_at_repetitions_zero_through_four() -> None:
    for rq in range(1, 5):
        c4 = _load_config(f"rq{rq}_c4.yaml")
        cells = _build_execution_cells(
            c4["configs"],
            seed=int(c4["campaign"]["seed"]),
            shuffle_all_cells=bool(c4["campaign"]["shuffle_all_cells"]),
        )
        assert len(cells) == len(c4["configs"]) * 5
        for cfg in c4["configs"]:
            config_cells = [
                repetition for cell_cfg, repetition in cells if cell_cfg == cfg
            ]
            assert sorted(config_cells) == [0, 1, 2, 3, 4]


def test_c4_smoke_covers_every_non_calibration_workload_path() -> None:
    smoke = _load_config("c4_smoke.yaml")
    assert smoke["campaign"]["shuffle_all_cells"] is True
    assert {cfg["workload_type"] for cfg in smoke["configs"]} == {
        "latency_l2",
        "latency_l3",
        "control_plane",
        "int_multihop",
        "resource_only",
    }
    assert any(
        cfg["p4_program"] == "l3_lpm_int" and cfg["background_load_mbps"] > 0
        for cfg in smoke["configs"]
    )
    assert any(
        cfg["workload_type"] == "control_plane"
        and cfg["operation"] == "read"
        and cfg["mode"] == "async"
        for cfg in smoke["configs"]
    )
    assert any(
        cfg["workload_type"] == "resource_only"
        and cfg["topology"] == "linear_n"
        for cfg in smoke["configs"]
    )


def test_general_manifest_records_resolved_schedule_config_and_hashes(tmp_path: Path) -> None:
    config_path = REPO_ROOT / "runner" / "configs" / "rq3_c4.yaml"
    campaign = _load_config("rq3_c4.yaml")
    cells = _build_execution_cells(
        campaign["configs"], seed=42, shuffle_all_cells=True
    )
    manifest_path = tmp_path / "manifest.json"

    _write_campaign_manifest(
        path=manifest_path,
        campaign=campaign,
        config_path=config_path,
        run_id="run-c4-test",
        cells=cells,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["run_id"] == "run-c4-test"
    assert manifest["created_utc"].endswith("Z")
    assert manifest["config_path"] == "runner/configs/rq3_c4.yaml"
    assert manifest["campaign"] == campaign
    assert len(manifest["scheduled_cells"]) == 30
    assert [cell["schedule_index"] for cell in manifest["scheduled_cells"]] == list(
        range(30)
    )
    assert all(cell["config"]["repetitions"] == 5 for cell in manifest["scheduled_cells"])
    assert {cell["repetition"] for cell in manifest["scheduled_cells"]} == set(range(5))

    required_hashes = {
        "analysis/aggregate_clean.py",
        "p4/include/instrument.p4h",
        "runner/configs/rq3_c4.yaml",
        "runner/runner.py",
        "workloads/int_collector.py",
        "workloads/resource_monitor.py",
        "topologies/linear_n.py",
        "p4/l3_lpm_int_chain.p4",
        "pyproject.toml",
    }
    assert required_hashes <= manifest["sha256"].keys()
    for relative_path, expected_hash in manifest["sha256"].items():
        source = REPO_ROOT / relative_path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash


def test_calibration_manifest_retains_concern3_shape(tmp_path: Path) -> None:
    config_path = REPO_ROOT / "runner" / "configs" / "saturation_sweep.yaml"
    campaign = _load_config("saturation_sweep.yaml")
    cells = _build_execution_cells(
        campaign["configs"], seed=42, shuffle_all_cells=True
    )
    manifest_path = tmp_path / "manifest.json"

    _write_calibration_manifest(
        path=manifest_path,
        campaign=campaign,
        config_path=config_path,
        run_id="run-c3-test",
        cells=cells,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(manifest["scheduled_cells"]) == 25
    assert set(manifest["scheduled_cells"][0]) == {
        "schedule_index",
        "rate_mbps",
        "repetition",
    }
    assert "workloads/saturation_sweep.py" in manifest["sha256"]


def test_completion_binds_final_files_and_counts(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text('{"one": 1}\n{"two": 2}\n', encoding="utf-8")
    system_info_path = tmp_path / "system.json"
    system_info_path.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    runner_log_path = tmp_path / "runner.log"
    runner_log_path.write_text("complete\n", encoding="utf-8")
    completion_path = tmp_path / "artifacts" / "completion.json"
    completion_path.parent.mkdir()

    exit_code = _write_campaign_completion(
        path=completion_path,
        run_id="completion-test",
        raw_path=raw_path,
        system_info_path=system_info_path,
        manifest_path=manifest_path,
        runner_log_path=runner_log_path,
        scheduled_cell_count=2,
        attempted_cell_count=2,
        failure_count=0,
    )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert completion["status"] == "complete"
    assert completion["raw_record_count"] == 2
    assert completion["successful_cell_count"] == 2
    for record in completion["files"].values():
        assert len(record["sha256"]) == 64
        assert record["size_bytes"] > 0


def test_new_raw_payloads_retain_run_design_fields() -> None:
    latency_cfg = {
        "workload_type": "latency_l3",
        "p4_program": "l3_lpm",
        "packet_size_bytes": 256,
        "background_load_mbps": 25,
        "n_probes": 1000,
        "probe_interval_ms": 60.0,
        "schedule_index": 8,
    }
    latency = _latency_config_payload(latency_cfg, repetition=3)
    assert latency["n_probes"] == 1000
    assert latency["probe_interval_ms"] == 60.0
    assert latency["packet_size_bytes"] == 256
    assert latency["repetition"] == 3
    assert latency["schedule_index"] == 8
    assert latency["post_instrument_bytes"] == 0

    latency_int = _latency_config_payload(
        {**latency_cfg, "p4_program": "l3_lpm_int"}, repetition=3
    )
    assert latency_int["post_instrument_bytes"] == 13

    int_cfg = {
        "workload_type": "int_multihop",
        "p4_program": "l3_lpm_int_chain",
        "topology": "linear_n",
        "n_switches": 3,
        "background_load_mbps": 45,
        "packet_size_bytes": 256,
        "n_probes": 1000,
        "probe_interval_ms": 60.0,
        "schedule_index": 11,
    }
    int_payload = _int_config_payload(int_cfg, repetition=4)
    assert int_payload["n_probes"] == 1000
    assert int_payload["probe_interval_ms"] == 60.0
    assert int_payload["packet_size_bytes"] == 256
    assert int_payload["repetition"] == 4
    assert int_payload["schedule_index"] == 11

    control_cfg = {
        "workload_type": "control_plane",
        "p4_program": "l3_lpm",
        "topology": "linear_n",
        "n_switches": 8,
        "n_entries_per_switch": 1000,
        "operation": "read",
        "mode": "async",
        "schedule_index": 12,
    }
    control = _control_plane_config_payload(control_cfg, repetition=2)
    assert control["repetition"] == 2
    assert control["schedule_index"] == 12

    resource_cfg = {
        "rq": 4,
        "workload_type": "resource_only",
        "p4_program": "l3_lpm",
        "topology": "linear_n",
        "n_switches": 4,
        "background_load_mbps": 45,
        "duration_s": 60,
        "schedule_index": 9,
    }
    resource = _resource_config_payload(resource_cfg, repetition=1)
    assert resource["source_rq"] == 4
    assert resource["duration_s"] == 60.0
    assert resource["resource_sample_interval_s"] == 0.1
    assert resource["repetition"] == 1
    assert resource["schedule_index"] == 9

    control_resource = _resource_config_payload(control_cfg, repetition=2)
    assert control_resource["n_entries_per_switch"] == 1000
    assert control_resource["operation"] == "read"
    assert control_resource["mode"] == "async"


def test_control_plane_result_guard_rejects_partial_or_inconsistent_batches() -> None:
    valid = {
        "success_count": 20,
        "failure_count": 0,
        "total_wall_clock_s": 0.1,
        "entries_per_second": 200.0,
    }
    _validate_control_plane_result(
        valid, expected_operations=20, phase="test"
    )

    with pytest.raises(RuntimeError, match="batch incomplete"):
        _validate_control_plane_result(
            {**valid, "failure_count": 1},
            expected_operations=20,
            phase="read prefill",
        )
    with pytest.raises(RuntimeError, match="batch incomplete"):
        _validate_control_plane_result(
            {**valid, "success_count": 19},
            expected_operations=20,
            phase="read",
        )
    with pytest.raises(RuntimeError, match="does not match"):
        _validate_control_plane_result(
            {**valid, "entries_per_second": 199.0},
            expected_operations=20,
            phase="insert",
        )


def test_resource_sample_guard_rejects_missing_process_or_samples() -> None:
    sample = {
        "timestamp_us": 1,
        "cpu_percent_per_bmv2": {123: 1.0},
        "rss_per_bmv2_bytes": {123: 100},
        "net_io_per_iface": {"s1-eth1": {}},
    }
    _validate_resource_samples(
        [sample], bmv2_pids=[123], switch_ifaces=["s1-eth1"]
    )
    with pytest.raises(RuntimeError, match="expected at least"):
        _validate_resource_samples(
            [], bmv2_pids=[123], switch_ifaces=["s1-eth1"]
        )
    with pytest.raises(RuntimeError, match="process mismatch"):
        _validate_resource_samples(
            [{**sample, "cpu_percent_per_bmv2": {}}],
            bmv2_pids=[123],
            switch_ifaces=["s1-eth1"],
        )


@pytest.mark.integration
@pytest.mark.requires_p4c
@pytest.mark.requires_bmv2
def test_rq2_async_read_restarts_share_loop_without_closed_loop_callbacks(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exercise repeated N=8, K=10 async-read cells through the real runner."""
    config_path = tmp_path / "rq2_async_read_loop_smoke.yaml"
    config_path.write_text(
        """campaign:
  name: rq2_async_read_loop_smoke
  seed: 42
  warmup_seconds: 0
  cooldown_seconds: 0
  shuffle_all_cells: true
configs:
  - rq: 2
    workload_type: control_plane
    p4_program: l3_lpm
    topology: linear_n
    n_switches: 8
    n_entries_per_switch: 10
    operation: read
    mode: async
    repetitions: 3
""",
        encoding="utf-8",
    )
    output_path = tmp_path / "raw"

    with caplog.at_level("INFO"):
        assert main(["--config", str(config_path), "--output", str(output_path)]) == 0

    raw_path = next(output_path.glob("rq2_async_read_loop_smoke_*.jsonl"))
    records = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
    ]
    primary = [
        record
        for record in records
        if record["metric"] == "control_plane_wall_clock_s"
    ]
    assert len(primary) == 3
    assert {record["config"]["repetition"] for record in primary} == {0, 1, 2}
    assert all(record["extras"]["success_count"] == 80 for record in primary)
    assert not any(record["metric"] == "config_failure" for record in records)

    artifacts_path = next(output_path.glob("rq2_async_read_loop_smoke_*_artifacts"))
    runner_log = (artifacts_path / "runner.log").read_text(encoding="utf-8")
    assert "Event loop is closed" not in runner_log
    assert "Exception in callback" not in runner_log
    assert runner_log.count("Network.start: 2 hosts, 8 switches") == 3
    # p4net cleanup is idempotent and may emit an additional stop log from
    # object finalization; every experimental unit must still reach teardown.
    assert runner_log.count("Network.stop: tearing down") >= 3

    completion = json.loads(
        (artifacts_path / "completion.json").read_text(encoding="utf-8")
    )
    assert completion["status"] == "complete"
    assert completion["failure_count"] == 0


def test_non_calibration_failure_returns_nonzero_and_preserves_partial_raw(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "failure_campaign.yaml"
    config_path.write_text(
        """campaign:
  name: c4_failure_test
  seed: 42
  warmup_seconds: 0
  cooldown_seconds: 0
  shuffle_all_cells: true
configs:
  - rq: 1
    workload_type: latency_l3
    p4_program: l3_lpm
    packet_size_bytes: 64
    background_load_mbps: 0
    n_probes: 2
    probe_interval_ms: 1.0
    repetitions: 1
""",
        encoding="utf-8",
    )
    output_path = tmp_path / "raw"

    with (
        patch("runner.runner.capture_system_info", return_value={}),
        patch("runner.runner._verify_schema"),
        patch("runner.runner._run_latency", side_effect=RuntimeError("injected")),
    ):
        return_code = main(
            ["--config", str(config_path), "--output", str(output_path)]
        )

    assert return_code == 1
    raw_files = list(output_path.glob("c4_failure_test_*.jsonl"))
    assert len(raw_files) == 1
    records = [
        json.loads(line)
        for line in raw_files[0].read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["metric"] == "config_failure"
    assert records[0]["config"]["schedule_index"] == 0
    artifact_dirs = list(output_path.glob("c4_failure_test_*_artifacts"))
    assert len(artifact_dirs) == 1
    assert (artifact_dirs[0] / "manifest.json").is_file()
    completion = json.loads(
        (artifact_dirs[0] / "completion.json").read_text(encoding="utf-8")
    )
    assert completion["status"] == "failed"
    assert completion["exit_code"] == 1
    assert completion["failure_count"] == 1
    assert "runner_log" in completion["files"]
