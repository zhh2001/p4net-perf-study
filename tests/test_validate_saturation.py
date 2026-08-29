"""Artifact-level validation tests for the calibration sweep."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from analysis.validate_saturation import _validate_iperf_artifacts
from workloads.saturation_sweep import parse_iperf3_json


def _interval(
    start: float,
    *,
    sender: bool,
    bytes_count: int,
    packets: int,
    lost_packets: int = 0,
    omitted: bool = False,
) -> dict:
    return {
        "sum": {
            "start": start,
            "end": start + 1.0,
            "seconds": 1.0,
            "bytes": bytes_count,
            "packets": packets,
            "lost_packets": lost_packets,
            "omitted": omitted,
            "sender": sender,
        }
    }


def _documents() -> tuple[dict, dict]:
    test_start = {
        "protocol": "UDP",
        "omit": 30,
        "duration": 4,
        "blksize": 1448,
        "target_bitrate": 10_000_000,
    }
    server = {
        "start": {"version": "iperf 3.16", "test_start": copy.deepcopy(test_start)},
        "intervals": [
            _interval(
                0.0,
                sender=False,
                bytes_count=1_448_000,
                packets=1000,
                omitted=True,
            ),
            _interval(0.0, sender=False, bytes_count=1_448_000, packets=1000),
            _interval(
                1.0,
                sender=False,
                bytes_count=1_433_520,
                packets=1000,
                lost_packets=10,
            ),
            _interval(
                2.0,
                sender=False,
                bytes_count=1_440_760,
                packets=1000,
                lost_packets=5,
            ),
        ],
    }
    client = {
        "start": {"version": "iperf 3.16", "test_start": copy.deepcopy(test_start)},
        "intervals": [
            _interval(
                0.0,
                sender=True,
                bytes_count=1_448_000,
                packets=1000,
                omitted=True,
            ),
            _interval(0.0, sender=True, bytes_count=1_448_000, packets=1000),
            _interval(1.0, sender=True, bytes_count=1_448_000, packets=1000),
            _interval(2.0, sender=True, bytes_count=1_448_000, packets=1000),
        ],
        "server_output_json": copy.deepcopy(server),
    }
    return client, server


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(tmp_path: Path) -> dict:
    client, server = _documents()
    client_path = tmp_path / "client.json"
    server_path = tmp_path / "server.json"
    client_stderr = tmp_path / "client.stderr"
    server_stderr = tmp_path / "server.stderr"
    client_path.write_text(json.dumps(client), encoding="utf-8")
    server_path.write_text(json.dumps(server), encoding="utf-8")
    client_stderr.write_bytes(b"")
    server_stderr.write_bytes(b"")
    parsed = parse_iperf3_json(
        client,
        server,
        nominal_offered_mbps=10,
        measurement_seconds=2,
        post_omit_guard_intervals=1,
    )
    extras = {
        **parsed,
        "iperf_post_omit_guard_intervals": 1,
        "iperf_client_json_path": str(client_path),
        "iperf_server_json_path": str(server_path),
        "iperf_client_stderr_path": str(client_stderr),
        "iperf_server_stderr_path": str(server_stderr),
        "iperf_client_json_sha256": _sha256(client_path),
        "iperf_server_json_sha256": _sha256(server_path),
    }
    return {
        "config": {
            "schema_version": 3,
            "rate_mbps": 10,
            "repetition": 0,
            "warmup_seconds": 30,
            "measurement_seconds": 2,
            "iperf_tail_seconds": 1,
            "iperf_post_omit_guard_intervals": 1,
            "iperf_udp_length_bytes": 1448,
        },
        "extras": extras,
    }


def test_validate_iperf_artifacts_accepts_reparsed_summary(tmp_path: Path) -> None:
    _validate_iperf_artifacts(_record(tmp_path))


def test_validate_iperf_artifacts_rejects_stale_summary(tmp_path: Path) -> None:
    record = _record(tmp_path)
    record["extras"]["actual_offered_mbps"] *= 0.98
    with pytest.raises(ValueError, match="stale iperf3 field actual_offered_mbps"):
        _validate_iperf_artifacts(record)


def test_validate_iperf_artifacts_rejects_hash_mismatch(tmp_path: Path) -> None:
    record = _record(tmp_path)
    record["extras"]["iperf_client_json_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="client artifact hash mismatch"):
        _validate_iperf_artifacts(record)
