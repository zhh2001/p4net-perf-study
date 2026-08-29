"""Completeness and consistency gate for the Concern 3 calibration sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from analysis.aggregate import (
    METRIC_SATURATION_SUMMARY,
    aggregate_saturation_runs,
    aggregate_saturation_sensitivity,
)
from workloads.saturation_sweep import parse_iperf3_json

EXPECTED_RATES = (10, 25, 50, 75, 100)
EXPECTED_REPETITIONS = tuple(range(5))
MIN_CPU_SAMPLES_PER_CELL = 500


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_iperf_artifacts(record: dict[str, Any]) -> None:
    """Reparse one cell's artifacts and reject stale derived summaries."""
    config = record.get("config", {})
    extras = record.get("extras", {})
    rate = int(config["rate_mbps"])
    repetition = int(config["repetition"])
    if int(config.get("schema_version", 0)) < 3:
        raise ValueError(
            f"rate={rate} rep={repetition}: pre-repair calibration schema"
        )

    paths: dict[str, Path] = {}
    for side in ("client", "server"):
        json_path = Path(extras[f"iperf_{side}_json_path"])
        stderr_path = Path(extras[f"iperf_{side}_stderr_path"])
        if not json_path.is_file():
            raise ValueError(
                f"rate={rate} rep={repetition}: missing iperf3 artifact {json_path}"
            )
        if not stderr_path.is_file():
            raise ValueError(
                f"rate={rate} rep={repetition}: missing iperf3 stderr {stderr_path}"
            )
        if stderr_path.stat().st_size != 0:
            raise ValueError(
                f"rate={rate} rep={repetition}: non-empty iperf3 {side} stderr"
            )
        expected_hash = str(extras[f"iperf_{side}_json_sha256"])
        if _sha256(json_path) != expected_hash:
            raise ValueError(
                f"rate={rate} rep={repetition}: iperf3 {side} artifact hash mismatch"
            )
        paths[side] = json_path

    try:
        client_document = json.loads(paths["client"].read_text(encoding="utf-8"))
        server_document = json.loads(paths["server"].read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"rate={rate} rep={repetition}: invalid iperf3 artifact JSON: {exc}"
        ) from exc

    expected_duration = (
        int(config["iperf_post_omit_guard_intervals"])
        + int(config["measurement_seconds"])
        + int(config["iperf_tail_seconds"])
    )
    for side, document in (
        ("client", client_document),
        ("server", server_document),
    ):
        test_start = document.get("start", {}).get("test_start", {})
        expected_test_start = {
            "protocol": "UDP",
            "omit": int(config["warmup_seconds"]),
            "duration": expected_duration,
            "blksize": int(config["iperf_udp_length_bytes"]),
            "target_bitrate": rate * 1_000_000,
        }
        for field, expected in expected_test_start.items():
            if test_start.get(field) != expected:
                raise ValueError(
                    f"rate={rate} rep={repetition}: iperf3 {side} test_start "
                    f"{field}={test_start.get(field)!r}, expected {expected!r}"
                )

    reparsed = parse_iperf3_json(
        client_document,
        server_document,
        nominal_offered_mbps=rate,
        measurement_seconds=int(config["measurement_seconds"]),
        post_omit_guard_intervals=int(config["iperf_post_omit_guard_intervals"]),
    )
    integer_fields = (
        "iperf_udp_length_bytes",
        "sender_datagrams",
        "receiver_total_datagrams",
        "receiver_lost_datagrams",
        "receiver_datagrams",
        "iperf_intervals_used",
        "iperf_measurement_first_bin",
        "iperf_measurement_last_bin",
    )
    float_fields = (
        "nominal_offered_mbps",
        "actual_offered_mbps",
        "achieved_mbps",
        "achieved_to_actual_offered_pct",
        "achieved_to_nominal_pct",
        "sender_seconds",
        "receiver_seconds",
        "sender_pps",
        "receiver_pps",
        "iperf_receiver_loss_pct",
    )
    for field in integer_fields:
        if int(extras[field]) != int(reparsed[field]):
            raise ValueError(
                f"rate={rate} rep={repetition}: stale iperf3 field {field}"
            )
    if int(extras["iperf_post_omit_guard_intervals"]) != int(
        config["iperf_post_omit_guard_intervals"]
    ):
        raise ValueError(
            f"rate={rate} rep={repetition}: iperf guard setting mismatch"
        )
    for field in float_fields:
        recorded = float(extras[field])
        expected = float(reparsed[field])
        if not math.isclose(recorded, expected, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(
                f"rate={rate} rep={repetition}: stale iperf3 field {field}: "
                f"recorded={recorded}, reparsed={expected}"
            )
    if str(extras["iperf3_version"]) != str(reparsed["iperf3_version"]):
        raise ValueError(
            f"rate={rate} rep={repetition}: stale iperf3 field iperf3_version"
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def validate_records(
    records: list[dict[str, Any]],
    *,
    expected_rmax_mbps: int | None = None,
) -> tuple[Any, Any]:
    """Validate one complete run and return run/sensitivity DataFrames."""
    failures = [
        record
        for record in records
        if record.get("metric") == "config_failure"
        and record.get("config", {}).get("workload_type") == "saturation_sweep"
    ]
    if failures:
        cells = [
            (
                record.get("config", {}).get("rate_mbps"),
                record.get("config", {}).get("repetition"),
                record.get("extras", {}).get("error"),
            )
            for record in failures
        ]
        raise ValueError(f"calibration contains failed cells: {cells}")

    summary_records = [
        record for record in records if record.get("metric") == METRIC_SATURATION_SUMMARY
    ]
    for record in summary_records:
        _validate_iperf_artifacts(record)
    run_ids = {str(record["run_id"]) for record in summary_records}
    if len(run_ids) != 1:
        raise ValueError(f"expected one calibration run_id, found {sorted(run_ids)}")

    run_summary = aggregate_saturation_runs(records)
    expected_cells = {
        (rate, repetition)
        for rate in EXPECTED_RATES
        for repetition in EXPECTED_REPETITIONS
    }
    actual_cells = {
        (int(row.rate_mbps), int(row.repetition))
        for row in run_summary.itertuples(index=False)
    }
    if actual_cells != expected_cells:
        missing = sorted(expected_cells - actual_cells)
        extra = sorted(actual_cells - expected_cells)
        raise ValueError(f"calibration cell mismatch: missing={missing}, extra={extra}")
    if len(run_summary) != len(expected_cells):
        raise ValueError("calibration contains duplicate run-level cells")

    for row in run_summary.itertuples(index=False):
        if row.latency_n_samples != row.probes_received:
            raise ValueError(
                f"rate={row.rate_mbps} rep={row.repetition}: latency/probe count mismatch"
            )
        if row.bmv2_cpu_n_samples < MIN_CPU_SAMPLES_PER_CELL:
            raise ValueError(
                f"rate={row.rate_mbps} rep={row.repetition}: only "
                f"{row.bmv2_cpu_n_samples} BMv2 CPU samples"
            )
        if row.system_cpu_n_samples < MIN_CPU_SAMPLES_PER_CELL:
            raise ValueError(
                f"rate={row.rate_mbps} rep={row.repetition}: only "
                f"{row.system_cpu_n_samples} system CPU samples"
            )
        for path_text in (row.iperf_client_json_path, row.iperf_server_json_path):
            if not Path(path_text).is_file():
                raise ValueError(f"missing iperf3 artifact: {path_text}")
        numeric = [
            row.actual_offered_mbps,
            row.achieved_mbps,
            row.sender_pps,
            row.receiver_pps,
            row.probe_loss_pct,
            row.latency_median_us,
            row.latency_p99_us,
            row.bmv2_cpu_mean_pct,
            row.bmv2_cpu_p95_pct,
            row.system_cpu_mean_pct,
            row.system_cpu_p95_pct,
        ]
        if not np.isfinite(np.asarray(numeric, dtype=float)).all():
            raise ValueError(
                f"rate={row.rate_mbps} rep={row.repetition}: non-finite summary value"
            )

    sensitivity = aggregate_saturation_sensitivity(run_summary)
    for cutoff, group in sensitivity.groupby("loss_cutoff_pct", sort=True):
        ordered = group.sort_values("rate_mbps")
        qualifies = [bool(value) for value in ordered["rate_qualifies"]]
        seen_failure = False
        for qualifies_at_rate in qualifies:
            if not qualifies_at_rate:
                seen_failure = True
            elif seen_failure:
                raise ValueError(
                    f"non-monotonic pass/fail pattern at loss cutoff {cutoff}%"
                )
        rmax_values = {value for value in ordered["rmax_mbps"] if value == value}
        if (
            cutoff == 5.0
            and expected_rmax_mbps is not None
            and rmax_values != {expected_rmax_mbps}
        ):
            raise ValueError(
                f"5% criterion produced Rmax={sorted(rmax_values)}, "
                f"expected {expected_rmax_mbps} Mbps"
            )
    return run_summary, sensitivity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, required=True)
    parser.add_argument("--expected-rmax-mbps", type=int)
    args = parser.parse_args(argv)

    try:
        records = _read_jsonl(args.raw_file)
        run_summary, sensitivity = validate_records(
            records,
            expected_rmax_mbps=args.expected_rmax_mbps,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(
        f"VALID: {len(run_summary)} cells; "
        "5% highest qualifying tested rate="
        f"{sensitivity[sensitivity['loss_cutoff_pct'] == 5.0]['rmax_mbps'].iloc[0]} Mbps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
