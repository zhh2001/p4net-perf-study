"""Tests for two-stage calibration aggregation and sensitivity rules."""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.aggregate import (
    aggregate_saturation,
    aggregate_saturation_runs,
    aggregate_saturation_sensitivity,
)


def _config(rate: int, repetition: int) -> dict:
    return {
        "rate_mbps": rate,
        "background_load_mbps": rate,
        "repetition": repetition,
        "schedule_index": repetition,
        "source_workload_type": "saturation_sweep",
    }


def _summary_record(rate: int, repetition: int, *, received: int = 2) -> dict:
    sent = 2
    loss = 100.0 * (1.0 - received / sent)
    return {
        "run_id": "run-a",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "rq": 0,
        "config": _config(rate, repetition),
        "metric": "saturation_probe_loss_pct",
        "value": loss,
        "extras": {
            "nominal_offered_mbps": float(rate),
            "actual_offered_mbps": float(rate) - 0.1,
            "achieved_mbps": float(rate) - 0.2,
            "achieved_to_actual_offered_pct": 99.8,
            "achieved_to_nominal_pct": 99.6,
            "sender_seconds": 60.0,
            "receiver_seconds": 60.0,
            "sender_datagrams": 1000,
            "receiver_total_datagrams": 1000,
            "receiver_lost_datagrams": 1,
            "receiver_datagrams": 999,
            "sender_pps": 16.667,
            "receiver_pps": 16.65,
            "iperf_receiver_loss_pct": 0.1,
            "probes_sent": sent,
            "probes_received": received,
            "probe_loss_pct": loss,
            "duplicate_probes": 0,
            "out_of_range_probes": 0,
            "measurement_start_monotonic_us": 1_000_000,
            "measurement_end_monotonic_us": 61_000_000,
            "probe_campaign_seconds": 63.0,
            "iperf_udp_length_bytes": 1460,
            "iperf3_version": "iperf 3.16",
            "iperf_client_json_path": "client.json",
            "iperf_server_json_path": "server.json",
        },
    }


def _latency_record(rate: int, repetition: int, sequence: int, value: float) -> dict:
    return {
        "run_id": "run-a",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "rq": 0,
        "config": _config(rate, repetition),
        "metric": "saturation_ingress_to_egress_start_us",
        "value": value,
        "extras": {"sequence": sequence},
    }


def _cpu_record(rate: int, repetition: int, metric: str, value: float) -> dict:
    return {
        "run_id": "run-a",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "rq": 4,
        "config": _config(rate, repetition),
        "metric": metric,
        "value": value,
        "extras": {},
    }


def test_saturation_aggregation_computes_runs_before_cross_run_statistics() -> None:
    records: list[dict] = []
    for repetition, bmv2_values in ((0, [10.0]), (1, [30.0, 30.0, 30.0])):
        records.append(_summary_record(50, repetition))
        records.extend(
            [
                _latency_record(50, repetition, 0, 100.0 + repetition),
                _latency_record(50, repetition, 1, 200.0 + repetition),
            ]
        )
        records.extend(
            _cpu_record(50, repetition, "cpu_percent_per_bmv2", value)
            for value in bmv2_values
        )
        records.extend(
            _cpu_record(50, repetition, "cpu_percent_total", value)
            for value in (20.0 + repetition, 22.0 + repetition)
        )

    run_summary = aggregate_saturation_runs(records)
    assert len(run_summary) == 2
    assert list(run_summary["bmv2_cpu_mean_pct"]) == [10.0, 30.0]
    assert list(run_summary["latency_median_us"]) == [150.0, 151.0]

    summary = aggregate_saturation(run_summary)
    row = summary.iloc[0]
    # Median of the two run means is 20, whereas pooling all four CPU ticks
    # would produce 25. This detects accidental sample-level pooling.
    assert row["bmv2_cpu_mean_pct_median"] == 20.0
    assert row["latency_median_us_median"] == 150.5
    assert row["n_reps"] == 2


def test_sensitivity_uses_joint_three_of_five_rule_and_inclusive_boundaries() -> None:
    run_summary = pd.DataFrame(
        {
            "rate_mbps": [50] * 5 + [75] * 5,
            "repetition": list(range(5)) * 2,
            # 50 Mbps has three joint passes at the exact inclusive boundary.
            "probe_loss_pct": [5.0, 4.0, 1.0, 6.0, 0.0, 4.0, 4.0, 4.0, 6.0, 6.0],
            "achieved_to_nominal_pct": [95.0, 96.0, 95.0, 99.0, 94.0, 94.0, 94.0, 94.0, 99.0, 99.0],
        }
    )
    sensitivity = aggregate_saturation_sensitivity(
        run_summary,
        loss_cutoffs_pct=(5.0,),
        throughput_floor_pct=95.0,
        minimum_passing_repetitions=3,
    )

    row_50 = sensitivity[sensitivity["rate_mbps"] == 50].iloc[0]
    row_75 = sensitivity[sensitivity["rate_mbps"] == 75].iloc[0]
    assert row_50["passing_reps"] == 3
    assert bool(row_50["rate_qualifies"])
    assert row_75["passing_reps"] == 0
    assert not bool(row_75["rate_qualifies"])
    assert row_50["rmax_mbps"] == 50
    assert row_75["rmax_mbps"] == 50


def test_saturation_run_aggregation_rejects_missing_raw_latency() -> None:
    records = [
        _summary_record(25, 0),
        _cpu_record(25, 0, "cpu_percent_per_bmv2", 10.0),
        _cpu_record(25, 0, "cpu_percent_total", 20.0),
    ]
    with pytest.raises(ValueError, match="latency records"):
        aggregate_saturation_runs(records)
