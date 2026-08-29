"""Unit tests for calibration cell scheduling and raw-record tagging."""

from __future__ import annotations

import io
import json

from runner.runner import (
    _build_execution_cells,
    _resource_config_payload,
    _write_saturation_point,
)


def _saturation_config() -> dict:
    return {
        "rq": 0,
        "workload_type": "saturation_sweep",
        "p4_program": "l3_lpm",
        "topology": "single_switch",
        "n_switches": 1,
        "rates_mbps": [10, 25, 50, 75, 100],
        "n_probes": 1000,
        "probe_interval_ms": 60.0,
        "probe_packet_size_bytes": 256,
        "measurement_seconds": 60,
        "iperf_tail_seconds": 15,
        "iperf_post_omit_guard_intervals": 1,
        "iperf_udp_length_bytes": 1448,
        "repetitions": 5,
    }


def test_build_execution_cells_expands_and_seed_shuffles_full_matrix() -> None:
    first = _build_execution_cells(
        [_saturation_config()], seed=42, shuffle_all_cells=True
    )
    second = _build_execution_cells(
        [_saturation_config()], seed=42, shuffle_all_cells=True
    )
    different = _build_execution_cells(
        [_saturation_config()], seed=43, shuffle_all_cells=True
    )

    identity = [(int(cfg["rate_mbps"]), rep) for cfg, rep in first]
    assert len(identity) == 25
    assert len(set(identity)) == 25
    for rate in (10, 25, 50, 75, 100):
        assert sum(cell_rate == rate for cell_rate, _ in identity) == 5
    for repetition in range(5):
        assert sum(rep == repetition for _, rep in identity) == 5
    assert identity == [(int(cfg["rate_mbps"]), rep) for cfg, rep in second]
    assert identity != [(int(cfg["rate_mbps"]), rep) for cfg, rep in different]


def test_saturation_resource_payload_uses_exact_rate_and_repetition() -> None:
    cfg = _saturation_config()
    cfg.pop("rates_mbps")
    cfg.update(rate_mbps=75, background_load_mbps=75, schedule_index=9)
    payload = _resource_config_payload(cfg, repetition=3)
    assert payload["rate_mbps"] == 75
    assert payload["background_load_mbps"] == 75
    assert payload["source_workload_type"] == "saturation_sweep"
    assert payload["repetition"] == 3
    assert payload["schedule_index"] == 9


def test_write_saturation_point_preserves_summary_and_raw_probe_samples() -> None:
    cfg = _saturation_config()
    cfg.pop("rates_mbps")
    cfg.update(rate_mbps=50, background_load_mbps=50, schedule_index=7)
    result = {
        "rate_mbps": 50,
        "probe_loss_pct": 50.0,
        "probes_sent": 2,
        "probes_received": 1,
        "probe_samples": [
            {
                "sequence": 4,
                "ingress_ts_us": 100,
                "egress_ts_us": 125,
                "switch_transit_us": 25.0,
            }
        ],
        "nominal_offered_mbps": 50.0,
    }
    fh = io.StringIO()
    _write_saturation_point(fh, result, "run-test", cfg, repetition=2)
    records = [json.loads(line) for line in fh.getvalue().splitlines()]

    assert len(records) == 2
    summary, latency = records
    assert summary["metric"] == "saturation_probe_loss_pct"
    assert summary["config"]["rate_mbps"] == 50
    assert summary["config"]["repetition"] == 2
    assert summary["config"]["schedule_index"] == 7
    assert "probe_samples" not in summary["extras"]
    assert latency["metric"] == "saturation_ingress_to_egress_start_us"
    assert latency["value"] == 25.0
    assert latency["extras"]["sequence"] == 4
