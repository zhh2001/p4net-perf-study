"""Unit tests for the calibration-specific iperf3 parser."""

from __future__ import annotations

import copy

import pytest

from workloads.saturation_sweep import parse_iperf3_json


def _interval(
    *,
    start: float,
    seconds: float,
    bytes_count: int,
    packets: int,
    lost_packets: int = 0,
    omitted: bool = False,
    sender: bool,
    end: float | None = None,
) -> dict:
    return {
        "sum": {
            "start": start,
            "end": start + seconds if end is None else end,
            "seconds": seconds,
            "bytes": bytes_count,
            "packets": packets,
            "lost_packets": lost_packets,
            "omitted": omitted,
            "sender": sender,
        }
    }


def _documents() -> tuple[dict, dict]:
    server = {
        "intervals": [
            _interval(
                start=0.0,
                seconds=1.0,
                bytes_count=146_000,
                packets=100,
                omitted=True,
                sender=False,
            ),
            _interval(
                start=0.0,
                seconds=1.0,
                bytes_count=146_000,
                packets=100,
                sender=False,
            ),
            _interval(
                start=1.0,
                seconds=1.0,
                bytes_count=131_400,
                packets=100,
                lost_packets=10,
                sender=False,
            ),
            _interval(
                start=2.0,
                seconds=1.0,
                bytes_count=138_700,
                packets=100,
                lost_packets=5,
                sender=False,
            ),
        ]
    }
    client = {
        "start": {
            "version": "iperf 3.16",
            "test_start": {"blksize": 1460},
        },
        "intervals": [
            _interval(
                start=0.0,
                seconds=1.0,
                bytes_count=146_000,
                packets=100,
                omitted=True,
                sender=True,
            ),
            _interval(
                start=0.0,
                seconds=1.0,
                bytes_count=146_000,
                packets=100,
                sender=True,
            ),
            _interval(
                start=1.0,
                seconds=1.0,
                bytes_count=146_000,
                packets=100,
                sender=True,
            ),
            _interval(
                start=2.0,
                seconds=1.0,
                bytes_count=146_000,
                packets=100,
                sender=True,
            ),
        ],
        "server_output_json": copy.deepcopy(server),
    }
    return client, server


def test_parse_iperf3_json_keeps_nominal_actual_and_receiver_metrics_distinct() -> None:
    client, server = _documents()
    result = parse_iperf3_json(
        client,
        server,
        nominal_offered_mbps=2,
        measurement_seconds=2,
    )

    assert result["nominal_offered_mbps"] == 2.0
    assert result["actual_offered_mbps"] == pytest.approx(1.168)
    assert result["achieved_mbps"] == pytest.approx(1.0804)
    assert result["sender_datagrams"] == 200
    assert result["receiver_total_datagrams"] == 200
    assert result["receiver_lost_datagrams"] == 15
    assert result["receiver_datagrams"] == 185
    assert result["sender_pps"] == pytest.approx(100.0)
    assert result["receiver_pps"] == pytest.approx(92.5)
    assert result["achieved_to_actual_offered_pct"] == pytest.approx(92.5)
    assert result["achieved_to_nominal_pct"] == pytest.approx(54.02)
    assert result["iperf_receiver_loss_pct"] == pytest.approx(7.5)
    assert result["iperf_udp_length_bytes"] == 1460
    assert result["iperf_intervals_used"] == 2


def test_parse_iperf3_json_rejects_incomplete_measurement_window() -> None:
    client, server = _documents()
    with pytest.raises(ValueError, match="measurement bin"):
        parse_iperf3_json(
            client,
            server,
            nominal_offered_mbps=2,
            measurement_seconds=3,
        )


def test_parse_iperf3_json_cross_checks_embedded_server_output() -> None:
    client, server = _documents()
    client["server_output_json"]["intervals"][2]["sum"]["bytes"] += 1
    with pytest.raises(ValueError, match="disagree on bytes"):
        parse_iperf3_json(
            client,
            server,
            nominal_offered_mbps=2,
            measurement_seconds=2,
        )


def test_parse_iperf3_json_skips_real_316_omit_boundary_shape() -> None:
    """Regression shape derived from local iperf3 3.16, not paper data."""
    client, server = _documents()
    malformed_client = _interval(
        start=0.999961,
        end=1.000085,
        seconds=2.000046,
        bytes_count=72_400,
        packets=100,
        sender=True,
    )
    malformed_server = _interval(
        start=0.999888,
        end=1.000073,
        seconds=1.999961,
        bytes_count=65_160,
        packets=100,
        lost_packets=10,
        sender=False,
    )
    client["intervals"][1] = malformed_client
    server["intervals"][1] = malformed_server
    client["server_output_json"] = copy.deepcopy(server)

    result = parse_iperf3_json(
        client,
        server,
        nominal_offered_mbps=2,
        measurement_seconds=2,
    )

    assert result["sender_seconds"] == pytest.approx(2.0)
    assert result["receiver_seconds"] == pytest.approx(2.0)
    assert result["sender_datagrams"] == 200
    assert result["receiver_total_datagrams"] == 200
    assert result["iperf_intervals_used"] == 2
    assert result["iperf_measurement_first_bin"] == 1
    assert result["iperf_measurement_last_bin"] == 2


@pytest.mark.parametrize("broken_side", ["client", "server"])
def test_parse_iperf3_json_rejects_insufficient_complete_intervals(
    broken_side: str,
) -> None:
    client, server = _documents()
    document = client if broken_side == "client" else server
    document["intervals"][2]["sum"].update(
        seconds=2.0,
        start=1.0,
        end=1.0,
    )
    if broken_side == "server":
        client["server_output_json"] = copy.deepcopy(server)

    with pytest.raises(ValueError, match=r"missing=\[1\]"):
        parse_iperf3_json(
            client,
            server,
            nominal_offered_mbps=2,
            measurement_seconds=2,
        )


def test_embedded_server_check_ignores_discarded_transition_bin() -> None:
    client, server = _documents()
    client["server_output_json"]["intervals"][1]["sum"]["bytes"] += 1

    result = parse_iperf3_json(
        client,
        server,
        nominal_offered_mbps=2,
        measurement_seconds=2,
    )
    assert result["receiver_datagrams"] == 185
