"""Tests for :mod:`workloads.latency_probe`.

Two layers:

1. ``test_build_probe_packet_*`` — unit tests that exercise just the
   scapy packet construction. They do not require root, p4c, or BMv2.

2. ``test_run_probe_against_l3_lpm`` — integration test that brings up
   the single-switch topology against ``p4/l3_lpm.p4``, programs the
   forwarding table, calls :func:`run_probe` with 10 probes, and
   asserts every sample's switch-transit latency is positive and below
   a 100 ms ceiling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workloads.latency_probe import (
    INSTRUMENT_HEADER_BYTES,
    IP_PROTO_PROBE,
    MIN_PROBE_BYTES,
    SEQ_BYTES,
    build_probe_packet,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Unit: probe packet construction.
# ---------------------------------------------------------------------------


def test_build_probe_packet_has_expected_layers() -> None:
    from scapy.all import IP, Ether

    pkt = build_probe_packet(
        sender_ip="10.0.0.1",
        receiver_ip="10.0.0.2",
        sender_mac="00:00:00:00:00:01",
        receiver_mac="00:00:00:00:00:02",
        sequence=7,
        packet_size_bytes=64,
    )
    assert Ether in pkt and IP in pkt
    assert pkt[Ether].dst == "00:00:00:00:00:02"
    assert pkt[Ether].src == "00:00:00:00:00:01"
    assert pkt[IP].src == "10.0.0.1"
    assert pkt[IP].dst == "10.0.0.2"
    assert int(pkt[IP].proto) == IP_PROTO_PROBE


def test_build_probe_packet_size_and_payload_layout() -> None:
    from scapy.all import IP

    pkt = build_probe_packet(
        sender_ip="10.0.0.1",
        receiver_ip="10.0.0.2",
        sender_mac="00:00:00:00:00:01",
        receiver_mac="00:00:00:00:00:02",
        sequence=12345,
        packet_size_bytes=128,
    )
    wire = bytes(pkt)
    assert len(wire) == 128
    payload = bytes(pkt[IP].payload)
    # First 12 bytes are the zeroed instrument header.
    assert payload[:INSTRUMENT_HEADER_BYTES] == b"\x00" * INSTRUMENT_HEADER_BYTES
    # Next 4 bytes are the sequence number, big-endian.
    seq_slice = payload[INSTRUMENT_HEADER_BYTES : INSTRUMENT_HEADER_BYTES + SEQ_BYTES]
    seq = int.from_bytes(seq_slice, "big")
    assert seq == 12345


def test_build_probe_packet_rejects_undersized() -> None:
    with pytest.raises(ValueError, match="below minimum"):
        build_probe_packet(
            sender_ip="10.0.0.1",
            receiver_ip="10.0.0.2",
            sender_mac="00:00:00:00:00:01",
            receiver_mac="00:00:00:00:00:02",
            sequence=0,
            packet_size_bytes=MIN_PROBE_BYTES - 1,
        )


# ---------------------------------------------------------------------------
# Integration: 10 probes through l3_lpm.p4.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_p4c
@pytest.mark.requires_bmv2
def test_run_probe_against_l3_lpm(tmp_path: Path) -> None:
    from p4net import Network

    from topologies.single_switch import build
    from workloads.latency_probe import run_probe

    topo = build(REPO_ROOT / "p4" / "l3_lpm.p4")
    net = Network(topo, log_dir=tmp_path / "logs")
    net.start()
    try:
        sw = net.switch("s1")
        sw.client.insert_table_entry(
            "MyIngress.ipv4_lpm",
            {"hdr.ipv4.dst_addr": "10.0.0.1/32"},
            "MyIngress.set_nhop",
            {"nhop_mac": "00:00:00:00:00:01", "port": 1},
        )
        sw.client.insert_table_entry(
            "MyIngress.ipv4_lpm",
            {"hdr.ipv4.dst_addr": "10.0.0.2/32"},
            "MyIngress.set_nhop",
            {"nhop_mac": "00:00:00:00:00:02", "port": 2},
        )

        samples = run_probe(
            net=net,
            sender_host="h1",
            receiver_host="h2",
            sender_ip="10.0.0.1",
            receiver_ip="10.0.0.2",
            receiver_mac="00:00:00:00:00:02",
            n_probes=10,
            probe_interval_ms=60.0,
            packet_size_bytes=64,
            sequence_start=0,
        )
    finally:
        net.stop()

    assert len(samples) == 10, f"expected 10 samples, got {len(samples)}: {samples}"
    seqs = sorted(s["sequence"] for s in samples)
    assert seqs == list(range(10))
    for s in samples:
        assert s["switch_transit_us"] > 0, f"non-positive transit: {s}"
        # Generous ceiling: real BMv2 transit is typically tens to hundreds of μs.
        assert s["switch_transit_us"] < 100_000, f"transit absurdly high: {s}"
