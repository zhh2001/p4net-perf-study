"""Tests for :mod:`workloads.latency_probe`.

Three layers:

1. ``test_build_l3_probe_*`` / ``test_build_l2_probe_*`` — unit tests
   exercising the scapy packet construction for both wire formats. No
   root, no p4c, no BMv2.

2. ``test_run_probe_against_l3_lpm`` — integration test for the L3
   probe path through ``p4/l3_lpm.p4``.

3. ``test_run_probe_against_l2_forward`` — integration test for the L2
   probe path through ``p4/l2_forward.p4``. Programs the MAC forwarding
   table (h1 MAC → port 2, h2 MAC → port 1) and runs 10 probes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workloads.latency_probe import (
    ETHERTYPE_PROBE_L2,
    INSTRUMENT_HEADER_BYTES,
    IP_PROTO_PROBE,
    MIN_PROBE_BYTES_L2,
    MIN_PROBE_BYTES_L3,
    SEQ_BYTES,
    build_l2_probe,
    build_l3_probe,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Unit: L3 probe.
# ---------------------------------------------------------------------------


def test_build_l3_probe_has_expected_layers() -> None:
    from scapy.all import IP, Ether

    pkt = build_l3_probe(
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


def test_build_l3_probe_size_and_payload_layout() -> None:
    from scapy.all import IP

    pkt = build_l3_probe(
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
    assert payload[:INSTRUMENT_HEADER_BYTES] == b"\x00" * INSTRUMENT_HEADER_BYTES
    seq_slice = payload[INSTRUMENT_HEADER_BYTES : INSTRUMENT_HEADER_BYTES + SEQ_BYTES]
    assert int.from_bytes(seq_slice, "big") == 12345


def test_build_l3_probe_rejects_undersized() -> None:
    with pytest.raises(ValueError, match="below L3 minimum"):
        build_l3_probe(
            sender_ip="10.0.0.1",
            receiver_ip="10.0.0.2",
            sender_mac="00:00:00:00:00:01",
            receiver_mac="00:00:00:00:00:02",
            sequence=0,
            packet_size_bytes=MIN_PROBE_BYTES_L3 - 1,
        )


# ---------------------------------------------------------------------------
# Unit: L2 probe.
# ---------------------------------------------------------------------------


def test_build_l2_probe_has_expected_layers() -> None:
    from scapy.all import IP, Ether

    pkt = build_l2_probe(
        sender_mac="00:00:00:00:00:01",
        receiver_mac="00:00:00:00:00:02",
        sequence=42,
        packet_size_bytes=64,
    )
    assert Ether in pkt
    # No IPv4 in an L2 probe; scapy must not have decoded one.
    assert IP not in pkt
    assert pkt[Ether].dst == "00:00:00:00:00:02"
    assert pkt[Ether].src == "00:00:00:00:00:01"
    assert int(pkt[Ether].type) == ETHERTYPE_PROBE_L2


def test_build_l2_probe_size_and_payload_layout() -> None:
    from scapy.all import Ether

    pkt = build_l2_probe(
        sender_mac="00:00:00:00:00:01",
        receiver_mac="00:00:00:00:00:02",
        sequence=9001,
        packet_size_bytes=128,
    )
    wire = bytes(pkt)
    assert len(wire) == 128
    payload = bytes(pkt[Ether].payload)
    assert payload[:INSTRUMENT_HEADER_BYTES] == b"\x00" * INSTRUMENT_HEADER_BYTES
    seq_slice = payload[INSTRUMENT_HEADER_BYTES : INSTRUMENT_HEADER_BYTES + SEQ_BYTES]
    assert int.from_bytes(seq_slice, "big") == 9001


def test_build_l2_probe_rejects_undersized() -> None:
    with pytest.raises(ValueError, match="below L2 minimum"):
        build_l2_probe(
            sender_mac="00:00:00:00:00:01",
            receiver_mac="00:00:00:00:00:02",
            sequence=0,
            packet_size_bytes=MIN_PROBE_BYTES_L2 - 1,
        )


# ---------------------------------------------------------------------------
# Integration: L3 probe through l3_lpm.p4.
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
            sender_mac="00:00:00:00:00:01",
            receiver_mac="00:00:00:00:00:02",
            sender_ip="10.0.0.1",
            receiver_ip="10.0.0.2",
            probe_layer="l3",
            n_probes=10,
            probe_interval_ms=60.0,
            packet_size_bytes=64,
            sequence_start=0,
        )
    finally:
        net.stop()

    assert len(samples) == 10, f"expected 10 samples, got {len(samples)}: {samples}"
    assert sorted(s["sequence"] for s in samples) == list(range(10))
    for s in samples:
        assert s["switch_transit_us"] > 0, f"non-positive transit: {s}"
        assert s["switch_transit_us"] < 100_000, f"transit absurdly high: {s}"


# ---------------------------------------------------------------------------
# Integration: L2 probe through l2_forward.p4.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_p4c
@pytest.mark.requires_bmv2
def test_run_probe_against_l2_forward(tmp_path: Path) -> None:
    from p4net import Network

    from topologies.single_switch import build
    from workloads.latency_probe import run_probe

    topo = build(REPO_ROOT / "p4" / "l2_forward.p4")
    net = Network(topo, log_dir=tmp_path / "logs")
    net.start()
    try:
        sw = net.switch("s1")
        # mac_forward keys on destination MAC; send to h2 from h1 -> port 2.
        sw.client.insert_table_entry(
            "MyIngress.mac_forward",
            {"hdr.ethernet.dst_addr": "00:00:00:00:00:02"},
            "MyIngress.set_egress",
            {"port": 2},
        )
        sw.client.insert_table_entry(
            "MyIngress.mac_forward",
            {"hdr.ethernet.dst_addr": "00:00:00:00:00:01"},
            "MyIngress.set_egress",
            {"port": 1},
        )

        samples = run_probe(
            net=net,
            sender_host="h1",
            receiver_host="h2",
            sender_mac="00:00:00:00:00:01",
            receiver_mac="00:00:00:00:00:02",
            sender_ip=None,
            receiver_ip=None,
            probe_layer="l2",
            n_probes=10,
            probe_interval_ms=60.0,
            packet_size_bytes=64,
            sequence_start=0,
        )
    finally:
        net.stop()

    assert len(samples) == 10, f"expected 10 samples, got {len(samples)}: {samples}"
    assert sorted(s["sequence"] for s in samples) == list(range(10))
    for s in samples:
        assert s["switch_transit_us"] > 0, f"non-positive transit: {s}"
        assert s["switch_transit_us"] < 100_000, f"transit absurdly high: {s}"


# ---------------------------------------------------------------------------
# API validation: probe_layer dispatch.
# ---------------------------------------------------------------------------


def test_run_probe_rejects_unknown_probe_layer() -> None:
    from workloads.latency_probe import run_probe

    with pytest.raises(ValueError, match="probe_layer must be 'l2' or 'l3'"):
        run_probe(
            net=None,  # not reached; arg validation fires first
            sender_host="h1",
            receiver_host="h2",
            sender_mac="00:00:00:00:00:01",
            receiver_mac="00:00:00:00:00:02",
            sender_ip=None,
            receiver_ip=None,
            probe_layer="l4",  # type: ignore[arg-type]
            n_probes=1,
            probe_interval_ms=10.0,
            packet_size_bytes=64,
        )


def test_run_probe_l3_requires_ips() -> None:
    from workloads.latency_probe import run_probe

    with pytest.raises(ValueError, match="probe_layer='l3' requires"):
        run_probe(
            net=None,
            sender_host="h1",
            receiver_host="h2",
            sender_mac="00:00:00:00:00:01",
            receiver_mac="00:00:00:00:00:02",
            sender_ip=None,
            receiver_ip=None,
            probe_layer="l3",
            n_probes=1,
            probe_interval_ms=10.0,
            packet_size_bytes=64,
        )
