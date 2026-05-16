"""Tests for :mod:`workloads.int_collector`.

Three layers:

1. Unit: ``build_int_probe`` produces a valid packet (Ethernet + IPv4
   proto 0xFD + instrument + int_meta + seq + pad), and
   ``_decode_int_payload`` round-trips a hand-built shim stack.

2. Unit: ``_enrich_samples`` reverses front-of-stack-first shim arrays
   to chronological order and computes per-hop drift correctly.

3. Integration: bring up ``linear_n.build(n_switches=2)`` against
   ``p4/l3_lpm_int_chain.p4``, program switch_id registers + forwarding
   entries, run ``run_collection(...)``, assert 10 probes return with
   ``hop_count == 2``, ``switch_ids == [1, 2]``, and finite drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workloads.int_collector import (
    INSTRUMENT_HEADER_BYTES,
    INT_META_BYTES,
    IP_PROTO_PROBE,
    MIN_PROBE_BYTES,
    SEQ_BYTES,
    SHIM_BYTES,
    _decode_int_payload,
    _enrich_samples,
    build_int_probe,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Unit: packet construction.
# ---------------------------------------------------------------------------


def test_build_int_probe_has_expected_layers() -> None:
    from scapy.all import IP, Ether

    pkt = build_int_probe(
        sender_ip="10.0.0.1",
        receiver_ip="10.0.0.2",
        sender_mac="00:00:00:00:00:01",
        receiver_mac="00:00:00:00:00:02",
        sequence=13,
        packet_size_bytes=64,
    )
    assert Ether in pkt and IP in pkt
    assert int(pkt[IP].proto) == IP_PROTO_PROBE
    assert pkt[IP].src == "10.0.0.1"
    assert pkt[IP].dst == "10.0.0.2"


def test_build_int_probe_size_and_initial_layout() -> None:
    from scapy.all import IP

    pkt = build_int_probe(
        sender_ip="10.0.0.1",
        receiver_ip="10.0.0.2",
        sender_mac="00:00:00:00:00:01",
        receiver_mac="00:00:00:00:00:02",
        sequence=4321,
        packet_size_bytes=128,
    )
    wire = bytes(pkt)
    assert len(wire) == 128
    payload = bytes(pkt[IP].payload)
    # instrument zeroed, int_meta hop_count = 0
    assert payload[:INSTRUMENT_HEADER_BYTES] == b"\x00" * INSTRUMENT_HEADER_BYTES
    assert payload[INSTRUMENT_HEADER_BYTES] == 0  # hop_count
    # seq immediately follows int_meta (no shims at sender)
    seq_offset = INSTRUMENT_HEADER_BYTES + INT_META_BYTES
    seq = int.from_bytes(payload[seq_offset : seq_offset + SEQ_BYTES], "big")
    assert seq == 4321


def test_build_int_probe_rejects_undersized() -> None:
    with pytest.raises(ValueError, match="below INT minimum"):
        build_int_probe(
            sender_ip="10.0.0.1",
            receiver_ip="10.0.0.2",
            sender_mac="00:00:00:00:00:01",
            receiver_mac="00:00:00:00:00:02",
            sequence=0,
            packet_size_bytes=MIN_PROBE_BYTES - 1,
        )


# ---------------------------------------------------------------------------
# Unit: decoder round-trips a hand-built two-hop payload.
# ---------------------------------------------------------------------------


def test_decode_int_payload_two_hops() -> None:
    instrument = (0).to_bytes(6, "big") + (1000).to_bytes(6, "big")
    int_meta = bytes([2])  # two hops
    # Most-recent hop (s2) at front of stack.
    shim_s2 = bytes([2]) + (500).to_bytes(6, "big") + (550).to_bytes(6, "big")
    shim_s1 = bytes([1]) + (100).to_bytes(6, "big") + (150).to_bytes(6, "big")
    seq = (77).to_bytes(SEQ_BYTES, "big")
    payload = instrument + int_meta + shim_s2 + shim_s1 + seq + b"\x00" * 8

    decoded = _decode_int_payload(payload)
    assert decoded is not None
    assert decoded["sequence"] == 77
    assert decoded["hop_count"] == 2
    assert decoded["shims"][0]["switch_id"] == 2  # front-of-stack
    assert decoded["shims"][1]["switch_id"] == 1
    assert decoded["shims"][0]["ingress_ts_us"] == 500
    assert decoded["shims"][1]["ingress_ts_us"] == 100


def test_decode_int_payload_truncated_returns_none() -> None:
    # hop_count claims 2 shims but payload only has bytes for one.
    instrument = b"\x00" * INSTRUMENT_HEADER_BYTES
    int_meta = bytes([2])
    shim_one = bytes(SHIM_BYTES)
    payload = instrument + int_meta + shim_one  # short by 1 shim + seq
    assert _decode_int_payload(payload) is None


def test_decode_int_payload_rejects_oversized_hop_count() -> None:
    instrument = b"\x00" * INSTRUMENT_HEADER_BYTES
    int_meta = bytes([9])  # MAX_HOPS = 8
    payload = instrument + int_meta + b"\x00" * 200
    assert _decode_int_payload(payload) is None


# ---------------------------------------------------------------------------
# Unit: enrich + drift computation.
# ---------------------------------------------------------------------------


def test_enrich_samples_reverses_to_chronological_order() -> None:
    raw = [
        {
            "sequence": 1,
            "hop_count": 2,
            "instrument_ingress_us": 0,
            "instrument_egress_us": 0,
            # Front-of-stack first: s2 then s1
            "shims": [
                {"switch_id": 2, "ingress_ts_us": 500, "egress_ts_us": 550},
                {"switch_id": 1, "ingress_ts_us": 100, "egress_ts_us": 150},
            ],
        }
    ]
    boot = {1: 1_000_000_000, 2: 1_000_000_050}
    enriched = _enrich_samples(raw, boot)
    assert len(enriched) == 1
    s = enriched[0]
    assert s["switch_ids"] == [1, 2]
    assert s["raw_ingress_us"] == [100, 500]
    assert s["raw_egress_us"] == [150, 550]
    assert s["aligned_ingress_us"] == [1_000_000_100, 1_000_000_550]
    assert s["aligned_egress_us"] == [1_000_000_150, 1_000_000_600]
    # drift[0] = aligned_ingress[1] - aligned_egress[0]
    #          = 1_000_000_550 - 1_000_000_150 = 400
    assert s["drift_us"] == [400]
    assert s["avg_drift_us"] == 400.0


# ---------------------------------------------------------------------------
# Integration: 10 probes through 2-switch chain.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_p4c
@pytest.mark.requires_bmv2
def test_run_collection_against_2_hop_chain(tmp_path: Path) -> None:
    from p4net import Network

    from runner.host_setup import disable_l4_offload
    from topologies.linear_n import build
    from workloads.int_collector import run_collection

    topo = build(
        n_switches=2,
        p4_program=REPO_ROOT / "p4" / "l3_lpm_int_chain.p4",
        subnet_per_switch=False,
    )
    net = Network(topo, log_dir=tmp_path / "logs")
    net.start()
    try:
        # Each switch needs its switch_id and forwarding entries.
        # For a 2-hop chain h1 -- s1 -- s2 -- h2:
        #   s1: forward dst=h1 → port 1 (left), dst=h2 → port 2 (right)
        #   s2: forward dst=h1 → port 1 (left), dst=h2 → port 2 (right)
        for idx, sw_name in enumerate(("s1", "s2"), start=1):
            sw = net.switch(sw_name)
            sw.client.write_register("MyEgress.switch_id_reg", 0, idx)
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
        for host_name, peer_ip, peer_mac in (
            ("h1", "10.0.0.2", "00:00:00:00:00:02"),
            ("h2", "10.0.0.1", "00:00:00:00:00:01"),
        ):
            net.host(host_name).exec(
                [
                    "ip",
                    "neigh",
                    "replace",
                    peer_ip,
                    "lladdr",
                    peer_mac,
                    "dev",
                    f"{host_name}-eth0",
                    "nud",
                    "permanent",
                ]
            )
        disable_l4_offload(net, ["h1", "h2"])

        samples = run_collection(
            net=net,
            sender_host="h1",
            receiver_host="h2",
            sender_mac="00:00:00:00:00:01",
            receiver_mac="00:00:00:00:00:02",
            sender_ip="10.0.0.1",
            receiver_ip="10.0.0.2",
            switch_names=["s1", "s2"],
            n_probes=10,
            probe_interval_ms=60.0,
            packet_size_bytes=128,
        )
    finally:
        net.stop()

    assert len(samples) == 10, f"expected 10 samples, got {len(samples)}: {samples}"
    for s in samples:
        assert s["hop_count"] == 2
        assert s["switch_ids"] == [1, 2]
        assert len(s["drift_us"]) == 1
        # drift can be negative if BMv2 timestamps' wall-clock alignment
        # is dominated by per-switch boot-time skew that exceeds the
        # inter-hop propagation delay; we only require finite values.
        assert isinstance(s["drift_us"][0], int)
        assert s["aligned_ingress_us"][1] > 0
        assert s["aligned_egress_us"][0] > 0
