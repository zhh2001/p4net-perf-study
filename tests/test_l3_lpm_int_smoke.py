"""Regression tests for ``l3_lpm_int.p4``.

Phase C added the conditional INT branch (probe-only). Phase G further
restructured the wire format so the L3 latency_probe receiver can
decode probes through l3_lpm_int without modification — see the file
header in ``p4/l3_lpm_int.p4`` for the format change rationale.

Two integration tests live here:

* ``test_iperf3_through_l3_lpm_int`` — Phase C regression. Background
  iperf3 UDP through l3_lpm_int still receives non-zero bytes (the
  conditional etherType + shim-only-on-probe branch holds).

* ``test_latency_probe_through_l3_lpm_int`` — Phase G regression. An
  L3 latency probe through l3_lpm_int returns non-empty samples with
  ``switch_transit_us > 0``; this is the fix that lets the 9 previously
  missing l3_lpm_int RQ1 matrix cells produce measurable data.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.integration
@pytest.mark.requires_p4c
@pytest.mark.requires_bmv2
def test_iperf3_through_l3_lpm_int(tmp_path: Path) -> None:
    if shutil.which("iperf3") is None:
        pytest.skip("iperf3 not on PATH")

    from p4net import Network

    from topologies.single_switch import build

    topo = build(REPO_ROOT / "p4" / "l3_lpm_int.p4")
    net = Network(topo, log_dir=tmp_path / "logs")
    net.start()
    try:
        sw = net.switch("s1")
        for ip, mac, port in (
            ("10.0.0.1", "00:00:00:00:00:01", 1),
            ("10.0.0.2", "00:00:00:00:00:02", 2),
        ):
            sw.client.insert_table_entry(
                "MyIngress.ipv4_lpm",
                {"hdr.ipv4.dst_addr": f"{ip}/32"},
                "MyIngress.set_nhop",
                {"nhop_mac": mac, "port": port},
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

        # Disable L4 checksum offload on the host veths. BMv2 does not
        # recompute TCP/UDP checksums and Linux's default veth offload
        # ships packets with a zero L4 checksum, so the receiver kernel
        # drops them. ICMP doesn't carry an L4 checksum and works
        # without this step — which is why ping passes but iperf3 fails
        # if this is omitted.
        for host_name in ("h1", "h2"):
            net.host(host_name).exec(
                [
                    "ethtool",
                    "-K",
                    f"{host_name}-eth0",
                    "tx",
                    "off",
                    "rx",
                    "off",
                    "tso",
                    "off",
                    "gso",
                    "off",
                    "gro",
                    "off",
                ],
                capture_output=True,
                check=False,
            )

        h2 = net.host("h2")
        h1 = net.host("h1")
        server_log = tmp_path / "iperf3_server.log"
        with open(server_log, "wb") as server_log_fh:
            server_proc = h2.popen(
                ["iperf3", "-s", "-1", "-p", "5201"],
                stdout=server_log_fh,
                stderr=server_log_fh,
            )
            try:
                # 5-second 100 Mbps UDP burst (-J = JSON summary on stdout).
                result = h1.exec(
                    [
                        "iperf3",
                        "-c",
                        "10.0.0.2",
                        "-p",
                        "5201",
                        "-u",
                        "-b",
                        "100M",
                        "-t",
                        "5",
                        "-J",
                    ],
                    capture_output=True,
                    check=False,
                    timeout=30.0,
                )
            finally:
                if server_proc.poll() is None:
                    server_proc.terminate()
                    try:
                        server_proc.wait(timeout=5.0)
                    except Exception:
                        server_proc.kill()
                        server_proc.wait()

        assert result.returncode == 0, (
            f"iperf3 client rc={result.returncode}\nstderr={result.stderr.decode(errors='replace')}"
        )
        report = json.loads(result.stdout.decode("utf-8", errors="replace"))
        # The client-side JSON summary in iperf3 -u reports the server's
        # sum_received block: that is the authoritative "did anything
        # actually arrive?" datum.
        bytes_received = int(report["end"]["sum_received"]["bytes"])
        assert bytes_received > 0, (
            "iperf3 server reported zero bytes received — INT etherType "
            "rewrite is still hitting background IPv4 traffic.\n"
            f"full report: {json.dumps(report, indent=2)}"
        )
    finally:
        net.stop()


@pytest.mark.integration
@pytest.mark.requires_p4c
@pytest.mark.requires_bmv2
def test_latency_probe_through_l3_lpm_int(tmp_path: Path) -> None:
    """An L3 probe through l3_lpm_int.p4 must return non-empty samples
    with positive switch_transit_us.

    The Phase G restructure keeps the outer Ethernet/IPv4 envelope
    intact (no etherType rewrite) and appends the int_shim after the
    instrument header. The receiver decodes the IPv4 frame normally;
    the additional shim bytes sit between instrument and the
    sequence + padding payload, which only affects the sequence-number
    field's decoding — not the switch_transit_us calculation. So the
    JSONL ``value`` column is correct; only ``extras.sequence`` is
    garbled, which the aggregator does not use.
    """
    from p4net import Network

    from runner.host_setup import disable_l4_offload
    from topologies.single_switch import build
    from workloads.latency_probe import run_probe

    topo = build(REPO_ROOT / "p4" / "l3_lpm_int.p4")
    net = Network(topo, log_dir=tmp_path / "logs")
    net.start()
    try:
        sw = net.switch("s1")
        for ip, mac, port in (
            ("10.0.0.1", "00:00:00:00:00:01", 1),
            ("10.0.0.2", "00:00:00:00:00:02", 2),
        ):
            sw.client.insert_table_entry(
                "MyIngress.ipv4_lpm",
                {"hdr.ipv4.dst_addr": f"{ip}/32"},
                "MyIngress.set_nhop",
                {"nhop_mac": mac, "port": port},
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
            packet_size_bytes=128,
        )
    finally:
        net.stop()

    assert len(samples) == 10, f"expected 10 samples, got {len(samples)}: {samples}"
    for s in samples:
        # The shim adds a fixed ~µs of egress work; switch_transit_us
        # must still be a positive number on the order of single-digit
        # to low-double-digit μs.
        assert s["switch_transit_us"] > 0, f"non-positive transit: {s}"
        assert s["switch_transit_us"] < 100_000, f"transit absurdly high: {s}"
