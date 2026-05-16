"""Regression test for the Phase B open question on ``l3_lpm_int.p4``.

Before the Phase C fix, the egress control rewrote every forwarded IPv4
packet's outer Ethernet etherType to ``ETHERTYPE_INT`` (0x88B6), so
background iperf3 UDP traffic was delivered to the receiver kernel with
an ether_type the IP stack does not handle and silently dropped. After
the fix, the rewrite is conditioned on ``hdr.instrument.isValid()`` so
only probe frames carry the INT shim; background IPv4 keeps its 0x0800
etherType and the iperf3 server receives it normally.

The test brings up the single-switch topology against ``l3_lpm_int.p4``,
runs iperf3 client → server for 5 seconds at 100 Mbps, and asserts the
client-side JSON summary reports non-zero bytes received. Zero bytes
would indicate the conditional fix has been undone.
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
