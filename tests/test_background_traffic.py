"""Tests for :mod:`workloads.background_traffic`.

Two layers:

* ``test_background_traffic_zero_rate_is_noop`` — unit test verifying
  that ``rate_mbps=0`` neither spawns processes nor touches the
  network. Runs without root.

* ``test_background_traffic_lifecycle_at_10mbps`` — integration test
  bringing up ``l3_lpm.p4``, programming forwarding + static ARP, then
  starting and stopping a 10 Mbps UDP iperf3 pair. Asserts both
  processes are alive after ``start()`` and both are reaped after
  ``stop()``, with no zombies left behind.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from workloads.background_traffic import BackgroundTraffic

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_background_traffic_zero_rate_is_noop() -> None:
    bg = BackgroundTraffic(
        net=None,  # never dereferenced when rate_mbps == 0
        sender_host="h1",
        receiver_host="h2",
        sender_ip="10.0.0.1",
        receiver_ip="10.0.0.2",
        rate_mbps=0,
    )
    bg.start()
    assert bg._server_proc is None
    assert bg._client_proc is None
    bg.stop()
    assert bg._server_proc is None
    assert bg._client_proc is None


def test_background_traffic_rejects_negative_rate() -> None:
    with pytest.raises(ValueError, match="rate_mbps must be >= 0"):
        BackgroundTraffic(
            net=None,
            sender_host="h1",
            receiver_host="h2",
            sender_ip="10.0.0.1",
            receiver_ip="10.0.0.2",
            rate_mbps=-1,
        )


@pytest.mark.integration
@pytest.mark.requires_p4c
@pytest.mark.requires_bmv2
def test_background_traffic_lifecycle_at_10mbps(tmp_path: Path) -> None:
    if shutil.which("iperf3") is None:
        pytest.skip("iperf3 not on PATH")

    from p4net import Network

    from topologies.single_switch import build

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
        for host_name, peer_ip, peer_mac, iface in [
            ("h1", "10.0.0.2", "00:00:00:00:00:02", "h1-eth0"),
            ("h2", "10.0.0.1", "00:00:00:00:00:01", "h2-eth0"),
        ]:
            net.host(host_name).exec(
                [
                    "ip",
                    "neigh",
                    "replace",
                    peer_ip,
                    "lladdr",
                    peer_mac,
                    "dev",
                    iface,
                    "nud",
                    "permanent",
                ]
            )

        bg = BackgroundTraffic(
            net=net,
            sender_host="h1",
            receiver_host="h2",
            sender_ip="10.0.0.1",
            receiver_ip="10.0.0.2",
            rate_mbps=10,
            log_dir=tmp_path / "iperf3",
        )
        bg.start()
        # Give iperf3 a moment to establish + push some packets.
        time.sleep(1.0)
        assert bg._server_proc is not None
        assert bg._client_proc is not None
        assert bg._server_proc.poll() is None, "iperf3 server died unexpectedly"
        assert bg._client_proc.poll() is None, "iperf3 client died unexpectedly"

        bg.stop()
        assert bg._server_proc is None
        assert bg._client_proc is None

        # Server log should exist and have non-zero bytes (banner + reports).
        server_log = tmp_path / "iperf3" / "iperf3_server.log"
        assert server_log.is_file()
        assert server_log.stat().st_size > 0
    finally:
        net.stop()
