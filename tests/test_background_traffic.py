"""Tests for :mod:`workloads.background_traffic`.

Three layers:

* ``test_background_traffic_zero_rate_is_noop`` — unit test verifying
  that ``rate_mbps=0`` neither spawns processes nor touches the
  network. No root.

* ``test_background_traffic_rejects_negative_rate`` — unit test.

* ``test_background_traffic_actually_flows_10mbps`` — integration test
  that brings up ``l3_lpm.p4``, disables veth L4 offload, runs a brief
  10 Mbps UDP burst, and asserts the **receiver's veth RX counter
  actually advanced**. This is the test that would have caught the
  Phase B regression where iperf3 was silently producing zero received
  bytes due to TX checksum offload.

The byte-counter check looks at ``/sys/class/net/<iface>/statistics/rx_bytes``
instead of parsing iperf3 text output. The kernel counter is the
authoritative ground truth and is robust against iperf3 output-format
variation, against ``SIGTERM`` interrupting any final summary, and
against UDP-mode quirks where the JSON summary buffers until the
client disconnects cleanly.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from runner.host_setup import disable_l4_offload
from workloads.background_traffic import BackgroundTraffic

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_background_traffic_zero_rate_is_noop() -> None:
    bg = BackgroundTraffic(
        net=None,
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


def _rx_bytes(host: object, iface: str) -> int:
    """Read the kernel's cumulative RX byte counter for ``iface``.

    Runs ``cat`` inside the host netns because ``/sys/class/net`` is
    per-netns on Linux and only the netns sees its own interfaces.
    """
    result = host.exec(
        ["cat", f"/sys/class/net/{iface}/statistics/rx_bytes"],
        capture_output=True,
        check=True,
    )
    return int(result.stdout.strip())


@pytest.mark.integration
@pytest.mark.requires_p4c
@pytest.mark.requires_bmv2
def test_background_traffic_actually_flows_10mbps(tmp_path: Path) -> None:
    """Verify iperf3 data really lands at the receiver veth, not just that
    the iperf3 *processes* are alive — that was the Phase B gap that let
    the L4 checksum offload regression ship.
    """
    if shutil.which("iperf3") is None:
        pytest.skip("iperf3 not on PATH")

    from p4net import Network

    from topologies.single_switch import build

    rate_mbps = 10
    burst_seconds = 3.0
    expected_bytes = int(rate_mbps * 1_000_000 / 8 * burst_seconds)
    # 0.5x floor accommodates UDP loss + setup overhead; the test is
    # primarily designed to catch the zero-bytes signature of the
    # offload bug, not to certify exact throughput.
    min_acceptable_bytes = expected_bytes // 2

    topo = build(REPO_ROOT / "p4" / "l3_lpm.p4")
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
        for host_name, peer_ip, peer_mac, iface in (
            ("h1", "10.0.0.2", "00:00:00:00:00:02", "h1-eth0"),
            ("h2", "10.0.0.1", "00:00:00:00:00:01", "h2-eth0"),
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
                    iface,
                    "nud",
                    "permanent",
                ]
            )
        # Phase B's offload bug would survive if this step were skipped.
        disable_l4_offload(net, ["h1", "h2"])

        h2 = net.host("h2")
        bg = BackgroundTraffic(
            net=net,
            sender_host="h1",
            receiver_host="h2",
            sender_ip="10.0.0.1",
            receiver_ip="10.0.0.2",
            rate_mbps=rate_mbps,
            log_dir=tmp_path / "iperf3",
        )
        bg.start()
        try:
            # Lifecycle invariant: both processes alive after start().
            assert bg._server_proc is not None
            assert bg._client_proc is not None
            assert bg._server_proc.poll() is None, "iperf3 server died unexpectedly"
            assert bg._client_proc.poll() is None, "iperf3 client died unexpectedly"

            rx_before = _rx_bytes(h2, "h2-eth0")
            t_before = time.monotonic()
            time.sleep(burst_seconds)
            t_after = time.monotonic()
            rx_after = _rx_bytes(h2, "h2-eth0")
        finally:
            bg.stop()

        assert bg._server_proc is None
        assert bg._client_proc is None

        elapsed = t_after - t_before
        delta_bytes = rx_after - rx_before
        # Headline assertion — would fail under the Phase B offload bug
        # because the receiver kernel was dropping every iperf3 packet.
        assert delta_bytes > 0, (
            f"h2-eth0 received zero bytes in {elapsed:.2f}s under "
            f"{rate_mbps} Mbps iperf3 UDP — offload regression?"
        )
        assert delta_bytes >= min_acceptable_bytes, (
            f"h2-eth0 received {delta_bytes} bytes in {elapsed:.2f}s "
            f"(expected ~{expected_bytes}, floor {min_acceptable_bytes})"
        )
    finally:
        net.stop()
