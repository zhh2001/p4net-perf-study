"""Tests for :mod:`workloads.resource_monitor`.

Unit test exercises cadence + structure against the test process'
own PID and the loopback interface.

Integration test brings up a single-switch network, runs a brief
iperf3 burst, and verifies the BMv2 PID shows non-zero CPU while
the switch-side veth shows non-zero packet rates.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from workloads.resource_monitor import ResourceMonitor

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Unit.
# ---------------------------------------------------------------------------


def test_resource_monitor_rejects_nonpositive_interval() -> None:
    with pytest.raises(ValueError, match="sample_interval_s must be > 0"):
        ResourceMonitor(sample_interval_s=0)


def test_resource_monitor_collects_at_target_cadence() -> None:
    """Run for ~1 second at 100 ms cadence; expect 8-12 samples (±20%)."""
    with ResourceMonitor(
        sample_interval_s=0.1,
        target_processes=[os.getpid()],
        target_interfaces=["lo"],
    ) as mon:
        # Generate some load so cpu_percent is not literally zero.
        end = time.monotonic() + 1.0
        x = 0
        while time.monotonic() < end:
            x = (x + 1) % 1000000
    samples = mon.samples()
    assert 7 <= len(samples) <= 13, f"unexpected sample count: {len(samples)}"
    # Schema checks on each sample.
    for s in samples:
        assert isinstance(s["timestamp_us"], int)
        assert isinstance(s["cpu_percent_total"], float)
        assert os.getpid() in s["cpu_percent_per_bmv2"]
        assert os.getpid() in s["rss_per_bmv2_bytes"]
        # lo may or may not have non-zero traffic; just check the key exists.
        assert "lo" in s["net_io_per_iface"] or s is samples[0]
    # At least one of the busy-loop samples must report non-zero CPU.
    cpus = [s["cpu_percent_per_bmv2"][os.getpid()] for s in samples[1:]]
    assert max(cpus) > 0.0


def test_resource_monitor_handles_missing_pid() -> None:
    """A bogus PID must not crash the monitor."""
    with ResourceMonitor(
        sample_interval_s=0.1,
        target_processes=[2_147_483_646],  # essentially-guaranteed unused PID
        target_interfaces=["lo"],
    ) as mon:
        time.sleep(0.3)
    samples = mon.samples()
    assert samples  # samples collected anyway
    # No bogus PID should appear in per-bmv2 maps.
    for s in samples:
        assert 2_147_483_646 not in s["cpu_percent_per_bmv2"]


# ---------------------------------------------------------------------------
# Integration.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_p4c
@pytest.mark.requires_bmv2
def test_resource_monitor_captures_bmv2_under_load(tmp_path: Path) -> None:
    if shutil.which("iperf3") is None:
        pytest.skip("iperf3 not on PATH")

    from p4net import Network

    from runner.host_setup import disable_l4_offload
    from topologies.single_switch import build
    from workloads.background_traffic import BackgroundTraffic

    topo = build(REPO_ROOT / "p4" / "l3_lpm.p4")
    net = Network(topo, log_dir=tmp_path / "logs")
    net.start()
    try:
        sw = net.switch("s1")
        bmv2_pid = sw.bmv2.pid
        assert bmv2_pid is not None and bmv2_pid > 0

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

        bg = BackgroundTraffic(
            net=net,
            sender_host="h1",
            receiver_host="h2",
            sender_ip="10.0.0.1",
            receiver_ip="10.0.0.2",
            rate_mbps=10,
        )
        with ResourceMonitor(
            sample_interval_s=0.1,
            target_processes=[bmv2_pid],
            target_interfaces=["s1-eth1", "s1-eth2"],
        ) as mon:
            bg.start()
            try:
                time.sleep(2.0)
            finally:
                bg.stop()

        samples = mon.samples()
        assert len(samples) >= 15
        # BMv2 should have done *some* work moving 20 Mbps total.
        cpu_samples = [s["cpu_percent_per_bmv2"].get(bmv2_pid, 0.0) for s in samples]
        assert max(cpu_samples) > 0.0, f"BMv2 cpu_percent never advanced: {cpu_samples!r}"
        # At least one switch-side veth should have observed packets.
        any_traffic = any(
            s["net_io_per_iface"].get("s1-eth1", {}).get("rx_pps", 0.0) > 0.0
            or s["net_io_per_iface"].get("s1-eth2", {}).get("tx_pps", 0.0) > 0.0
            for s in samples
        )
        assert any_traffic, "no switch-side veth packets observed"
    finally:
        net.stop()
