"""Tests for :mod:`topologies.linear_n`.

Unit tests verify the structural shape of the produced topology (host
count, switch count, link count) without bringing the network up.
Integration tests bring up an N=2 instance against ``l3_lpm.p4`` and
exercise both the sync and async P4Runtime clients on each switch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from topologies.linear_n import build

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Unit: structural shape.
# ---------------------------------------------------------------------------


def test_linear_n_rejects_n_lt_1() -> None:
    with pytest.raises(ValueError, match="n_switches must be >= 1"):
        build(n_switches=0, p4_program=REPO_ROOT / "p4" / "l3_lpm.p4")


def test_linear_n_n1_matches_single_switch_shape() -> None:
    topo = build(n_switches=1, p4_program=REPO_ROOT / "p4" / "l3_lpm.p4")
    assert len(topo.hosts) == 2
    assert len(topo.switches) == 1
    # Two links: h1<->s1 and s1<->h2.
    assert len(topo.links) == 2


def test_linear_n_n4_has_5_links() -> None:
    topo = build(n_switches=4, p4_program=REPO_ROOT / "p4" / "l3_lpm.p4")
    assert len(topo.hosts) == 2
    assert len(topo.switches) == 4
    assert len(topo.links) == 5  # h1-s1, s1-s2, s2-s3, s3-s4, s4-h2


def test_linear_n_per_link_subnet_layout() -> None:
    topo = build(n_switches=3, p4_program=REPO_ROOT / "p4" / "l3_lpm.p4")
    host_ips = {h.name: h.ip for h in topo.hosts.values()}
    assert host_ips["h1"] == "10.0.0.1/30"
    assert host_ips["h2"] == "10.0.3.1/30"


def test_linear_n_flat_subnet_layout() -> None:
    topo = build(
        n_switches=2,
        p4_program=REPO_ROOT / "p4" / "l3_lpm.p4",
        subnet_per_switch=False,
    )
    host_ips = {h.name: h.ip for h in topo.hosts.values()}
    assert host_ips["h1"] == "10.0.0.1/24"
    assert host_ips["h2"] == "10.0.0.2/24"


def test_linear_n_rejects_unsupported_endpoint_fanout() -> None:
    with pytest.raises(NotImplementedError, match="hosts_per_endpoint=1"):
        build(
            n_switches=2,
            p4_program=REPO_ROOT / "p4" / "l3_lpm.p4",
            hosts_per_endpoint=2,
        )


# ---------------------------------------------------------------------------
# Integration: N=2 brings up cleanly with both sync and async clients.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_p4c
@pytest.mark.requires_bmv2
def test_linear_n2_brings_up_with_async_clients(tmp_path: Path) -> None:
    """N=2 topology comes up cleanly and both async clients connect.

    This test must be plain ``def`` (not ``async def``): the
    pytest-asyncio harness already owns an event loop, but
    ``Network.start()`` spawns its own loop for netlink/grpc setup and
    the two collide. Running our async work inside ``asyncio.run()``
    from a sync test isolates them.
    """
    import asyncio

    from p4net import Network

    topo = build(n_switches=2, p4_program=REPO_ROOT / "p4" / "l3_lpm.p4")
    net = Network(topo, log_dir=tmp_path / "logs")
    net.start()
    try:
        s1 = net.switch("s1")
        s2 = net.switch("s2")
        assert s1.client is not None
        assert s2.client is not None
        # Confirm both async clients reach their gRPC endpoint. They are
        # lazy-constructed; ``connect()`` / ``disconnect()`` are
        # coroutines, so drive them via ``asyncio.run``.
        a1 = s1.async_client
        a2 = s2.async_client

        async def _connect_both() -> None:
            await asyncio.gather(a1.connect(), a2.connect())
            assert a1.is_connected
            assert a2.is_connected
            await asyncio.gather(a1.disconnect(), a2.disconnect())

        asyncio.run(_connect_both())
    finally:
        net.stop()
