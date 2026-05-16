"""Tests for :mod:`workloads.control_plane_ops`.

Three test groups:

1. Unit tests for ``default_lpm_entry_generator`` — deterministic,
   prefix-unique, rejects out-of-range indices.

2. Integration tests that bring up ``linear_n.build(n_switches=2)``
   against ``l3_lpm.p4`` and exercise sync insert + read.

3. The same matrix in async mode (``run_insert_async`` + ``run_read_async``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workloads.control_plane_ops import (
    default_lpm_entry_generator,
    run_insert_async,
    run_insert_sync,
    run_read_async,
    run_read_sync,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE_NAME = "MyIngress.ipv4_lpm"


# ---------------------------------------------------------------------------
# Unit: entry generator.
# ---------------------------------------------------------------------------


def test_entry_generator_is_deterministic() -> None:
    g1 = default_lpm_entry_generator(seed=0)
    g2 = default_lpm_entry_generator(seed=0)
    for i in (0, 1, 99, 1000, (1 << 24) - 1):
        assert g1(i) == g2(i)


def test_entry_generator_unique_prefixes_within_range() -> None:
    g = default_lpm_entry_generator(seed=0)
    prefixes = {g(i)["match"]["hdr.ipv4.dst_addr"] for i in range(256)}
    assert len(prefixes) == 256


def test_entry_generator_rejects_out_of_range() -> None:
    g = default_lpm_entry_generator(seed=0)
    with pytest.raises(ValueError, match="out of supported range"):
        g(1 << 24)
    with pytest.raises(ValueError, match="out of supported range"):
        g(-1)


# ---------------------------------------------------------------------------
# Integration: sync insert + read on N=2.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_p4c
@pytest.mark.requires_bmv2
def test_control_plane_ops_sync_insert_then_read(tmp_path: Path) -> None:
    from p4net import Network

    from topologies.linear_n import build

    topo = build(n_switches=2, p4_program=REPO_ROOT / "p4" / "l3_lpm.p4")
    net = Network(topo, log_dir=tmp_path / "logs")
    net.start()
    try:
        gen = default_lpm_entry_generator(seed=42)
        insert_result = run_insert_sync(
            net=net,
            switches=["s1", "s2"],
            table_name=TABLE_NAME,
            n_entries_per_switch=10,
            entry_generator=gen,
        )
        assert insert_result["success_count"] == 20
        assert insert_result["failure_count"] == 0
        assert insert_result["total_wall_clock_s"] > 0
        assert set(insert_result["per_switch_s"].keys()) == {"s1", "s2"}

        read_result = run_read_sync(net=net, switches=["s1", "s2"], table_name=TABLE_NAME)
        assert read_result["total_entries_observed"] == 20
        assert read_result["failure_count"] == 0
    finally:
        net.stop()


# ---------------------------------------------------------------------------
# Integration: async insert + read on N=2.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_p4c
@pytest.mark.requires_bmv2
def test_control_plane_ops_async_insert_then_read(tmp_path: Path) -> None:
    from p4net import Network

    from topologies.linear_n import build

    topo = build(n_switches=2, p4_program=REPO_ROOT / "p4" / "l3_lpm.p4")
    net = Network(topo, log_dir=tmp_path / "logs")
    net.start()
    try:
        gen = default_lpm_entry_generator(seed=42)
        insert_result = run_insert_async(
            net=net,
            switches=["s1", "s2"],
            table_name=TABLE_NAME,
            n_entries_per_switch=10,
            entry_generator=gen,
        )
        assert insert_result["success_count"] == 20
        assert insert_result["failure_count"] == 0
        assert insert_result["total_wall_clock_s"] > 0
        # Async result intentionally omits per_switch_s.
        assert "per_switch_s" not in insert_result

        read_result = run_read_async(net=net, switches=["s1", "s2"], table_name=TABLE_NAME)
        assert read_result["total_entries_observed"] == 20
        assert read_result["failure_count"] == 0
    finally:
        net.stop()
