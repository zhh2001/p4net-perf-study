"""Tests for :mod:`workloads.control_plane_ops`.

Three test groups:

1. Unit tests for ``default_lpm_entry_generator`` — deterministic,
   prefix-unique, rejects out-of-range indices.

2. Integration tests that bring up ``linear_n.build(n_switches=2)``
   against ``l3_lpm.p4`` and exercise sync insert + read.

3. The same matrix in async mode (``run_insert_async`` + ``run_read_async``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from workloads.control_plane_ops import (
    AsyncWorkloadLoop,
    default_lpm_entry_generator,
    run_insert_async,
    run_insert_sync,
    run_read_async,
    run_read_sync,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLE_NAME = "MyIngress.ipv4_lpm"


class _FakeAsyncClient:
    def __init__(self, events: list[tuple[str, str, int]], switch_name: str) -> None:
        self._events = events
        self._switch_name = switch_name
        self._entry_count = 0

    def _record(self, event: str) -> None:
        self._events.append(
            (event, self._switch_name, id(asyncio.get_running_loop()))
        )

    async def connect(self) -> None:
        self._record("connect")

    async def disconnect(self) -> None:
        self._record("disconnect")

    async def insert_table_entry(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs
        self._record("insert")
        self._entry_count += 1

    def list_table_entries(self, table_name: str):
        _ = table_name

        async def _entries():
            self._record("read")
            for _ in range(self._entry_count):
                yield {}

        return _entries()


class _FakeNetwork:
    def __init__(self, events: list[tuple[str, str, int]], switches: list[str]) -> None:
        self._switches = {
            name: SimpleNamespace(async_client=_FakeAsyncClient(events, name))
            for name in switches
        }

    def switch(self, name: str) -> SimpleNamespace:
        return self._switches[name]


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


def test_async_prefill_and_read_share_one_open_loop_and_separate_timers() -> None:
    """Regression: read prefill must not close its loop before measurement."""
    switches = ["s1", "s2"]
    events: list[tuple[str, str, int]] = []
    net = _FakeNetwork(events, switches)
    generator = default_lpm_entry_generator(seed=0)

    # Two calls per phase: start and end.  Distinct intervals prove that the
    # primary read timer starts only after the prefill phase has completed.
    with (
        AsyncWorkloadLoop() as workload_loop,
        patch(
            "workloads.control_plane_ops.asyncio.run",
            side_effect=AssertionError("phase opened a second event loop"),
        ),
        patch(
            "workloads.control_plane_ops.time.perf_counter",
            side_effect=[10.0, 15.0, 100.0, 102.0],
        ),
    ):
        prefill = run_insert_async(
            net=net,
            switches=switches,
            table_name=TABLE_NAME,
            n_entries_per_switch=3,
            entry_generator=generator,
            workload_loop=workload_loop,
        )
        primary = run_read_async(
            net=net,
            switches=switches,
            table_name=TABLE_NAME,
            workload_loop=workload_loop,
        )

    assert prefill["success_count"] == 6
    assert prefill["total_wall_clock_s"] == 5.0
    assert primary["total_entries_observed"] == 6
    assert primary["total_wall_clock_s"] == 2.0
    assert len({loop_id for _, _, loop_id in events}) == 1
    assert max(i for i, event in enumerate(events) if event[0] == "insert") < min(
        i for i, event in enumerate(events) if event[0] == "read"
    )
    for switch_name in switches:
        switch_events = [event for event, switch, _ in events if switch == switch_name]
        assert switch_events == [
            "connect",
            "insert",
            "insert",
            "insert",
            "disconnect",
            "connect",
            "read",
            "disconnect",
        ]


def test_async_workload_loop_turns_callback_errors_into_failures() -> None:
    with AsyncWorkloadLoop() as workload_loop:

        async def _fail_in_callback() -> None:
            loop = asyncio.get_running_loop()
            loop.call_soon(lambda: 1 / 0)
            await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="event loop reported 1 unhandled error"):
            workload_loop.run(_fail_in_callback())


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
        with AsyncWorkloadLoop() as workload_loop:
            insert_result = run_insert_async(
                net=net,
                switches=["s1", "s2"],
                table_name=TABLE_NAME,
                n_entries_per_switch=10,
                entry_generator=gen,
                workload_loop=workload_loop,
            )
            assert insert_result["success_count"] == 20
            assert insert_result["failure_count"] == 0
            assert insert_result["total_wall_clock_s"] > 0
            # Async result intentionally omits per_switch_s.
            assert "per_switch_s" not in insert_result

            read_result = run_read_async(
                net=net,
                switches=["s1", "s2"],
                table_name=TABLE_NAME,
                workload_loop=workload_loop,
            )
            assert read_result["total_entries_observed"] == 20
            assert read_result["failure_count"] == 0
    finally:
        net.stop()
