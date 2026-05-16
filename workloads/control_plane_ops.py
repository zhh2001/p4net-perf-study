"""Control-plane operation workloads for RQ2 (multi-switch scaling).

Drives insert / read operations against N switches in either synchronous
or asynchronous mode and records wall-clock timing per repetition. The
unit of measurement for RQ2 is "one batch of K entries pushed to each of
N switches" — the runner repeats this batch some number of times per
configuration to surface variance.

Sync vs. async layering:

* Sync mode goes through :class:`p4net.control.P4RuntimeClient` and walks
  switches in series, entries in series within each switch. Per-switch
  wall-clock is reported.

* Async mode goes through :class:`p4net.control.AsyncP4RuntimeClient`.
  We parallelize across switches with ``asyncio.gather`` and walk
  entries serially within each switch — the gRPC stream is sequential
  per-client by design, but the N streams to N switches all run
  concurrently. Per-switch wall-clock is not reported in async mode
  because the per-switch coroutines interleave on the event loop.

Reproducibility: ``default_lpm_entry_generator`` produces unique
deterministic ``10.{i//65536}.{(i//256)%256}.{i%256}/32`` entries keyed
by the index alone, so re-running the same config yields the same
table state. The ``seed`` parameter is wired through so future variants
(e.g., randomized ternary ACL entries) can plug in without changing
call sites.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import p4net

logger = logging.getLogger(__name__)


EntryGenerator = Callable[[int], dict[str, Any]]


def default_lpm_entry_generator(seed: int = 0) -> EntryGenerator:
    """Return a deterministic generator for IPv4 LPM /32 entries.

    For index ``i`` the generated entry has destination prefix
    ``10.(i>>16).((i>>8)&0xFF).(i&0xFF)/32``. The seed is currently
    unused — kept in the signature so future variants (randomized
    ternary, weighted prefix lengths, etc.) can swap in without
    breaking call sites.
    """
    _ = seed  # reserved for future variants

    def gen(i: int) -> dict[str, Any]:
        if not (0 <= i < (1 << 24)):
            raise ValueError(f"entry index {i} out of supported range [0, 2^24)")
        a = (i >> 16) & 0xFF
        b = (i >> 8) & 0xFF
        c = i & 0xFF
        return {
            "match": {"hdr.ipv4.dst_addr": f"10.{a}.{b}.{c}/32"},
            "action_name": "MyIngress.set_nhop",
            "params": {"nhop_mac": "00:00:00:00:00:0a", "port": 1},
        }

    return gen


# ---------------------------------------------------------------------------
# Insert workloads.
# ---------------------------------------------------------------------------


def run_insert_sync(
    net: p4net.Network,
    switches: list[str],
    table_name: str,
    n_entries_per_switch: int,
    entry_generator: EntryGenerator,
) -> dict[str, Any]:
    """Insert ``n_entries_per_switch`` entries on each switch, in series.

    Returns a timing dict with ``total_wall_clock_s``, ``per_switch_s``,
    ``success_count``, ``failure_count`` and ``entries_per_second``.
    """
    success = 0
    failure = 0
    per_switch_s: dict[str, float] = {}

    t0 = time.perf_counter()
    for sw_name in switches:
        sw_t0 = time.perf_counter()
        client = net.switch(sw_name).client
        for i in range(n_entries_per_switch):
            entry = entry_generator(i)
            try:
                client.insert_table_entry(
                    table_name,
                    entry["match"],
                    entry["action_name"],
                    entry["params"],
                )
                success += 1
            except Exception as exc:
                logger.warning("sync insert failed sw=%s i=%d: %r", sw_name, i, exc)
                failure += 1
        per_switch_s[sw_name] = time.perf_counter() - sw_t0
    total = time.perf_counter() - t0

    return {
        "total_wall_clock_s": total,
        "per_switch_s": per_switch_s,
        "success_count": success,
        "failure_count": failure,
        "entries_per_second": success / total if total > 0 else 0.0,
    }


def run_insert_async(
    net: p4net.Network,
    switches: list[str],
    table_name: str,
    n_entries_per_switch: int,
    entry_generator: EntryGenerator,
) -> dict[str, Any]:
    """Insert entries across all switches concurrently via asyncio."""

    async def _insert_one_switch(sw_name: str) -> tuple[int, int]:
        success = 0
        failure = 0
        client = net.switch(sw_name).async_client
        await client.connect()
        try:
            for i in range(n_entries_per_switch):
                entry = entry_generator(i)
                try:
                    await client.insert_table_entry(
                        table_name,
                        entry["match"],
                        entry["action_name"],
                        entry["params"],
                    )
                    success += 1
                except Exception as exc:
                    logger.warning("async insert failed sw=%s i=%d: %r", sw_name, i, exc)
                    failure += 1
        finally:
            await client.disconnect()
        return success, failure

    async def _all() -> list[tuple[int, int]]:
        return await asyncio.gather(*[_insert_one_switch(s) for s in switches])

    t0 = time.perf_counter()
    results = asyncio.run(_all())
    total = time.perf_counter() - t0

    success_count = sum(r[0] for r in results)
    failure_count = sum(r[1] for r in results)
    return {
        "total_wall_clock_s": total,
        "success_count": success_count,
        "failure_count": failure_count,
        "entries_per_second": success_count / total if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Read workloads.
# ---------------------------------------------------------------------------


def run_read_sync(
    net: p4net.Network,
    switches: list[str],
    table_name: str,
) -> dict[str, Any]:
    """List all entries in ``table_name`` on each switch, in series."""
    total_entries = 0
    per_switch_s: dict[str, float] = {}

    t0 = time.perf_counter()
    for sw_name in switches:
        sw_t0 = time.perf_counter()
        entries = net.switch(sw_name).client.list_table_entries(table_name)
        total_entries += len(entries)
        per_switch_s[sw_name] = time.perf_counter() - sw_t0
    total = time.perf_counter() - t0

    return {
        "total_wall_clock_s": total,
        "per_switch_s": per_switch_s,
        "total_entries_observed": total_entries,
        "success_count": total_entries,
        "failure_count": 0,
        "entries_per_second": total_entries / total if total > 0 else 0.0,
    }


def run_read_async(
    net: p4net.Network,
    switches: list[str],
    table_name: str,
) -> dict[str, Any]:
    """List entries from each switch concurrently via asyncio."""

    async def _read_one(sw_name: str) -> int:
        client = net.switch(sw_name).async_client
        await client.connect()
        try:
            count = 0
            async for _ in client.list_table_entries(table_name):
                count += 1
            return count
        finally:
            await client.disconnect()

    async def _all() -> list[int]:
        return await asyncio.gather(*[_read_one(s) for s in switches])

    t0 = time.perf_counter()
    counts = asyncio.run(_all())
    total = time.perf_counter() - t0

    total_entries = sum(counts)
    return {
        "total_wall_clock_s": total,
        "total_entries_observed": total_entries,
        "success_count": total_entries,
        "failure_count": 0,
        "entries_per_second": total_entries / total if total > 0 else 0.0,
    }
