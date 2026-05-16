"""psutil-based resource sampler for RQ4.

A context-manager wrapper around a background thread that polls
``psutil`` at a fixed cadence and produces one dict per sample. Designed
to run *concurrently* with another workload so each measurement
configuration carries its own RQ4 resource time-series alongside its
primary RQ1/RQ2/RQ3 metric.

Sample fields::

    {
        "timestamp_us":         int,                 # time.monotonic-based
        "cpu_percent_total":    float,               # whole-system, 0..100
        "cpu_percent_per_bmv2": {pid: float, ...},   # per BMv2 process
        "rss_per_bmv2_bytes":   {pid: int, ...},
        "net_io_per_iface":     {iface: {            # all in last interval
                                   "tx_bytes": int,
                                   "rx_bytes": int,
                                   "tx_pps":   float,
                                   "rx_pps":   float,
                                }, ...}
    }

Notes:

* CPU percentages need a baseline call before they produce useful
  numbers; ``__enter__`` primes ``psutil.cpu_percent`` and
  ``Process.cpu_percent`` for every target PID. The first sample after
  ``__enter__`` is therefore the first sample with meaningful CPU.
* Net-IO is reported as *deltas* over the previous sample's window,
  so units are bytes/pps per ``sample_interval_s``. The first sample
  has zeros because there is no prior interval to diff against.
* Cadence is driven by ``time.monotonic()`` offsets, not chained
  ``time.sleep(0.1)`` calls, so the sampler does not accumulate drift
  even if individual psutil calls take longer than expected.
* Threading is used in preference to asyncio because the underlying
  psutil calls are blocking (procfs reads) and the rest of the runner
  is already happy with worker threads (see ``latency_probe``).
"""

from __future__ import annotations

import logging
import threading
import time
from types import TracebackType
from typing import TYPE_CHECKING, Any

import psutil

if TYPE_CHECKING:
    import p4net

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Periodic psutil sampler. Use as ``with ResourceMonitor(...) as mon: ...``."""

    def __init__(
        self,
        net: p4net.Network | None = None,
        sample_interval_s: float = 0.1,
        target_processes: list[int] | None = None,
        target_interfaces: list[str] | None = None,
    ) -> None:
        if sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be > 0")
        self.net = net
        self.sample_interval_s = float(sample_interval_s)
        self.target_processes = list(target_processes or [])
        self.target_interfaces = list(target_interfaces or [])
        self._samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._processes: list[psutil.Process] = []
        self._prev_io: dict[str, Any] = {}
        self._prev_io_t: float = 0.0

    def __enter__(self) -> ResourceMonitor:
        self._processes = []
        for pid in self.target_processes:
            try:
                proc = psutil.Process(pid)
                # Prime: the first cpu_percent call returns 0; subsequent
                # calls return percent since previous call.
                proc.cpu_percent(interval=None)
                self._processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.warning("resource_monitor: pid %d not accessible at prime", pid)
        psutil.cpu_percent(interval=None)
        self._prev_io = self._read_net_io()
        self._prev_io_t = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._sampling_loop,
            daemon=True,
            name="resource-monitor",
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def samples(self) -> list[dict[str, Any]]:
        """Return a snapshot copy of all collected samples to date."""
        return list(self._samples)

    # ------------------------------------------------------------------

    def _read_net_io(self) -> dict[str, Any]:
        try:
            io = psutil.net_io_counters(pernic=True)
        except Exception:
            return {}
        return {iface: io[iface] for iface in self.target_interfaces if iface in io}

    def _sampling_loop(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            try:
                self._samples.append(self._take_sample())
            except Exception as exc:
                logger.exception("resource_monitor: sample failed: %s", exc)
            next_t += self.sample_interval_s
            sleep = next_t - time.monotonic()
            if sleep > 0:
                # ``Event.wait`` returns early if stop is set, so we
                # don't oversleep at teardown.
                self._stop.wait(timeout=sleep)

    def _take_sample(self) -> dict[str, Any]:
        now = time.monotonic()
        cpu_total = float(psutil.cpu_percent(interval=None))

        cpu_per_bmv2: dict[int, float] = {}
        rss_per_bmv2: dict[int, int] = {}
        for proc in self._processes:
            try:
                cpu_per_bmv2[proc.pid] = float(proc.cpu_percent(interval=None))
                rss_per_bmv2[proc.pid] = int(proc.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        cur_io = self._read_net_io()
        dt = now - self._prev_io_t
        net_io: dict[str, dict[str, float]] = {}
        for iface in self.target_interfaces:
            if iface not in cur_io or iface not in self._prev_io:
                continue
            prev = self._prev_io[iface]
            cur = cur_io[iface]
            net_io[iface] = {
                "tx_bytes": int(cur.bytes_sent - prev.bytes_sent),
                "rx_bytes": int(cur.bytes_recv - prev.bytes_recv),
                "tx_pps": (cur.packets_sent - prev.packets_sent) / dt if dt > 0 else 0.0,
                "rx_pps": (cur.packets_recv - prev.packets_recv) / dt if dt > 0 else 0.0,
            }
        self._prev_io = cur_io
        self._prev_io_t = now

        return {
            "timestamp_us": int(now * 1_000_000),
            "cpu_percent_total": cpu_total,
            "cpu_percent_per_bmv2": cpu_per_bmv2,
            "rss_per_bmv2_bytes": rss_per_bmv2,
            "net_io_per_iface": net_io,
        }
