"""BMv2 saturation point characterization for RQ1 background-load calibration.

Runs a sequence of (rate_mbps, latency probe) pairs against an already-up
single-switch network, measuring both *probe loss* (BMv2's data-plane
ability to keep instrumented traffic moving) and *iperf3 receipt
fraction* (the host kernel's evidence that nominal background bytes
actually arrived). The output is a per-rate dict that the harness's
analysis code (or a paragraph in §5 of the paper) reduces to a single
``R_max`` value — the highest rate at which both signals stay above the
0.95 threshold.

This module is intentionally a *thin orchestrator*. It does not decide
``R_max`` itself; the decision is made by inspection of the JSONL
output. Keeping decision logic in analysis (not in the workload) lets
us re-use the same data to characterize different sustainability
thresholds (e.g. tighter 0.99 for paper figures vs. looser 0.90 for
quick sanity checks).

Methodology per rate:

1. ``BackgroundTraffic.start()`` at the target rate.
2. Brief warmup so iperf3 reaches steady state.
3. Capture ``rx_bytes`` at the receiver veth.
4. Launch the L3 latency probe inline.
5. Continue background traffic to fill the requested ``duration_s``.
6. Capture ``rx_bytes`` again and compute the actual / expected ratio.
7. ``BackgroundTraffic.stop()``; record per-rate dict and continue.

The ``rx_bytes`` window is the full BG-active interval (warmup excluded),
not just the probe duration, because the iperf3 throughput we want to
characterize is steady-state, not transient.
"""

from __future__ import annotations

import logging
import statistics
import time
from typing import TYPE_CHECKING, Any

from workloads.background_traffic import BackgroundTraffic
from workloads.latency_probe import run_probe

if TYPE_CHECKING:
    import p4net

logger = logging.getLogger(__name__)

WARMUP_SECONDS = 2.0


def _rx_bytes(host: Any, iface: str) -> int:
    """Read cumulative RX byte counter from the host's view of sysfs."""
    result = host.exec(
        ["cat", f"/sys/class/net/{iface}/statistics/rx_bytes"],
        capture_output=True,
        check=True,
    )
    return int(result.stdout.strip())


def _pct(xs: list[float], q: float) -> float:
    """Linear-interpolation percentile."""
    if not xs:
        return float("nan")
    xs_s = sorted(xs)
    k = (len(xs_s) - 1) * q / 100.0
    f = int(k)
    c = min(f + 1, len(xs_s) - 1)
    return xs_s[f] + (k - f) * (xs_s[c] - xs_s[f])


def find_sustainable_load(
    net: p4net.Network,
    sender_host: str,
    receiver_host: str,
    sender_ip: str,
    receiver_ip: str,
    sender_mac: str,
    receiver_mac: str,
    rates_mbps: list[int],
    n_probes_per_rate: int = 100,
    probe_interval_ms: float = 60.0,
    packet_size_bytes: int = 256,
    duration_s: int = 60,
) -> list[dict[str, Any]]:
    """Sweep ``rates_mbps`` and return one dict per rate. See module docstring."""
    if not rates_mbps:
        raise ValueError("rates_mbps must be non-empty")

    results: list[dict[str, Any]] = []
    receiver_iface = f"{receiver_host}-eth0"
    receiver = net.host(receiver_host)

    for rate in rates_mbps:
        logger.info("saturation sweep rate=%d Mbps", rate)
        bg = BackgroundTraffic(
            net=net,
            sender_host=sender_host,
            receiver_host=receiver_host,
            sender_ip=sender_ip,
            receiver_ip=receiver_ip,
            rate_mbps=int(rate),
        )
        bg.start()
        try:
            # Let iperf3 reach steady state before we start counting.
            time.sleep(WARMUP_SECONDS)
            rx_before = _rx_bytes(receiver, receiver_iface)
            t_before = time.monotonic()

            samples = run_probe(
                net=net,
                sender_host=sender_host,
                receiver_host=receiver_host,
                sender_mac=sender_mac,
                receiver_mac=receiver_mac,
                sender_ip=sender_ip,
                receiver_ip=receiver_ip,
                probe_layer="l3",
                n_probes=n_probes_per_rate,
                probe_interval_ms=probe_interval_ms,
                packet_size_bytes=packet_size_bytes,
            )

            elapsed = time.monotonic() - t_before
            remaining = max(0.0, float(duration_s) - WARMUP_SECONDS - elapsed)
            if remaining > 0:
                time.sleep(remaining)
            rx_after = _rx_bytes(receiver, receiver_iface)
            t_after = time.monotonic()
        finally:
            bg.stop()

        rx_delta = rx_after - rx_before
        rx_window_s = t_after - t_before
        if rate > 0 and rx_window_s > 0:
            expected_bytes = rate * 1_000_000.0 / 8.0 * rx_window_s
            iperf3_pct = 100.0 * rx_delta / expected_bytes
        else:
            iperf3_pct = float("nan")

        transits = [float(s["switch_transit_us"]) for s in samples]
        probes_received = len(samples)
        probe_loss_pct = 100.0 * (1.0 - probes_received / n_probes_per_rate)
        median_us = statistics.median(transits) if transits else float("nan")
        p99_us = _pct(transits, 99.0)
        p99_9_us = _pct(transits, 99.9)

        record = {
            "rate_mbps": int(rate),
            "probes_sent": int(n_probes_per_rate),
            "probes_received": probes_received,
            "probe_loss_pct": probe_loss_pct,
            "median_us": median_us,
            "p99_us": p99_us,
            "p99_9_us": p99_9_us,
            "iperf3_received_bytes_pct": iperf3_pct,
            "rx_delta_bytes": int(rx_delta),
            "rx_window_seconds": rx_window_s,
        }
        logger.info(
            "  rate=%d Mbps: probe_loss=%.1f%% iperf3=%.1f%% median=%.1f μs",
            rate,
            probe_loss_pct,
            iperf3_pct,
            median_us,
        )
        results.append(record)
    return results
