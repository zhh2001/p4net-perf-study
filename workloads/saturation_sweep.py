"""Single-point BMv2 operating-load calibration.

Each invocation evaluates one offered UDP rate on a fresh single-switch
network. A bounded iperf3 session provides sender and receiver JSON, while
the existing instrumented probe records BMv2 ingress-to-egress-start latency.
Resource samples are retained only for the first ``measurement_seconds``
after the carrier warmup, so CPU values have an explicit per-rate window.

The workload records measurements only. The study-specific operating rule
and its sensitivity cutoffs are applied later by :mod:`analysis.aggregate`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from workloads.latency_probe import run_probe
from workloads.resource_monitor import ResourceMonitor

if TYPE_CHECKING:
    import p4net

logger = logging.getLogger(__name__)

IPERF_INTERVAL_SECONDS = 1
IPERF_SERVER_BIND_GRACE_SECONDS = 0.3
IPERF_PROCESS_TIMEOUT_HEADROOM_SECONDS = 20.0
PROCESS_TERMINATE_TIMEOUT_SECONDS = 5.0
IPERF_POST_OMIT_GUARD_INTERVALS = 1
IPERF_INTERVAL_ENDPOINT_TOLERANCE_SECONDS = 0.05
IPERF_INTERVAL_DURATION_TOLERANCE_SECONDS = 0.05
IPERF_INTERVAL_INTERNAL_TOLERANCE_SECONDS = 0.01
IPERF_CLIENT_SERVER_ENDPOINT_TOLERANCE_SECONDS = 0.05


def _iface_counter(host: Any, iface: str, counter: str) -> int:
    """Read one cumulative interface counter inside ``host``."""
    if counter not in {"rx_bytes", "rx_packets", "tx_bytes", "tx_packets"}:
        raise ValueError(f"unsupported interface counter {counter!r}")
    result = host.exec(
        ["cat", f"/sys/class/net/{iface}/statistics/{counter}"],
        capture_output=True,
        check=True,
    )
    return int(result.stdout.strip())


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile for a non-empty list."""
    if not values:
        raise ValueError("cannot compute a percentile of an empty list")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (rank - lower) * (ordered[upper] - ordered[lower])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_interval_sums(
    document: dict[str, Any],
    measurement_seconds: int,
    *,
    expected_sender: bool,
    post_omit_guard_intervals: int = IPERF_POST_OMIT_GUARD_INTERVALS,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Sum a fixed set of complete post-omit one-second intervals.

    Local iperf3 3.16 occasionally emits a malformed transition record at
    the omit boundary: ``seconds`` is approximately two while ``start`` and
    ``end`` are both approximately one. Selecting records only by
    ``omitted=false`` therefore mixes 60 seconds of bytes with roughly 61
    seconds of duration. Bin zero is uniformly treated as a transition
    guard, and both endpoints must provide exactly one complete record for
    every subsequent integer-second bin in the measurement window.
    """
    if measurement_seconds < 1:
        raise ValueError("measurement_seconds must be >= 1")
    if post_omit_guard_intervals < 0:
        raise ValueError("post_omit_guard_intervals must be >= 0")
    if measurement_seconds % IPERF_INTERVAL_SECONDS:
        raise ValueError("measurement_seconds must be a whole number of iperf intervals")

    required = measurement_seconds // IPERF_INTERVAL_SECONDS
    first_bin = post_omit_guard_intervals
    target_bins = tuple(range(first_bin, first_bin + required))
    candidates: dict[int, list[dict[str, float]]] = {
        bin_index: [] for bin_index in target_bins
    }

    for interval in document.get("intervals", []):
        item = interval.get("sum", {})
        if bool(item.get("omitted", False)):
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
            seconds = float(item["seconds"])
            bytes_count = int(item["bytes"])
            packets = int(item["packets"])
            lost_packets = int(item.get("lost_packets", 0))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        numeric = (start, end, seconds, float(bytes_count), float(packets))
        if not all(math.isfinite(value) for value in numeric):
            continue

        bin_index = round(start / IPERF_INTERVAL_SECONDS)
        if bin_index not in candidates:
            continue
        expected_start = float(bin_index * IPERF_INTERVAL_SECONDS)
        expected_end = expected_start + IPERF_INTERVAL_SECONDS
        complete = (
            abs(start - expected_start) <= IPERF_INTERVAL_ENDPOINT_TOLERANCE_SECONDS
            and abs(end - expected_end) <= IPERF_INTERVAL_ENDPOINT_TOLERANCE_SECONDS
            and abs(seconds - IPERF_INTERVAL_SECONDS)
            <= IPERF_INTERVAL_DURATION_TOLERANCE_SECONDS
            and abs(seconds - (end - start))
            <= IPERF_INTERVAL_INTERNAL_TOLERANCE_SECONDS
            and bytes_count > 0
            and packets > 0
            and 0 <= lost_packets <= packets
            and item.get("sender") is expected_sender
        )
        if complete:
            candidates[bin_index].append(
                {
                    "bin": float(bin_index),
                    "start": start,
                    "end": end,
                    "seconds": seconds,
                    "bytes": float(bytes_count),
                    "packets": float(packets),
                    "lost_packets": float(lost_packets),
                }
            )

    missing = [bin_index for bin_index, rows in candidates.items() if not rows]
    duplicate = [bin_index for bin_index, rows in candidates.items() if len(rows) > 1]
    if missing or duplicate:
        raise ValueError(
            "iperf3 JSON does not contain exactly one complete interval for each "
            f"measurement bin: missing={missing}, duplicate={duplicate}"
        )
    selected = [candidates[bin_index][0] for bin_index in target_bins]
    seconds = sum(float(item.get("seconds", 0.0)) for item in selected)
    packets = sum(int(item.get("packets", 0)) for item in selected)
    lost_packets = sum(int(item.get("lost_packets", 0)) for item in selected)
    bytes_count = sum(int(item.get("bytes", 0)) for item in selected)
    if seconds <= 0 or packets <= 0 or bytes_count <= 0:
        raise ValueError(
            "iperf3 selected intervals must have positive duration, packet count, and bytes"
        )
    if lost_packets < 0 or lost_packets > packets:
        raise ValueError("iperf3 receiver lost-packet count is outside [0, packets]")
    if abs(seconds - measurement_seconds) > 0.1:
        raise ValueError(
            f"iperf3 selected interval duration is {seconds:.6f}s; "
            f"expected approximately {measurement_seconds}s"
        )
    return (
        {
            "seconds": seconds,
            "packets": float(packets),
            "lost_packets": float(lost_packets),
            "bytes": float(bytes_count),
        },
        selected,
    )


def parse_iperf3_json(
    client_document: dict[str, Any],
    server_document: dict[str, Any],
    *,
    nominal_offered_mbps: int,
    measurement_seconds: int,
    post_omit_guard_intervals: int = IPERF_POST_OMIT_GUARD_INTERVALS,
) -> dict[str, Any]:
    """Extract calibrated sender/receiver throughput and datagram rates.

    ``iperf3`` receiver interval ``packets`` is the total datagram count used
    for its loss denominator. Successfully received datagrams are therefore
    recorded explicitly as ``packets - lost_packets``. Both quantities are
    retained to keep that interpretation auditable.
    """
    if nominal_offered_mbps <= 0:
        raise ValueError("nominal_offered_mbps must be > 0")
    for side, document in (("client", client_document), ("server", server_document)):
        if document.get("error"):
            raise ValueError(f"iperf3 {side} JSON reports an error: {document['error']}")

    sender, sender_intervals = _selected_interval_sums(
        client_document,
        measurement_seconds,
        expected_sender=True,
        post_omit_guard_intervals=post_omit_guard_intervals,
    )
    receiver, receiver_intervals = _selected_interval_sums(
        server_document,
        measurement_seconds,
        expected_sender=False,
        post_omit_guard_intervals=post_omit_guard_intervals,
    )
    if len(sender_intervals) != len(receiver_intervals):
        raise ValueError("client/server iperf3 interval counts differ")
    for sender_interval, receiver_interval in zip(
        sender_intervals, receiver_intervals, strict=True
    ):
        if int(sender_interval["bin"]) != int(receiver_interval["bin"]):
            raise ValueError("client/server iperf3 interval bins differ")
        if (
            abs(sender_interval["start"] - receiver_interval["start"])
            > IPERF_CLIENT_SERVER_ENDPOINT_TOLERANCE_SECONDS
            or abs(sender_interval["end"] - receiver_interval["end"])
            > IPERF_CLIENT_SERVER_ENDPOINT_TOLERANCE_SECONDS
        ):
            raise ValueError("client/server iperf3 interval endpoints differ")

    sender_seconds = float(sender["seconds"])
    receiver_seconds = float(receiver["seconds"])
    sender_datagrams = int(sender["packets"])
    receiver_total_datagrams = int(receiver["packets"])
    receiver_lost_datagrams = int(receiver["lost_packets"])
    receiver_datagrams = receiver_total_datagrams - receiver_lost_datagrams

    actual_offered_mbps = float(sender["bytes"]) * 8.0 / sender_seconds / 1_000_000.0
    achieved_mbps = float(receiver["bytes"]) * 8.0 / receiver_seconds / 1_000_000.0
    if actual_offered_mbps <= 0:
        raise ValueError("actual offered throughput must be positive")

    embedded_server = client_document.get("server_output_json")
    if isinstance(embedded_server, dict):
        embedded, embedded_intervals = _selected_interval_sums(
            embedded_server,
            measurement_seconds,
            expected_sender=False,
            post_omit_guard_intervals=post_omit_guard_intervals,
        )
        for key in ("bytes", "packets", "lost_packets"):
            if int(embedded[key]) != int(receiver[key]):
                raise ValueError(
                    f"independent and client-embedded server JSON disagree on {key}"
                )
        if not math.isclose(
            float(embedded["seconds"]),
            float(receiver["seconds"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ) or embedded_intervals != receiver_intervals:
            raise ValueError(
                "independent and client-embedded server JSON disagree on intervals"
            )

    start = client_document.get("start", {})
    test_start = start.get("test_start", {})
    udp_length = int(test_start.get("blksize", 0))
    if udp_length <= 0:
        raise ValueError("iperf3 JSON does not contain a positive UDP block size")

    return {
        "iperf3_version": str(start.get("version", "")),
        "iperf_udp_length_bytes": udp_length,
        "nominal_offered_mbps": float(nominal_offered_mbps),
        "actual_offered_mbps": actual_offered_mbps,
        "achieved_mbps": achieved_mbps,
        "achieved_to_actual_offered_pct": 100.0 * achieved_mbps / actual_offered_mbps,
        "achieved_to_nominal_pct": 100.0 * achieved_mbps / nominal_offered_mbps,
        "sender_seconds": sender_seconds,
        "receiver_seconds": receiver_seconds,
        "sender_datagrams": sender_datagrams,
        "receiver_total_datagrams": receiver_total_datagrams,
        "receiver_lost_datagrams": receiver_lost_datagrams,
        "receiver_datagrams": receiver_datagrams,
        "sender_pps": sender_datagrams / sender_seconds,
        "receiver_pps": receiver_datagrams / receiver_seconds,
        "iperf_receiver_loss_pct": (
            100.0 * receiver_lost_datagrams / receiver_total_datagrams
        ),
        "iperf_intervals_used": len(sender_intervals),
        "iperf_measurement_first_bin": post_omit_guard_intervals,
        "iperf_measurement_last_bin": (
            post_omit_guard_intervals + len(sender_intervals) - 1
        ),
    }


def _stop_process(proc: Any) -> None:
    """Best-effort cleanup used only on exceptional iperf3 paths."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=PROCESS_TERMINATE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=PROCESS_TERMINATE_TIMEOUT_SECONDS)


def run_calibration_point(
    *,
    net: p4net.Network,
    sender_host: str,
    receiver_host: str,
    sender_ip: str,
    receiver_ip: str,
    sender_mac: str,
    receiver_mac: str,
    rate_mbps: int,
    n_probes: int,
    probe_interval_ms: float,
    probe_packet_size_bytes: int,
    sequence_start: int,
    warmup_seconds: int,
    measurement_seconds: int,
    iperf_tail_seconds: int,
    iperf_post_omit_guard_intervals: int,
    iperf_udp_length_bytes: int,
    artifact_dir: Path,
    resource_sample_interval_s: float,
    target_processes: list[int],
    target_interfaces: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure one calibration rate and return its result and CPU samples."""
    if rate_mbps <= 0:
        raise ValueError("rate_mbps must be > 0")
    if n_probes < 1:
        raise ValueError("n_probes must be >= 1")
    if warmup_seconds < 0 or measurement_seconds < 1 or iperf_tail_seconds < 1:
        raise ValueError("invalid warmup/measurement/tail duration")
    if iperf_post_omit_guard_intervals < 0:
        raise ValueError("iperf_post_omit_guard_intervals must be >= 0")
    if iperf_udp_length_bytes < 0:
        raise ValueError("iperf_udp_length_bytes must be >= 0")

    artifact_dir.mkdir(parents=True, exist_ok=False)
    client_json_path = artifact_dir / "iperf3_client.json"
    server_json_path = artifact_dir / "iperf3_server.json"
    client_stderr_path = artifact_dir / "iperf3_client.stderr"
    server_stderr_path = artifact_dir / "iperf3_server.stderr"

    sender = net.host(sender_host)
    receiver = net.host(receiver_host)
    sender_iface = f"{sender_host}-eth0"
    receiver_iface = f"{receiver_host}-eth0"

    post_omit_guard_seconds = (
        iperf_post_omit_guard_intervals * IPERF_INTERVAL_SECONDS
    )
    iperf_test_seconds = (
        post_omit_guard_seconds + measurement_seconds + iperf_tail_seconds
    )
    server_proc = None
    client_proc = None
    resource_samples: list[dict[str, Any]] = []

    with (
        server_json_path.open("wb") as server_stdout,
        server_stderr_path.open("wb") as server_stderr,
        client_json_path.open("wb") as client_stdout,
        client_stderr_path.open("wb") as client_stderr,
    ):
        try:
            server_proc = receiver.popen(
                ["iperf3", "-s", "-1", "-J"],
                stdout=server_stdout,
                stderr=server_stderr,
            )
            time.sleep(IPERF_SERVER_BIND_GRACE_SECONDS)
            if server_proc.poll() is not None:
                raise RuntimeError(
                    f"iperf3 server exited before client launch (rc={server_proc.poll()})"
                )

            client_argv = [
                "iperf3",
                "-c",
                receiver_ip,
                "-u",
                "-b",
                f"{rate_mbps}M",
            ]
            if iperf_udp_length_bytes > 0:
                client_argv.extend(("-l", str(iperf_udp_length_bytes)))
            client_argv.extend(
                (
                    "-i",
                    str(IPERF_INTERVAL_SECONDS),
                    "-O",
                    str(warmup_seconds),
                    "-t",
                    str(iperf_test_seconds),
                    "-J",
                    "--get-server-output",
                    "--udp-counters-64bit",
                )
            )
            client_proc = sender.popen(
                client_argv,
                stdout=client_stdout,
                stderr=client_stderr,
            )
            # The fixed post-omit guard keeps probe/CPU collection aligned
            # with the complete iperf interval bins selected by the parser.
            time.sleep(float(warmup_seconds + post_omit_guard_seconds))
            if client_proc.poll() is not None:
                raise RuntimeError(
                    f"iperf3 client exited during warmup (rc={client_proc.poll()})"
                )

            iface_before = {
                "sender_tx_bytes": _iface_counter(sender, sender_iface, "tx_bytes"),
                "sender_tx_packets": _iface_counter(sender, sender_iface, "tx_packets"),
                "receiver_rx_bytes": _iface_counter(receiver, receiver_iface, "rx_bytes"),
                "receiver_rx_packets": _iface_counter(receiver, receiver_iface, "rx_packets"),
            }

            with ResourceMonitor(
                sample_interval_s=resource_sample_interval_s,
                target_processes=target_processes,
                target_interfaces=target_interfaces,
            ) as monitor:
                measurement_start = time.monotonic()
                measurement_start_us = int(measurement_start * 1_000_000)
                probe_samples = run_probe(
                    net=net,
                    sender_host=sender_host,
                    receiver_host=receiver_host,
                    sender_mac=sender_mac,
                    receiver_mac=receiver_mac,
                    sender_ip=sender_ip,
                    receiver_ip=receiver_ip,
                    probe_layer="l3",
                    n_probes=n_probes,
                    probe_interval_ms=probe_interval_ms,
                    packet_size_bytes=probe_packet_size_bytes,
                    sequence_start=sequence_start,
                )
                probe_end = time.monotonic()
            all_resource_samples = monitor.samples()

            measurement_end = measurement_start + float(measurement_seconds)
            measurement_end_us = int(measurement_end * 1_000_000)
            resource_samples = [
                sample
                for sample in all_resource_samples
                if measurement_start_us <= int(sample["timestamp_us"]) < measurement_end_us
            ]
            if not resource_samples:
                raise RuntimeError("no resource samples fall inside the measurement window")

            iface_after = {
                "sender_tx_bytes": _iface_counter(sender, sender_iface, "tx_bytes"),
                "sender_tx_packets": _iface_counter(sender, sender_iface, "tx_packets"),
                "receiver_rx_bytes": _iface_counter(receiver, receiver_iface, "rx_bytes"),
                "receiver_rx_packets": _iface_counter(receiver, receiver_iface, "rx_packets"),
            }

            if probe_end - measurement_start > measurement_seconds + iperf_tail_seconds:
                raise RuntimeError("probe campaign outlasted the bounded iperf3 carrier")

            client_rc = client_proc.wait(
                timeout=float(warmup_seconds + iperf_test_seconds)
                + IPERF_PROCESS_TIMEOUT_HEADROOM_SECONDS
            )
            server_rc = server_proc.wait(timeout=IPERF_PROCESS_TIMEOUT_HEADROOM_SECONDS)
            if client_rc != 0 or server_rc != 0:
                raise RuntimeError(
                    f"iperf3 failed: client rc={client_rc}, server rc={server_rc}"
                )
        finally:
            _stop_process(client_proc)
            _stop_process(server_proc)

    try:
        client_document = json.loads(client_json_path.read_text(encoding="utf-8"))
        server_document = json.loads(server_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read complete iperf3 JSON: {exc}") from exc

    iperf = parse_iperf3_json(
        client_document,
        server_document,
        nominal_offered_mbps=rate_mbps,
        measurement_seconds=measurement_seconds,
        post_omit_guard_intervals=iperf_post_omit_guard_intervals,
    )
    if (
        iperf_udp_length_bytes > 0
        and iperf["iperf_udp_length_bytes"] != iperf_udp_length_bytes
    ):
        raise RuntimeError(
            "iperf3 used a UDP block size different from the configured value"
        )

    expected_min = sequence_start
    expected_max = sequence_start + n_probes
    valid_by_sequence: dict[int, dict[str, Any]] = {}
    duplicate_probes = 0
    out_of_range_probes = 0
    for sample in probe_samples:
        sequence = int(sample["sequence"])
        if not expected_min <= sequence < expected_max:
            out_of_range_probes += 1
            continue
        if sequence in valid_by_sequence:
            duplicate_probes += 1
            continue
        valid_by_sequence[sequence] = sample
    valid_samples = [valid_by_sequence[key] for key in sorted(valid_by_sequence)]
    probes_received = len(valid_samples)
    if probes_received > n_probes:
        raise RuntimeError("received probe count exceeds sent count")

    latencies = [float(sample["switch_transit_us"]) for sample in valid_samples]
    if not latencies:
        raise RuntimeError("calibration point received no valid probes")
    probe_loss_pct = 100.0 * (1.0 - probes_received / n_probes)
    interface_window_seconds = probe_end - measurement_start
    interface_delta = {
        key: int(iface_after[key] - iface_before[key]) for key in iface_before
    }
    if interface_window_seconds <= 0 or any(value < 0 for value in interface_delta.values()):
        raise RuntimeError("invalid interface-counter measurement window")

    result = {
        **iperf,
        "rate_mbps": int(rate_mbps),
        "iperf_post_omit_guard_intervals": iperf_post_omit_guard_intervals,
        "probes_sent": int(n_probes),
        "probes_received": probes_received,
        "probe_loss_pct": probe_loss_pct,
        "duplicate_probes": duplicate_probes,
        "out_of_range_probes": out_of_range_probes,
        "latency_median_us": statistics.median(latencies),
        "latency_p99_us": _percentile(latencies, 99.0),
        "probe_samples": valid_samples,
        "measurement_start_monotonic_us": measurement_start_us,
        "measurement_end_monotonic_us": measurement_end_us,
        "probe_campaign_end_monotonic_us": int(probe_end * 1_000_000),
        "probe_campaign_seconds": interface_window_seconds,
        "interface_counter_window_seconds": interface_window_seconds,
        **interface_delta,
        "iperf_client_json_path": str(client_json_path),
        "iperf_server_json_path": str(server_json_path),
        "iperf_client_stderr_path": str(client_stderr_path),
        "iperf_server_stderr_path": str(server_stderr_path),
        "iperf_client_json_sha256": _sha256(client_json_path),
        "iperf_server_json_sha256": _sha256(server_json_path),
    }
    logger.info(
        "calibration rate=%d Mbps: offered=%.3f achieved=%.3f "
        "probe_loss=%.2f%% median=%.1f us p99=%.1f us",
        rate_mbps,
        result["actual_offered_mbps"],
        result["achieved_mbps"],
        result["probe_loss_pct"],
        result["latency_median_us"],
        result["latency_p99_us"],
    )
    return result, resource_samples
