"""Independent external validation of the RQ1 latency metric.

The validation uses the same single-switch ``l3_lpm`` path as RQ1 and
captures probe packets at the two switch-side veth interfaces with one
``dumpcap`` process.  The common capture process gives both interfaces a
shared host clock.  Packets are matched by the probe sequence number.

For each matched probe, the script compares:

* internal latency: BMv2 ``egress_global_timestamp`` minus
  ``ingress_global_timestamp``; and
* external interval: capture time on ``s1-eth2`` minus capture time on
  ``s1-eth1``.

The external interval deliberately brackets the internal BMv2 endpoints.  It
therefore includes host-side capture boundaries and BMv2 processing after the
egress timestamp; exact equality is neither assumed nor used as a pass/fail
criterion.

This command requires root because p4net creates network namespaces and veth
pairs::

    sudo -E env PATH="$PATH" .venv/bin/python \
        -m analysis.validate_rq1_external \
        --output-dir data/validation/rq1_external_validation
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from runner.host_setup import disable_l4_offload
from topologies.single_switch import H1_IP, H1_MAC, H2_IP, H2_MAC
from topologies.single_switch import build as build_single_switch
from workloads.background_traffic import BackgroundTraffic
from workloads.latency_probe import IP_PROTO_PROBE, run_probe

REPO_ROOT = Path(__file__).resolve().parent.parent
INGRESS_IFACE = "s1-eth1"
EGRESS_IFACE = "s1-eth2"


def _bare_ip(address: str) -> str:
    return address.split("/", 1)[0]


def _require_runtime() -> None:
    if os.geteuid() != 0:
        raise PermissionError("external validation requires root")
    required = ("dumpcap", "tshark", "iperf3", "p4c", "simple_switch_grpc")
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"missing required executables: {', '.join(missing)}")


def _command_version(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return (result.stdout or result.stderr).strip().splitlines()[0]


def _start_capture(path: Path) -> tuple[subprocess.Popen[bytes], BinaryIO]:
    capture_filter = f"ip proto {IP_PROTO_PROBE}"
    argv = [
        "dumpcap",
        "-q",
        "-i",
        INGRESS_IFACE,
        "-f",
        capture_filter,
        "-i",
        EGRESS_IFACE,
        "-f",
        capture_filter,
        "-w",
        "-",
    ]
    capture_handle = path.open("wb")
    try:
        proc = subprocess.Popen(argv, stdout=capture_handle, stderr=subprocess.PIPE)
    except Exception:
        capture_handle.close()
        raise
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            capture_handle.close()
            raise RuntimeError(f"dumpcap exited before capture became ready: {stderr}")
        if path.exists() and path.stat().st_size > 0:
            time.sleep(0.25)
            return proc, capture_handle
        time.sleep(0.05)
    proc.terminate()
    proc.wait(timeout=5.0)
    capture_handle.close()
    raise TimeoutError("dumpcap did not create the capture file within 10 seconds")


def _stop_capture(proc: subprocess.Popen[bytes], capture_handle: BinaryIO) -> None:
    try:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=5.0)
        if proc.returncode not in (0, 2):
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            raise RuntimeError(f"dumpcap exited with status {proc.returncode}: {stderr}")
    finally:
        capture_handle.close()


def _payload_sequence(payload_hex: str) -> int:
    payload = bytes.fromhex(payload_hex.replace(":", ""))
    if len(payload) < 16:
        raise ValueError(f"probe payload is only {len(payload)} bytes")
    return int.from_bytes(payload[12:16], "big")


def _read_external_capture(path: Path) -> dict[str, dict[int, Decimal]]:
    display_filter = (
        f"ip.proto == {IP_PROTO_PROBE} && "
        f"ip.src == {_bare_ip(H1_IP)} && ip.dst == {_bare_ip(H2_IP)}"
    )
    argv = [
        "tshark",
        "-r",
        str(path),
        "-Y",
        display_filter,
        "-T",
        "fields",
        "-E",
        "separator=,",
        "-E",
        "quote=n",
        "-E",
        "occurrence=f",
        "-e",
        "frame.interface_id",
        "-e",
        "frame.interface_name",
        "-e",
        "frame.time_epoch",
        "-e",
        "data.data",
    ]
    result = subprocess.run(argv, check=True, capture_output=True, text=True)
    observations: dict[str, dict[int, list[Decimal]]] = {
        INGRESS_IFACE: defaultdict(list),
        EGRESS_IFACE: defaultdict(list),
    }
    id_to_iface = {"0": INGRESS_IFACE, "1": EGRESS_IFACE}
    for line_number, line in enumerate(result.stdout.splitlines(), start=1):
        fields = line.split(",")
        if len(fields) != 4:
            raise ValueError(f"unexpected tshark row {line_number}: {line!r}")
        interface_id, interface_name, epoch_text, payload_hex = fields
        iface = interface_name or id_to_iface.get(interface_id, "")
        if iface not in observations:
            raise ValueError(
                f"unexpected capture interface {interface_name!r} (id={interface_id!r})"
            )
        sequence = _payload_sequence(payload_hex)
        observations[iface][sequence].append(Decimal(epoch_text))

    duplicates = {
        iface: {seq: len(values) for seq, values in per_seq.items() if len(values) != 1}
        for iface, per_seq in observations.items()
    }
    if any(duplicates.values()):
        raise ValueError(f"duplicate capture observations make pairing ambiguous: {duplicates}")
    return {
        iface: {seq: values[0] for seq, values in per_seq.items()}
        for iface, per_seq in observations.items()
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    array = np.asarray(values, dtype=float)
    q1, median, q3, p99 = np.percentile(array, [25, 50, 75, 99])
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "p99": float(p99),
        "max": float(array.max()),
    }


def _average_ranks(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=float)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(internal: list[float], external: list[float]) -> float:
    if len(internal) < 2:
        return float("nan")
    internal_ranks = _average_ranks(internal)
    external_ranks = _average_ranks(external)
    if np.all(internal_ranks == internal_ranks[0]) or np.all(external_ranks == external_ranks[0]):
        return float("nan")
    return float(np.corrcoef(internal_ranks, external_ranks)[0, 1])


def _write_outputs(
    *,
    output_dir: Path,
    capture_path: Path,
    internal_samples: list[dict[str, Any]],
    external: dict[str, dict[int, Decimal]],
    config: dict[str, Any],
) -> dict[str, Any]:
    internal_by_sequence: dict[int, dict[str, Any]] = {}
    for sample in internal_samples:
        sequence = int(sample["sequence"])
        if sequence in internal_by_sequence:
            raise ValueError(f"duplicate internal sequence {sequence}")
        internal_by_sequence[sequence] = sample

    expected = set(range(config["sequence_start"], config["sequence_start"] + config["n_probes"]))
    internal_sequences = set(internal_by_sequence)
    ingress_sequences = set(external[INGRESS_IFACE])
    egress_sequences = set(external[EGRESS_IFACE])
    matched = sorted(internal_sequences & ingress_sequences & egress_sequences)
    if not matched:
        raise RuntimeError("no probe could be matched across internal and external measurements")

    rows: list[dict[str, Any]] = []
    for sequence in matched:
        sample = internal_by_sequence[sequence]
        ingress_epoch = external[INGRESS_IFACE][sequence]
        egress_epoch = external[EGRESS_IFACE][sequence]
        external_us = float((egress_epoch - ingress_epoch) * Decimal(1_000_000))
        internal_us = float(sample["switch_transit_us"])
        rows.append(
            {
                "sequence": sequence,
                "ingress_timestamp_us": int(sample["ingress_ts_us"]),
                "egress_timestamp_us": int(sample["egress_ts_us"]),
                "internal_us": internal_us,
                "external_ingress_epoch_s": str(ingress_epoch),
                "external_egress_epoch_s": str(egress_epoch),
                "external_us": external_us,
                "external_minus_internal_us": external_us - internal_us,
            }
        )

    csv_path = output_dir / "paired_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    internal_values = [float(row["internal_us"]) for row in rows]
    external_values = [float(row["external_us"]) for row in rows]
    differences = [float(row["external_minus_internal_us"]) for row in rows]
    summary = {
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "configuration": config,
        "software": {
            "bmv2": _command_version(["simple_switch_grpc", "--version"]),
            "p4c": _command_version(["p4c", "--version"]),
            "p4net": version("p4net"),
            "dumpcap": _command_version(["dumpcap", "--version"]),
            "tshark": _command_version(["tshark", "--version"]),
        },
        "measurement_points": {
            "internal_ingress": "BMv2 ingress_global_timestamp",
            "internal_egress": "BMv2 egress_global_timestamp at egress dequeue",
            "external_ingress": INGRESS_IFACE,
            "external_egress": EGRESS_IFACE,
            "capture_clock": "one dumpcap process for both switch-side veth interfaces",
        },
        "counts": {
            "expected": len(expected),
            "internal": len(internal_sequences),
            "external_ingress": len(ingress_sequences),
            "external_egress": len(egress_sequences),
            "matched": len(matched),
            "missing_internal": sorted(expected - internal_sequences),
            "missing_external_ingress": sorted(expected - ingress_sequences),
            "missing_external_egress": sorted(expected - egress_sequences),
        },
        "statistics_us": {
            "internal": _summary(internal_values),
            "external": _summary(external_values),
            "external_minus_internal": _summary(differences),
        },
        "association": {
            "spearman_rank_correlation": _spearman(internal_values, external_values),
        },
        "files": {
            "capture": capture_path.name,
            "paired_samples": csv_path.name,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    _require_runtime()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    capture_path = output_dir / "probe_capture.pcapng"
    log_dir = output_dir / "logs"

    h1_ip = _bare_ip(H1_IP)
    h2_ip = _bare_ip(H2_IP)
    topo = build_single_switch(REPO_ROOT / "p4" / "l3_lpm.p4")

    from p4net import Network

    net = Network(topo, log_dir=log_dir)
    net.start()
    try:
        switch = net.switch("s1")
        for ip, mac, port in ((h1_ip, H1_MAC, 1), (h2_ip, H2_MAC, 2)):
            switch.client.insert_table_entry(
                "MyIngress.ipv4_lpm",
                {"hdr.ipv4.dst_addr": f"{ip}/32"},
                "MyIngress.set_nhop",
                {"nhop_mac": mac, "port": port},
            )
        for host_name, peer_ip, peer_mac in (
            ("h1", h2_ip, H2_MAC),
            ("h2", h1_ip, H1_MAC),
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

        carrier = BackgroundTraffic(
            net=net,
            sender_host="h1",
            receiver_host="h2",
            sender_ip=h1_ip,
            receiver_ip=h2_ip,
            rate_mbps=args.carrier_mbps,
            log_dir=log_dir,
        )
        carrier.start()
        try:
            if args.warmup_seconds > 0:
                time.sleep(args.warmup_seconds)
            capture_proc, capture_handle = _start_capture(capture_path)
            try:
                internal_samples = run_probe(
                    net=net,
                    sender_host="h1",
                    receiver_host="h2",
                    sender_mac=H1_MAC,
                    receiver_mac=H2_MAC,
                    sender_ip=h1_ip,
                    receiver_ip=h2_ip,
                    probe_layer="l3",
                    n_probes=args.n_probes,
                    probe_interval_ms=args.probe_interval_ms,
                    packet_size_bytes=args.packet_size_bytes,
                    sequence_start=args.sequence_start,
                )
                time.sleep(0.5)
            finally:
                _stop_capture(capture_proc, capture_handle)
        finally:
            carrier.stop()
    finally:
        net.stop()

    config = {
        "p4_program": "l3_lpm",
        "topology": "single_switch",
        "packet_size_bytes": args.packet_size_bytes,
        "carrier_mbps": args.carrier_mbps,
        "warmup_seconds": args.warmup_seconds,
        "n_probes": args.n_probes,
        "probe_interval_ms": args.probe_interval_ms,
        "sequence_start": args.sequence_start,
    }
    return _write_outputs(
        output_dir=output_dir,
        capture_path=capture_path,
        internal_samples=internal_samples,
        external=_read_external_capture(capture_path),
        config=config,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-probes", type=int, default=1000)
    parser.add_argument("--probe-interval-ms", type=float, default=60.0)
    parser.add_argument("--packet-size-bytes", type=int, default=256)
    parser.add_argument("--carrier-mbps", type=int, default=1)
    parser.add_argument("--warmup-seconds", type=float, default=30.0)
    parser.add_argument("--sequence-start", type=int, default=0)
    args = parser.parse_args(argv)
    if args.n_probes < 1:
        parser.error("--n-probes must be >= 1")
    if args.probe_interval_ms <= 0:
        parser.error("--probe-interval-ms must be > 0")
    if args.packet_size_bytes < 50:
        parser.error("--packet-size-bytes must be >= 50 for the L3 probe")
    if args.carrier_mbps < 0:
        parser.error("--carrier-mbps must be >= 0")
    if args.warmup_seconds < 0:
        parser.error("--warmup-seconds must be >= 0")
    summary = run_validation(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
