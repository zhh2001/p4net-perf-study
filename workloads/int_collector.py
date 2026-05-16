"""Multi-hop INT probe collector for RQ3.

Sender emits an L3 probe with an empty INT stack and a one-byte
``int_meta.hop_count`` field set to 0. Each switch running
``p4/l3_lpm_int_chain.p4`` pushes one shim onto the front of the stack
on egress and increments ``hop_count``. The receiver decodes the
variable-length stack to recover per-hop ``(switch_id, ingress_ts,
egress_ts)`` triples and the orchestrator aligns those BMv2-local μs
timestamps to wall-clock via ``net.boot_timestamps`` so per-hop drift
is meaningful across switch processes.

Wire format (probe path only — IPv4 protocol byte is 0xFD)::

    Ethernet (14)
    IPv4 (20)
    instrument (12) — ingress + egress ts (each 48b BE)
    int_meta (1)  — hop_count
    int_stack (N * 13) — stack[0] = most recent hop;
                         shim = switch_id(1) + ingress(6) + egress(6)
    seq (4) — sequence number, BE
    pad — zero bytes to packet_size_bytes

Minimum frame size: ``14 + 20 + 12 + 1 + 4 = 51`` bytes at the sender
(stack is empty); after ``N`` hops the frame on the wire is ``13*N``
bytes longer.

Public surface mirrors :mod:`workloads.latency_probe`:

* :func:`run_collection` — orchestrator (root namespace). Spawns the
  receiver as a backgrounded ``host.popen`` and the sender as a
  blocking ``host.exec``; uses the same ``ready`` / ``done`` file
  coordination so the receiver's sniffer drains for a bounded period
  after the last probe is sent.

* ``python -m workloads.int_collector --mode {send,receive}`` — child
  entrypoints run inside the netns. Kept in the same module so the
  wire format definitions cannot drift.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import p4net


IP_PROTO_PROBE = 0xFD
INSTRUMENT_HEADER_BYTES = 12
INT_META_BYTES = 1
SHIM_BYTES = 13
SEQ_BYTES = 4
MIN_PROBE_BYTES = 14 + 20 + INSTRUMENT_HEADER_BYTES + INT_META_BYTES + SEQ_BYTES
MAX_HOPS = 8


def _iface_for(host_name: str) -> str:
    return f"{host_name}-eth0"


def build_int_probe(
    *,
    sender_ip: str,
    receiver_ip: str,
    sender_mac: str,
    receiver_mac: str,
    sequence: int,
    packet_size_bytes: int,
) -> Any:
    """Construct an INT-chain probe with hop_count=0 and an empty stack."""
    from scapy.all import IP, Ether, Raw

    if packet_size_bytes < MIN_PROBE_BYTES:
        raise ValueError(
            f"packet_size_bytes={packet_size_bytes} below INT minimum {MIN_PROBE_BYTES}"
        )
    instrument_bytes = b"\x00" * INSTRUMENT_HEADER_BYTES
    int_meta_bytes = b"\x00"  # hop_count = 0
    seq_bytes = int(sequence).to_bytes(SEQ_BYTES, "big")
    pad_len = packet_size_bytes - MIN_PROBE_BYTES
    payload = instrument_bytes + int_meta_bytes + seq_bytes + (b"\x00" * pad_len)
    return (
        Ether(dst=receiver_mac, src=sender_mac)
        / IP(src=sender_ip, dst=receiver_ip, proto=IP_PROTO_PROBE, ttl=64)
        / Raw(load=payload)
    )


def _decode_int_payload(ip_payload: bytes) -> dict[str, Any] | None:
    """Parse the stacked-INT payload. Returns ``None`` if too short or
    if the encoded ``hop_count`` would exceed our bounded stack."""
    if len(ip_payload) < INSTRUMENT_HEADER_BYTES + INT_META_BYTES + SEQ_BYTES:
        return None
    instrument_ingress = int.from_bytes(ip_payload[0:6], "big")
    instrument_egress = int.from_bytes(ip_payload[6:12], "big")
    hop_count = ip_payload[12]
    if hop_count > MAX_HOPS:
        return None
    shims_end = INSTRUMENT_HEADER_BYTES + INT_META_BYTES + hop_count * SHIM_BYTES
    if len(ip_payload) < shims_end + SEQ_BYTES:
        return None
    shims: list[dict[str, int]] = []
    for i in range(hop_count):
        offset = INSTRUMENT_HEADER_BYTES + INT_META_BYTES + i * SHIM_BYTES
        switch_id = ip_payload[offset]
        ingress_ts = int.from_bytes(ip_payload[offset + 1 : offset + 7], "big")
        egress_ts = int.from_bytes(ip_payload[offset + 7 : offset + 13], "big")
        shims.append(
            {
                "switch_id": switch_id,
                "ingress_ts_us": ingress_ts,
                "egress_ts_us": egress_ts,
            }
        )
    sequence = int.from_bytes(ip_payload[shims_end : shims_end + SEQ_BYTES], "big")
    return {
        "sequence": sequence,
        "hop_count": hop_count,
        "instrument_ingress_us": instrument_ingress,
        "instrument_egress_us": instrument_egress,
        "shims": shims,
    }


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------


def run_collection(
    net: p4net.Network,
    sender_host: str,
    receiver_host: str,
    sender_ip: str,
    receiver_ip: str,
    sender_mac: str,
    receiver_mac: str,
    switch_names: list[str],
    n_probes: int,
    probe_interval_ms: float,
    packet_size_bytes: int,
    sequence_start: int = 0,
) -> list[dict[str, Any]]:
    """Run the INT collection and return one rich sample per received probe.

    For each probe, the returned dict includes:

    * ``sequence``, ``hop_count``
    * ``switch_ids`` — chronological order (front of chain = first hop)
    * ``raw_ingress_us``, ``raw_egress_us`` — chronological order,
      BMv2-local μs since each switch's boot
    * ``boot_us`` — chronological order, the per-switch wall-clock μs
      offset from :attr:`p4net.Network.boot_timestamps`
    * ``aligned_ingress_us``, ``aligned_egress_us`` — same lists with
      ``boot_us`` added to each element so all values are on a common
      wall-clock axis
    * ``drift_us`` — wall-clock gap between adjacent hops
      ``aligned_ingress[i+1] - aligned_egress[i]``, length ``hop_count - 1``

    The receiver-side INT shim stack is emitted front-of-stack first
    (most recent hop), so this function reverses it to chronological
    order before computing the aligned axes and drift.
    """
    if n_probes < 1:
        raise ValueError("n_probes must be >= 1")
    if probe_interval_ms <= 0:
        raise ValueError("probe_interval_ms must be > 0")

    # Snapshot the wall-clock-μs offsets while the network is running,
    # then build a switch_id → boot_us map. The runner programs each
    # switch's register so its switch_id matches its index (s1=1, s2=2, …),
    # so we use the same convention here.
    boot_timestamps = net.boot_timestamps
    switch_id_to_boot_us: dict[int, int] = {}
    for idx, sw_name in enumerate(switch_names, start=1):
        if sw_name not in boot_timestamps:
            raise RuntimeError(f"switch {sw_name!r} missing from boot_timestamps")
        switch_id_to_boot_us[idx] = int(boot_timestamps[sw_name])

    sender = net.host(sender_host)
    receiver = net.host(receiver_host)
    s_iface = _iface_for(sender_host)
    r_iface = _iface_for(receiver_host)

    tmpdir = Path(tempfile.mkdtemp(prefix="int-collect-"))
    samples_path = tmpdir / "samples.json"
    ready_path = tmpdir / "ready"
    done_path = tmpdir / "done"
    err_path = tmpdir / "receiver.stderr"

    ideal_send_seconds = (n_probes - 1) * probe_interval_ms / 1000.0
    max_capture_seconds = max(ideal_send_seconds * 2.0 + 10.0, 30.0)

    recv_argv = [
        sys.executable,
        "-m",
        "workloads.int_collector",
        "--mode",
        "receive",
        "--iface",
        r_iface,
        "--max-capture-seconds",
        f"{max_capture_seconds:.3f}",
        "--drain-seconds",
        "3.0",
        "--samples-path",
        str(samples_path),
        "--ready-path",
        str(ready_path),
        "--done-path",
        str(done_path),
    ]
    send_argv = [
        sys.executable,
        "-m",
        "workloads.int_collector",
        "--mode",
        "send",
        "--iface",
        s_iface,
        "--sender-mac",
        sender_mac,
        "--receiver-mac",
        receiver_mac,
        "--sender-ip",
        sender_ip,
        "--receiver-ip",
        receiver_ip,
        "--n-probes",
        str(n_probes),
        "--probe-interval-ms",
        f"{probe_interval_ms:.6f}",
        "--packet-size-bytes",
        str(packet_size_bytes),
        "--sequence-start",
        str(sequence_start),
    ]

    with open(err_path, "wb") as err_fh:
        recv_proc = receiver.popen(
            recv_argv,
            stdout=subprocess.DEVNULL,
            stderr=err_fh,
        )
        try:
            _wait_for_ready(ready_path, timeout=10.0, proc=recv_proc)
            send_result = sender.exec(
                send_argv,
                check=False,
                capture_output=True,
                timeout=ideal_send_seconds + 60.0,
            )
            if send_result.returncode != 0:
                raise RuntimeError(
                    f"sender exited rc={send_result.returncode} stderr={send_result.stderr!r}"
                )
            done_path.write_text("done\n", encoding="utf-8")
            rc = recv_proc.wait(timeout=max_capture_seconds + 15.0)
            if rc != 0:
                err = err_path.read_text(encoding="utf-8", errors="replace")
                raise RuntimeError(f"receiver exited rc={rc}: {err}")
        finally:
            if recv_proc.poll() is None:
                recv_proc.terminate()
                try:
                    recv_proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    recv_proc.kill()
                    recv_proc.wait()

    raw_samples = json.loads(samples_path.read_text(encoding="utf-8"))
    return _enrich_samples(raw_samples, switch_id_to_boot_us)


def _enrich_samples(
    raw_samples: list[dict[str, Any]],
    switch_id_to_boot_us: dict[int, int],
) -> list[dict[str, Any]]:
    """Convert raw front-of-stack-first shim arrays to wall-clock-aligned
    chronological samples and compute per-hop drift."""
    out: list[dict[str, Any]] = []
    for raw in raw_samples:
        # Receiver hands us shims with stack[0] = most recent hop.
        # Reverse to get chronological order (first hop = s_1).
        chrono = list(reversed(raw["shims"]))
        switch_ids = [int(s["switch_id"]) for s in chrono]
        raw_ingress = [int(s["ingress_ts_us"]) for s in chrono]
        raw_egress = [int(s["egress_ts_us"]) for s in chrono]
        boot_us = [int(switch_id_to_boot_us.get(sid, 0)) for sid in switch_ids]
        aligned_ingress = [b + r for b, r in zip(boot_us, raw_ingress, strict=True)]
        aligned_egress = [b + r for b, r in zip(boot_us, raw_egress, strict=True)]
        drift_us = [aligned_ingress[i + 1] - aligned_egress[i] for i in range(len(switch_ids) - 1)]
        avg_drift = statistics.mean(drift_us) if drift_us else 0.0
        out.append(
            {
                "sequence": int(raw["sequence"]),
                "hop_count": int(raw["hop_count"]),
                "switch_ids": switch_ids,
                "raw_ingress_us": raw_ingress,
                "raw_egress_us": raw_egress,
                "boot_us": boot_us,
                "aligned_ingress_us": aligned_ingress,
                "aligned_egress_us": aligned_egress,
                "drift_us": drift_us,
                "avg_drift_us": float(avg_drift),
            }
        )
    return out


def _wait_for_ready(path: Path, timeout: float, proc: Any) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"receiver exited before becoming ready (rc={proc.poll()})")
        if path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"receiver did not signal ready within {timeout:.1f}s")


# ---------------------------------------------------------------------------
# Child: sender.
# ---------------------------------------------------------------------------


def _send_main(args: argparse.Namespace) -> int:
    from scapy.all import conf

    # Same socket-reuse rationale as workloads.latency_probe._send_main:
    # scapy.sendp opens/closes a socket on every call (~50 ms each),
    # which is fine for 100-probe pilots but blows out the runner-side
    # timeout at 1000-probe matrix scale. Cache the L2 socket.
    sock = conf.L2socket(iface=args.iface)
    interval_s = float(args.probe_interval_ms) / 1000.0
    try:
        for i in range(args.n_probes):
            seq = args.sequence_start + i
            pkt = build_int_probe(
                sender_ip=args.sender_ip,
                receiver_ip=args.receiver_ip,
                sender_mac=args.sender_mac,
                receiver_mac=args.receiver_mac,
                sequence=seq,
                packet_size_bytes=args.packet_size_bytes,
            )
            sock.send(pkt)
            if i < args.n_probes - 1:
                time.sleep(interval_s)
    finally:
        sock.close()
    return 0


# ---------------------------------------------------------------------------
# Child: receiver.
# ---------------------------------------------------------------------------


def _receive_main(args: argparse.Namespace) -> int:
    from scapy.all import IP, AsyncSniffer

    samples: list[dict[str, Any]] = []

    def on_packet(pkt: Any) -> None:
        if IP not in pkt:
            return
        ip = pkt[IP]
        if int(ip.proto) != IP_PROTO_PROBE:
            return
        decoded = _decode_int_payload(bytes(ip.payload))
        if decoded is not None:
            samples.append(decoded)

    sniffer = AsyncSniffer(iface=args.iface, prn=on_packet, store=False)
    sniffer.start()
    Path(args.ready_path).write_text("ready\n", encoding="utf-8")
    done_path = Path(args.done_path)
    drain_s = float(args.drain_seconds)
    max_s = float(args.max_capture_seconds)
    deadline = time.monotonic() + max_s
    try:
        while time.monotonic() < deadline:
            if done_path.exists():
                time.sleep(drain_s)
                break
            time.sleep(0.05)
    finally:
        sniffer.stop()
    Path(args.samples_path).write_text(json.dumps(samples), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="INT-chain sender/receiver child")
    parser.add_argument("--mode", choices=("send", "receive"), required=True)
    parser.add_argument("--iface", required=True)
    # send args
    parser.add_argument("--sender-mac")
    parser.add_argument("--receiver-mac")
    parser.add_argument("--sender-ip")
    parser.add_argument("--receiver-ip")
    parser.add_argument("--n-probes", type=int)
    parser.add_argument("--probe-interval-ms", type=float)
    parser.add_argument("--packet-size-bytes", type=int)
    parser.add_argument("--sequence-start", type=int, default=0)
    # receive args
    parser.add_argument("--max-capture-seconds", type=float)
    parser.add_argument("--drain-seconds", type=float, default=3.0)
    parser.add_argument("--samples-path")
    parser.add_argument("--ready-path")
    parser.add_argument("--done-path")
    args = parser.parse_args(argv)
    if args.mode == "send":
        return _send_main(args)
    return _receive_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
