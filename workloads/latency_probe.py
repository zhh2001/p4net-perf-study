"""L3 switch-transit latency probe for RQ1.

The probe is a layer-3 UDP-free packet identified by IPv4 protocol byte
``0xFD``. The data plane fills a 12-byte ``instrument_t`` header with
its ingress and egress BMv2 global timestamps; the receiver reads them
directly off the wire and reports ``egress_ts - ingress_ts`` as the
switch-transit latency in microseconds. No external clock alignment is
needed because both timestamps come from the same monotonic source.

Module surface:

* :func:`run_probe` — orchestrates one probe campaign for the runner.
  Spawns a backgrounded receiver via :meth:`RunningHost.popen` and a
  blocking sender via :meth:`RunningHost.exec`, then collects samples
  from the receiver's JSON output.

* ``python -m workloads.latency_probe --mode {send,receive}`` — child
  entry-points executed inside the sender/receiver netns by
  :func:`run_probe`. Kept in the same module so the wire format
  definitions can't drift between sender, receiver, and orchestrator.

Probe wire format (L3 path, this module):

    Ethernet (14)  | dst=receiver_mac, src=sender_mac, type=0x0800
    IPv4    (20)   | proto=0xFD (probe), src=sender_ip, dst=receiver_ip
    instrument(12) | ingress_ts (48b BE) | egress_ts (48b BE)
    seq      (4)   | sequence number (32b BE)
    padding  (N)   | zero, to reach packet_size_bytes total

Minimum total packet size is 50 bytes (Eth+IP+instrument+seq).
"""

from __future__ import annotations

import argparse
import json
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
SEQ_BYTES = 4
# Eth(14) + IPv4(20) + instrument(12) + seq(4)
MIN_PROBE_BYTES = 14 + 20 + INSTRUMENT_HEADER_BYTES + SEQ_BYTES


def _iface_for(host_name: str) -> str:
    """Default veth name convention from ``topologies.single_switch.build()``."""
    return f"{host_name}-eth0"


def _iface_mac(iface: str) -> str:
    """Read an interface's MAC from sysfs; works inside netns."""
    return Path(f"/sys/class/net/{iface}/address").read_text(encoding="utf-8").strip()


def build_probe_packet(
    *,
    sender_ip: str,
    receiver_ip: str,
    sender_mac: str,
    receiver_mac: str,
    sequence: int,
    packet_size_bytes: int,
) -> Any:
    """Construct one probe packet. Returns a scapy ``Ether`` / ``IP`` / ``Raw``.

    Lazily imports scapy so callers that only want the constants don't
    pay the import cost.
    """
    from scapy.all import IP, Ether, Raw

    if packet_size_bytes < MIN_PROBE_BYTES:
        raise ValueError(f"packet_size_bytes={packet_size_bytes} below minimum {MIN_PROBE_BYTES}")
    pad_len = packet_size_bytes - MIN_PROBE_BYTES
    instrument_bytes = b"\x00" * INSTRUMENT_HEADER_BYTES
    seq_bytes = int(sequence).to_bytes(SEQ_BYTES, "big")
    payload = instrument_bytes + seq_bytes + (b"\x00" * pad_len)
    return (
        Ether(dst=receiver_mac, src=sender_mac)
        / IP(src=sender_ip, dst=receiver_ip, proto=IP_PROTO_PROBE, ttl=64)
        / Raw(load=payload)
    )


def _decode_sample(ip_payload: bytes) -> dict[str, int] | None:
    """Parse an instrument+sequence payload. Returns ``None`` if too short."""
    if len(ip_payload) < INSTRUMENT_HEADER_BYTES + SEQ_BYTES:
        return None
    ingress_ts = int.from_bytes(ip_payload[0:6], "big")
    egress_ts = int.from_bytes(ip_payload[6:12], "big")
    sequence = int.from_bytes(ip_payload[12:16], "big")
    return {
        "sequence": sequence,
        "ingress_ts_us": ingress_ts,
        "egress_ts_us": egress_ts,
    }


# ---------------------------------------------------------------------------
# Orchestrator (called by the runner in the root namespace).
# ---------------------------------------------------------------------------


def run_probe(
    net: p4net.Network,
    sender_host: str,
    receiver_host: str,
    sender_ip: str,
    receiver_ip: str,
    receiver_mac: str,
    n_probes: int,
    probe_interval_ms: float,
    packet_size_bytes: int,
    sequence_start: int = 0,
) -> list[dict[str, Any]]:
    """Send ``n_probes`` probes from ``sender_host`` to ``receiver_host``.

    Returns one dict per *received* probe:

        {
            "sequence":          int,
            "ingress_ts_us":     int,
            "egress_ts_us":      int,
            "switch_transit_us": float,
        }

    Out-of-order delivery is keyed by ``sequence``. Dropped probes are
    silently absent from the returned list; callers can detect loss by
    comparing ``len(result)`` to ``n_probes``.
    """
    if n_probes < 1:
        raise ValueError("n_probes must be >= 1")
    if probe_interval_ms <= 0:
        raise ValueError("probe_interval_ms must be > 0")

    sender = net.host(sender_host)
    receiver = net.host(receiver_host)
    s_iface = _iface_for(sender_host)
    r_iface = _iface_for(receiver_host)

    tmpdir = Path(tempfile.mkdtemp(prefix="latency-probe-"))
    samples_path = tmpdir / "samples.json"
    ready_path = tmpdir / "ready"
    done_path = tmpdir / "done"
    err_path = tmpdir / "receiver.stderr"

    ideal_send_seconds = (n_probes - 1) * probe_interval_ms / 1000.0
    # Hard ceiling on how long the receiver runs if the done signal
    # never arrives (sender crash, lost done-file, etc). Generous so
    # legitimate scapy ``sendp`` overhead — a few ms per packet — never
    # trips it; the normal path closes the receiver well before this.
    max_capture_seconds = max(ideal_send_seconds * 2.0 + 10.0, 30.0)

    recv_argv = [
        sys.executable,
        "-m",
        "workloads.latency_probe",
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

    with open(err_path, "wb") as err_fh:
        recv_proc = receiver.popen(
            recv_argv,
            stdout=subprocess.DEVNULL,
            stderr=err_fh,
        )
        try:
            _wait_for_ready(ready_path, timeout=10.0, proc=recv_proc)

            send_argv = [
                sys.executable,
                "-m",
                "workloads.latency_probe",
                "--mode",
                "send",
                "--iface",
                s_iface,
                "--sender-ip",
                sender_ip,
                "--receiver-ip",
                receiver_ip,
                "--receiver-mac",
                receiver_mac,
                "--n-probes",
                str(n_probes),
                "--probe-interval-ms",
                f"{probe_interval_ms:.6f}",
                "--packet-size-bytes",
                str(packet_size_bytes),
                "--sequence-start",
                str(sequence_start),
            ]
            send_result = sender.exec(
                send_argv,
                check=False,
                capture_output=True,
                timeout=ideal_send_seconds + 30.0,
            )
            if send_result.returncode != 0:
                raise RuntimeError(
                    f"sender exited rc={send_result.returncode} stderr={send_result.stderr!r}"
                )

            # Signal end-of-burst to the receiver so it can drain in
            # peace and stop the sniffer, instead of guessing how long
            # the sender will take.
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
    out: list[dict[str, Any]] = []
    for s in raw_samples:
        # BMv2 timestamps are monotonic μs since process start; for a
        # single switch and a single run, ingress always precedes egress.
        # Persist the unsigned difference so a wraparound shows up as an
        # outlier rather than a silent negative.
        transit = int(s["egress_ts_us"]) - int(s["ingress_ts_us"])
        out.append(
            {
                "sequence": int(s["sequence"]),
                "ingress_ts_us": int(s["ingress_ts_us"]),
                "egress_ts_us": int(s["egress_ts_us"]),
                "switch_transit_us": float(transit),
            }
        )
    return out


def _wait_for_ready(path: Path, timeout: float, proc: Any) -> None:
    """Poll for the receiver's ready-file; abort if the receiver dies."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"receiver exited before becoming ready (rc={proc.poll()})")
        if path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"receiver did not signal ready within {timeout:.1f}s")


# ---------------------------------------------------------------------------
# Child mode: sender (runs inside the sender netns).
# ---------------------------------------------------------------------------


def _send_main(args: argparse.Namespace) -> int:
    from scapy.all import sendp

    sender_mac = _iface_mac(args.iface)
    interval_s = float(args.probe_interval_ms) / 1000.0
    for i in range(args.n_probes):
        seq = args.sequence_start + i
        pkt = build_probe_packet(
            sender_ip=args.sender_ip,
            receiver_ip=args.receiver_ip,
            sender_mac=sender_mac,
            receiver_mac=args.receiver_mac,
            sequence=seq,
            packet_size_bytes=args.packet_size_bytes,
        )
        sendp(pkt, iface=args.iface, verbose=False)
        if i < args.n_probes - 1:
            time.sleep(interval_s)
    return 0


# ---------------------------------------------------------------------------
# Child mode: receiver (runs inside the receiver netns).
# ---------------------------------------------------------------------------


def _receive_main(args: argparse.Namespace) -> int:
    from scapy.all import IP, AsyncSniffer

    samples: list[dict[str, int]] = []

    def on_packet(pkt: Any) -> None:
        if IP not in pkt:
            return
        ip = pkt[IP]
        if int(ip.proto) != IP_PROTO_PROBE:
            return
        decoded = _decode_sample(bytes(ip.payload))
        if decoded is not None:
            samples.append(decoded)

    sniffer = AsyncSniffer(iface=args.iface, prn=on_packet, store=False)
    sniffer.start()
    # Signal ready after the socket is bound.
    Path(args.ready_path).write_text("ready\n", encoding="utf-8")
    done_path = Path(args.done_path)
    drain_s = float(args.drain_seconds)
    max_s = float(args.max_capture_seconds)
    deadline = time.monotonic() + max_s
    try:
        # Block until the orchestrator signals "sender finished" — then
        # drain for `drain_s` so probes still in flight land — or fall
        # back to a hard ceiling so a crashed sender can't pin us.
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
    parser = argparse.ArgumentParser(description="L3 latency probe sender/receiver child")
    parser.add_argument("--mode", choices=("send", "receive"), required=True)
    parser.add_argument("--iface", required=True)
    # send-mode args
    parser.add_argument("--sender-ip")
    parser.add_argument("--receiver-ip")
    parser.add_argument("--receiver-mac")
    parser.add_argument("--n-probes", type=int)
    parser.add_argument("--probe-interval-ms", type=float)
    parser.add_argument("--packet-size-bytes", type=int)
    parser.add_argument("--sequence-start", type=int, default=0)
    # receive-mode args
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
