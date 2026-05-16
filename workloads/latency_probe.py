"""Switch-transit latency probe for RQ1.

The probe is a custom-payload frame whose data plane fills a 12-byte
``instrument_t`` header with its ingress and egress BMv2 global
timestamps; the receiver reads them directly off the wire and reports
``egress_ts - ingress_ts`` as the switch-transit latency in microseconds.
Two wire formats are supported, selected by the ``probe_layer`` argument
to :func:`run_probe`:

* **L2** (``probe_layer="l2"``) — for ``l2_forward.p4`` and other
  programs that key on the outer Ethernet etherType. The frame is::

      Ethernet (14)  | dst=receiver_mac, src=sender_mac, type=0x88B5
      instrument(12) | ingress_ts (48b BE) | egress_ts (48b BE)
      seq      (4)   | sequence number (32b BE)
      padding  (N)   | zero, to reach packet_size_bytes total

  Minimum size 30 bytes (Eth + instrument + seq). Note that Linux pads
  frames shorter than 60 bytes on egress, so the on-wire frame may be
  larger than requested for very small probe sizes.

* **L3** (``probe_layer="l3"``) — for the IPv4 forwarding programs
  (``l3_lpm.p4`` and variants). The frame is::

      Ethernet (14)  | dst=receiver_mac, src=sender_mac, type=0x0800
      IPv4    (20)   | proto=0xFD (probe), src=sender_ip, dst=receiver_ip
      instrument(12) | ingress_ts (48b BE) | egress_ts (48b BE)
      seq      (4)   | sequence number (32b BE)
      padding  (N)   | zero, to reach packet_size_bytes total

  Minimum size 50 bytes (Eth + IPv4 + instrument + seq).

Module surface:

* :func:`run_probe` — orchestrates one probe campaign for the runner.
  Spawns a backgrounded receiver via :meth:`RunningHost.popen` and a
  blocking sender via :meth:`RunningHost.exec`, then collects samples
  from the receiver's JSON output. Coordination via a ``ready`` file
  (set by receiver after the sniffer is armed) and a ``done`` file
  (set by orchestrator after the sender exits) so the receiver drains
  for a bounded period regardless of scapy's actual send overhead.

* ``python -m workloads.latency_probe --mode {send,receive}`` — child
  entrypoints executed inside the sender / receiver netns. Kept in
  the same module so the wire format definitions can't drift between
  sender, receiver, and orchestrator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import p4net


PROBE_LAYER_L2 = "l2"
PROBE_LAYER_L3 = "l3"
ProbeLayer = Literal["l2", "l3"]

# Wire-format constants.
ETHERTYPE_IPV4 = 0x0800
ETHERTYPE_PROBE_L2 = 0x88B5  # IEEE local-experimental, matches l2_forward.p4.
IP_PROTO_PROBE = 0xFD  # IANA "experimental, do not allocate".

# Payload layout (after Ethernet, after IPv4 if L3) — same for both layers.
INSTRUMENT_HEADER_BYTES = 12  # two big-endian 48-bit timestamps
SEQ_BYTES = 4

# Minimum on-wire frame sizes accepted by ``build_probe_packet``.
MIN_PROBE_BYTES_L2 = 14 + INSTRUMENT_HEADER_BYTES + SEQ_BYTES
MIN_PROBE_BYTES_L3 = 14 + 20 + INSTRUMENT_HEADER_BYTES + SEQ_BYTES


def _iface_for(host_name: str) -> str:
    """Default veth name convention from ``topologies.single_switch.build()``."""
    return f"{host_name}-eth0"


def _iface_mac(iface: str) -> str:
    """Read an interface's MAC from sysfs; works inside netns."""
    return Path(f"/sys/class/net/{iface}/address").read_text(encoding="utf-8").strip()


def _instrument_seq_padding(sequence: int, payload_bytes: int) -> bytes:
    """The (instrument + seq + pad) suffix shared by both wire formats."""
    instrument_bytes = b"\x00" * INSTRUMENT_HEADER_BYTES
    seq_bytes = int(sequence).to_bytes(SEQ_BYTES, "big")
    pad = b"\x00" * (payload_bytes - INSTRUMENT_HEADER_BYTES - SEQ_BYTES)
    return instrument_bytes + seq_bytes + pad


def build_l2_probe(
    *,
    sender_mac: str,
    receiver_mac: str,
    sequence: int,
    packet_size_bytes: int,
) -> Any:
    """Construct one L2 probe (Eth(0x88B5) + instrument + seq + pad)."""
    from scapy.all import Ether, Raw

    if packet_size_bytes < MIN_PROBE_BYTES_L2:
        raise ValueError(
            f"packet_size_bytes={packet_size_bytes} below L2 minimum {MIN_PROBE_BYTES_L2}"
        )
    payload_bytes = packet_size_bytes - 14
    payload = _instrument_seq_padding(sequence, payload_bytes)
    return Ether(dst=receiver_mac, src=sender_mac, type=ETHERTYPE_PROBE_L2) / Raw(load=payload)


def build_l3_probe(
    *,
    sender_ip: str,
    receiver_ip: str,
    sender_mac: str,
    receiver_mac: str,
    sequence: int,
    packet_size_bytes: int,
) -> Any:
    """Construct one L3 probe (Eth + IPv4(proto=0xFD) + instrument + seq + pad)."""
    from scapy.all import IP, Ether, Raw

    if packet_size_bytes < MIN_PROBE_BYTES_L3:
        raise ValueError(
            f"packet_size_bytes={packet_size_bytes} below L3 minimum {MIN_PROBE_BYTES_L3}"
        )
    payload_bytes = packet_size_bytes - 14 - 20
    payload = _instrument_seq_padding(sequence, payload_bytes)
    return (
        Ether(dst=receiver_mac, src=sender_mac)
        / IP(src=sender_ip, dst=receiver_ip, proto=IP_PROTO_PROBE, ttl=64)
        / Raw(load=payload)
    )


def _decode_sample(payload: bytes) -> dict[str, int] | None:
    """Parse instrument+sequence from the start of ``payload``."""
    if len(payload) < INSTRUMENT_HEADER_BYTES + SEQ_BYTES:
        return None
    ingress_ts = int.from_bytes(payload[0:6], "big")
    egress_ts = int.from_bytes(payload[6:12], "big")
    sequence = int.from_bytes(payload[12:16], "big")
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
    sender_mac: str,
    receiver_mac: str,
    sender_ip: str | None,
    receiver_ip: str | None,
    probe_layer: ProbeLayer,
    n_probes: int,
    probe_interval_ms: float,
    packet_size_bytes: int,
    sequence_start: int = 0,
) -> list[dict[str, Any]]:
    """Send ``n_probes`` probes from ``sender_host`` to ``receiver_host``.

    Returns one dict per *received* probe::

        {
            "sequence":          int,
            "ingress_ts_us":     int,
            "egress_ts_us":      int,
            "switch_transit_us": float,
        }

    Out-of-order delivery is keyed by ``sequence``. Dropped probes are
    silently absent from the returned list; callers can detect loss by
    comparing ``len(result)`` to ``n_probes``.

    ``probe_layer`` selects the wire format (see module docstring).
    L3 requires ``sender_ip`` and ``receiver_ip``; L2 ignores them.
    """
    if probe_layer not in (PROBE_LAYER_L2, PROBE_LAYER_L3):
        raise ValueError(f"probe_layer must be 'l2' or 'l3', got {probe_layer!r}")
    if probe_layer == PROBE_LAYER_L3 and (sender_ip is None or receiver_ip is None):
        raise ValueError("probe_layer='l3' requires sender_ip and receiver_ip")
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
    # legitimate scapy ``sendp`` overhead never trips it; the normal
    # path closes the receiver via the done-file well before this.
    max_capture_seconds = max(ideal_send_seconds * 2.0 + 10.0, 30.0)

    recv_argv = [
        sys.executable,
        "-m",
        "workloads.latency_probe",
        "--mode",
        "receive",
        "--iface",
        r_iface,
        "--probe-layer",
        probe_layer,
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
                "--probe-layer",
                probe_layer,
                "--sender-mac",
                sender_mac,
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
            if probe_layer == PROBE_LAYER_L3:
                send_argv += [
                    "--sender-ip",
                    str(sender_ip),
                    "--receiver-ip",
                    str(receiver_ip),
                ]
            # Timeout headroom: even with the L2-socket optimisation in
            # _send_main, scapy plus kernel send-buffer pressure adds
            # 10-50 ms per probe under high carrier load. 60 s of
            # headroom comfortably absorbs that at 1000 probes.
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
    from scapy.all import conf

    # Reuse a single L2 socket across all sends. ``scapy.sendp`` opens
    # and closes a fresh socket per call, which adds ~50 ms of overhead
    # per probe — small for the 100-probe pilot but it blows past the
    # runner's sender timeout at the 1000-probe matrix scale (60 s
    # ideal becomes ~110 s actual). Caching the socket cuts the
    # overhead to a few μs per call.
    sock = conf.L2socket(iface=args.iface)
    interval_s = float(args.probe_interval_ms) / 1000.0
    try:
        for i in range(args.n_probes):
            seq = args.sequence_start + i
            if args.probe_layer == PROBE_LAYER_L2:
                pkt = build_l2_probe(
                    sender_mac=args.sender_mac,
                    receiver_mac=args.receiver_mac,
                    sequence=seq,
                    packet_size_bytes=args.packet_size_bytes,
                )
            else:
                pkt = build_l3_probe(
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
# Child mode: receiver (runs inside the receiver netns).
# ---------------------------------------------------------------------------


def _receive_main(args: argparse.Namespace) -> int:
    from scapy.all import IP, AsyncSniffer, Ether

    samples: list[dict[str, int]] = []
    probe_layer = args.probe_layer

    def on_packet_l2(pkt: Any) -> None:
        if Ether not in pkt:
            return
        if int(pkt[Ether].type) != ETHERTYPE_PROBE_L2:
            return
        decoded = _decode_sample(bytes(pkt[Ether].payload))
        if decoded is not None:
            samples.append(decoded)

    def on_packet_l3(pkt: Any) -> None:
        if IP not in pkt:
            return
        ip = pkt[IP]
        if int(ip.proto) != IP_PROTO_PROBE:
            return
        decoded = _decode_sample(bytes(ip.payload))
        if decoded is not None:
            samples.append(decoded)

    on_packet = on_packet_l2 if probe_layer == PROBE_LAYER_L2 else on_packet_l3

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
    parser = argparse.ArgumentParser(description="L2/L3 latency probe sender/receiver child")
    parser.add_argument("--mode", choices=("send", "receive"), required=True)
    parser.add_argument("--iface", required=True)
    parser.add_argument("--probe-layer", choices=(PROBE_LAYER_L2, PROBE_LAYER_L3), required=True)
    # send-mode args
    parser.add_argument("--sender-mac")
    parser.add_argument("--receiver-mac")
    parser.add_argument("--sender-ip")
    parser.add_argument("--receiver-ip")
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
