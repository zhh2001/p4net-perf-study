"""Config-driven measurement runner.

Reads a YAML campaign config, executes each configuration block in
seeded-random order, and writes one JSONL record per captured sample
to ``data/raw/{name}_{run_id}.jsonl``. A single ``system_info`` snapshot
is also written alongside (one per runner invocation, not per config).

JSONL schema — one record per sample, one line per record:

    {
        "run_id": "<uuid4>",
        "timestamp_utc": "2026-MM-DDTHH:MM:SSZ",
        "rq": 1,
        "config": {
            "p4_program": "l3_lpm",
            "packet_size_bytes": 256,
            "background_load_mbps": 0,
            "repetition": 0
        },
        "metric": "switch_transit_us",
        "value": 47.3,
        "extras": {
            "sequence":      12,
            "ingress_ts_us": 1234567,
            "egress_ts_us":  1234614
        }
    }

If a single configuration block fails, a ``metric: "config_failure"``
record is written and the campaign continues with the next config —
one bad cell does not abort the run.

This module is invoked as a script; it requires root because
``p4net.Network.start()`` programs network namespaces and BMv2 needs
to bind veth pairs. Invocation pattern::

    sudo -E .venv/bin/python -m runner.runner \\
        --config runner/configs/pilot.yaml \\
        --output data/raw/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from runner.system_info import _verify_schema
from runner.system_info import capture as capture_system_info
from topologies.single_switch import H1_IP, H1_MAC, H2_IP, H2_MAC
from topologies.single_switch import build as build_single_switch
from workloads.background_traffic import BackgroundTraffic
from workloads.latency_probe import run_probe

logger = logging.getLogger("runner")

REPO_ROOT = Path(__file__).resolve().parent.parent

P4_PROGRAM_PATHS = {
    "l2_forward": "p4/l2_forward.p4",
    "l3_lpm": "p4/l3_lpm.p4",
    "l3_lpm_acl": "p4/l3_lpm_acl.p4",
    "l3_lpm_int": "p4/l3_lpm_int.p4",
}


def _utc_now_iso() -> str:
    """RFC 3339 / ISO 8601 with seconds resolution, trailing ``Z`` for UTC."""
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _bare_ip(ip_with_prefix: str) -> str:
    """``"10.0.0.1/24"`` → ``"10.0.0.1"``."""
    return ip_with_prefix.split("/", 1)[0]


def _strip_prefix(mac: str) -> str:
    """Identity for MACs but mirrors :func:`_bare_ip` for symmetry."""
    return mac


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="p4net-perf-study measurement runner")
    parser.add_argument("--config", type=Path, required=True, help="YAML campaign config")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for JSONL + system_info JSON (default: data/raw)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if os.geteuid() != 0:
        logger.warning("Not running as root — netns operations will fail.")

    campaign = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    name = str(campaign["campaign"]["name"])
    seed = int(campaign["campaign"]["seed"])
    cooldown_s = float(campaign["campaign"].get("cooldown_seconds", 30))
    warmup_s = float(campaign["campaign"].get("warmup_seconds", 30))
    configs: list[dict[str, Any]] = list(campaign["configs"])

    # Randomized seeded order so the matrix can't accumulate a
    # lexicographic bias across configs (e.g. drift in the host machine
    # warming up correlating with packet_size_bytes ascending).
    rng = random.Random(seed)
    rng.shuffle(configs)

    args.output.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    jsonl_path = args.output / f"{name}_{run_id}.jsonl"
    sysinfo_path = args.output / f"system_info_{run_id}.json"

    info = capture_system_info()
    _verify_schema(info)
    sysinfo_path.write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Run ID: %s", run_id)
    logger.info("System info: %s", sysinfo_path)
    logger.info("JSONL output: %s", jsonl_path)
    logger.info("Campaign: %s — %d configs (post-shuffle order):", name, len(configs))
    for i, c in enumerate(configs):
        logger.info("  [%d] %s", i, c)

    with open(jsonl_path, "a", encoding="utf-8") as jsonl_fh:
        for i, cfg in enumerate(configs):
            reps = int(cfg.get("repetitions", 1))
            for rep in range(reps):
                logger.info(
                    "=== config %d/%d (rep %d/%d): %s ===",
                    i + 1,
                    len(configs),
                    rep + 1,
                    reps,
                    cfg,
                )
                try:
                    samples = _run_one(cfg, rep, warmup_s)
                    _write_samples(jsonl_fh, samples, run_id, cfg, rep)
                    logger.info("  → %d samples written", len(samples))
                except Exception as exc:
                    logger.exception("Config failed: %s", exc)
                    _write_failure(jsonl_fh, exc, run_id, cfg, rep)
                finally:
                    if cooldown_s > 0:
                        logger.info("  cooldown %.1fs", cooldown_s)
                        time.sleep(cooldown_s)

    logger.info("Campaign complete. JSONL: %s", jsonl_path)
    return 0


def _run_one(cfg: dict[str, Any], repetition: int, warmup_s: float) -> list[dict[str, Any]]:
    """Execute one (config, repetition) tuple end-to-end.

    Returns the list of samples produced by ``run_probe``. Raises if
    any stage fails; the caller logs and records a ``config_failure``.
    """
    from p4net import Network

    p4_program_name = str(cfg["p4_program"])
    if p4_program_name not in P4_PROGRAM_PATHS:
        raise ValueError(f"unknown p4_program {p4_program_name!r}")
    p4_path = REPO_ROOT / P4_PROGRAM_PATHS[p4_program_name]
    if not p4_path.is_file():
        raise FileNotFoundError(p4_path)

    n_probes = int(cfg["n_probes"])
    probe_interval_ms = float(cfg["probe_interval_ms"])
    packet_size_bytes = int(cfg["packet_size_bytes"])
    rate_mbps = int(cfg["background_load_mbps"])

    h1_ip = _bare_ip(H1_IP)
    h2_ip = _bare_ip(H2_IP)
    h1_mac = _strip_prefix(H1_MAC)
    h2_mac = _strip_prefix(H2_MAC)

    topo = build_single_switch(p4_path)
    net = Network(topo)
    net.start()
    try:
        sw = net.switch("s1")
        for ip, mac, port in (
            (h1_ip, h1_mac, 1),
            (h2_ip, h2_mac, 2),
        ):
            sw.client.insert_table_entry(
                "MyIngress.ipv4_lpm",
                {"hdr.ipv4.dst_addr": f"{ip}/32"},
                "MyIngress.set_nhop",
                {"nhop_mac": mac, "port": port},
            )
        # Static ARP so background iperf3 UDP traffic flows without
        # depending on resolution timing.
        for host_name, peer_ip, peer_mac in (
            ("h1", h2_ip, h2_mac),
            ("h2", h1_ip, h1_mac),
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

        bg = BackgroundTraffic(
            net=net,
            sender_host="h1",
            receiver_host="h2",
            sender_ip=h1_ip,
            receiver_ip=h2_ip,
            rate_mbps=rate_mbps,
        )
        bg.start()
        try:
            if warmup_s > 0:
                logger.info("  warmup %.1fs", warmup_s)
                time.sleep(warmup_s)
            return run_probe(
                net=net,
                sender_host="h1",
                receiver_host="h2",
                sender_ip=h1_ip,
                receiver_ip=h2_ip,
                receiver_mac=h2_mac,
                n_probes=n_probes,
                probe_interval_ms=probe_interval_ms,
                packet_size_bytes=packet_size_bytes,
                sequence_start=repetition * n_probes,
            )
        finally:
            bg.stop()
    finally:
        net.stop()


def _write_samples(
    fh: Any,
    samples: list[dict[str, Any]],
    run_id: str,
    cfg: dict[str, Any],
    repetition: int,
) -> None:
    rq = int(cfg["rq"])
    config_payload = {
        "p4_program": str(cfg["p4_program"]),
        "packet_size_bytes": int(cfg["packet_size_bytes"]),
        "background_load_mbps": int(cfg["background_load_mbps"]),
        "repetition": repetition,
    }
    for s in samples:
        record = {
            "run_id": run_id,
            "timestamp_utc": _utc_now_iso(),
            "rq": rq,
            "config": config_payload,
            "metric": "switch_transit_us",
            "value": float(s["switch_transit_us"]),
            "extras": {
                "sequence": int(s["sequence"]),
                "ingress_ts_us": int(s["ingress_ts_us"]),
                "egress_ts_us": int(s["egress_ts_us"]),
            },
        }
        fh.write(json.dumps(record) + "\n")
    fh.flush()


def _write_failure(
    fh: Any,
    exc: BaseException,
    run_id: str,
    cfg: dict[str, Any],
    repetition: int,
) -> None:
    record = {
        "run_id": run_id,
        "timestamp_utc": _utc_now_iso(),
        "rq": int(cfg.get("rq", 0)),
        "config": {
            "p4_program": cfg.get("p4_program"),
            "packet_size_bytes": cfg.get("packet_size_bytes"),
            "background_load_mbps": cfg.get("background_load_mbps"),
            "repetition": repetition,
        },
        "metric": "config_failure",
        "value": f"{type(exc).__name__}: {exc}",
        "extras": {},
    }
    fh.write(json.dumps(record) + "\n")
    fh.flush()


# Keep ``sys`` import live; main() does not currently use it directly
# but the entrypoint does for ``sys.exit`` semantics on KeyboardInterrupt.
_ = sys

if __name__ == "__main__":
    raise SystemExit(main())
