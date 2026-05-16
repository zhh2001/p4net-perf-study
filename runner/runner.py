"""Config-driven measurement runner.

Reads a YAML campaign config, executes each configuration block in
seeded-random order, and writes one JSONL record per captured sample
(RQ1) or per repetition (RQ2) to ``data/raw/{name}_{run_id}.jsonl``.
A single ``system_info`` snapshot is written alongside, one per
runner invocation.

Config dispatch — each block must carry a ``workload_type`` field:

* ``latency_l2`` — L2 probe (Ethernet 0x88B5) against the named P4
  program loaded on a single-switch topology. One JSONL record per
  received probe; ``metric == "switch_transit_us"``.

* ``latency_l3`` — L3 probe (IPv4 proto 0xFD). Otherwise identical to
  ``latency_l2``.

* ``control_plane`` — RQ2 multi-switch control-plane workload against
  a linear-N topology. One JSONL record per repetition;
  ``metric == "control_plane_wall_clock_s"``.

If a single configuration block fails, a ``metric: "config_failure"``
record is written and the campaign continues — one bad cell does not
abort the run.

The runner is invoked as a script under sudo; ``Network.start()``
programs netns and BMv2 needs CAP_NET_ADMIN::

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

from runner.host_setup import disable_l4_offload
from runner.system_info import _verify_schema
from runner.system_info import capture as capture_system_info
from topologies.linear_n import build as build_linear_n
from topologies.single_switch import H1_IP, H1_MAC, H2_IP, H2_MAC
from topologies.single_switch import build as build_single_switch
from workloads.background_traffic import BackgroundTraffic
from workloads.control_plane_ops import (
    default_lpm_entry_generator,
    run_insert_async,
    run_insert_sync,
    run_read_async,
    run_read_sync,
)
from workloads.latency_probe import run_probe

logger = logging.getLogger("runner")

REPO_ROOT = Path(__file__).resolve().parent.parent

P4_PROGRAM_PATHS = {
    "l2_forward": "p4/l2_forward.p4",
    "l3_lpm": "p4/l3_lpm.p4",
    "l3_lpm_acl": "p4/l3_lpm_acl.p4",
    "l3_lpm_int": "p4/l3_lpm_int.p4",
}

WORKLOAD_LATENCY_L2 = "latency_l2"
WORKLOAD_LATENCY_L3 = "latency_l3"
WORKLOAD_CONTROL_PLANE = "control_plane"
KNOWN_WORKLOAD_TYPES = {
    WORKLOAD_LATENCY_L2,
    WORKLOAD_LATENCY_L3,
    WORKLOAD_CONTROL_PLANE,
}


def _utc_now_iso() -> str:
    """RFC 3339 / ISO 8601 with seconds resolution, trailing ``Z`` for UTC."""
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _bare_ip(ip_with_prefix: str) -> str:
    """``"10.0.0.1/24"`` → ``"10.0.0.1"``."""
    return ip_with_prefix.split("/", 1)[0]


def _p4_path(program_name: str) -> Path:
    if program_name not in P4_PROGRAM_PATHS:
        raise ValueError(f"unknown p4_program {program_name!r}")
    path = REPO_ROOT / P4_PROGRAM_PATHS[program_name]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


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
            wl = cfg.get("workload_type")
            if wl not in KNOWN_WORKLOAD_TYPES:
                err_cfg = dict(cfg)
                err_cfg.setdefault("workload_type", wl)
                _write_failure(
                    jsonl_fh, ValueError(f"unknown workload_type {wl!r}"), run_id, err_cfg, 0
                )
                continue
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
                    if wl in (WORKLOAD_LATENCY_L2, WORKLOAD_LATENCY_L3):
                        samples = _run_latency(cfg, rep, warmup_s)
                        _write_latency_samples(jsonl_fh, samples, run_id, cfg, rep)
                        logger.info("  → %d samples written", len(samples))
                    else:
                        result = _run_control_plane(cfg, rep)
                        _write_control_plane_result(jsonl_fh, result, run_id, cfg, rep)
                        logger.info(
                            "  → wall_clock=%.3fs success=%d failure=%d",
                            result["total_wall_clock_s"],
                            result["success_count"],
                            result["failure_count"],
                        )
                except Exception as exc:
                    logger.exception("Config failed: %s", exc)
                    _write_failure(jsonl_fh, exc, run_id, cfg, rep)
                finally:
                    if cooldown_s > 0:
                        logger.info("  cooldown %.1fs", cooldown_s)
                        time.sleep(cooldown_s)

    logger.info("Campaign complete. JSONL: %s", jsonl_path)
    return 0


# ---------------------------------------------------------------------------
# Latency workload (RQ1).
# ---------------------------------------------------------------------------


def _run_latency(
    cfg: dict[str, Any],
    repetition: int,
    warmup_s: float,
) -> list[dict[str, Any]]:
    from p4net import Network

    p4_path = _p4_path(str(cfg["p4_program"]))
    n_probes = int(cfg["n_probes"])
    probe_interval_ms = float(cfg["probe_interval_ms"])
    packet_size_bytes = int(cfg["packet_size_bytes"])
    rate_mbps = int(cfg["background_load_mbps"])
    workload_type = str(cfg["workload_type"])
    probe_layer = "l2" if workload_type == WORKLOAD_LATENCY_L2 else "l3"

    h1_ip = _bare_ip(H1_IP)
    h2_ip = _bare_ip(H2_IP)
    h1_mac = H1_MAC
    h2_mac = H2_MAC

    topo = build_single_switch(p4_path)
    net = Network(topo)
    net.start()
    try:
        sw = net.switch("s1")
        if probe_layer == "l3":
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
        else:
            # L2 forwarding: exact match on destination MAC.
            for mac, port in ((h2_mac, 2), (h1_mac, 1)):
                sw.client.insert_table_entry(
                    "MyIngress.mac_forward",
                    {"hdr.ethernet.dst_addr": mac},
                    "MyIngress.set_egress",
                    {"port": port},
                )

        # Static ARP so iperf3 / control-plane peer reachability does
        # not depend on kernel ARP resolution timing.
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
        # Disable veth L4 offload so iperf3 background traffic actually
        # flows through BMv2 (Phase B pilot's 100 Mbps load was almost
        # certainly being dropped by the receiver kernel without this).
        disable_l4_offload(net, ["h1", "h2"])

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
                sender_mac=h1_mac,
                receiver_mac=h2_mac,
                sender_ip=h1_ip if probe_layer == "l3" else None,
                receiver_ip=h2_ip if probe_layer == "l3" else None,
                probe_layer=probe_layer,
                n_probes=n_probes,
                probe_interval_ms=probe_interval_ms,
                packet_size_bytes=packet_size_bytes,
                sequence_start=repetition * n_probes,
            )
        finally:
            bg.stop()
    finally:
        net.stop()


# ---------------------------------------------------------------------------
# Control-plane workload (RQ2).
# ---------------------------------------------------------------------------


def _run_control_plane(cfg: dict[str, Any], repetition: int) -> dict[str, Any]:
    from p4net import Network

    topology_name = str(cfg.get("topology", "linear_n"))
    if topology_name != "linear_n":
        raise ValueError(
            f"only topology=linear_n is supported for control_plane, got {topology_name!r}"
        )

    p4_path = _p4_path(str(cfg["p4_program"]))
    n_switches = int(cfg["n_switches"])
    n_entries = int(cfg["n_entries_per_switch"])
    operation = str(cfg["operation"])
    mode = str(cfg["mode"])

    if operation not in ("insert", "read"):
        raise ValueError(f"operation must be 'insert' or 'read', got {operation!r}")
    if mode not in ("sync", "async"):
        raise ValueError(f"mode must be 'sync' or 'async', got {mode!r}")

    topo = build_linear_n(n_switches=n_switches, p4_program=p4_path)
    net = Network(topo)
    net.start()
    try:
        switches = [f"s{i}" for i in range(1, n_switches + 1)]
        table_name = "MyIngress.ipv4_lpm"
        gen = default_lpm_entry_generator(seed=repetition)

        if operation == "read":
            # Reads need entries to exist; populate via the matching
            # mode so we measure read against state installed by the
            # same code path.
            if mode == "sync":
                run_insert_sync(net, switches, table_name, n_entries, gen)
                return run_read_sync(net, switches, table_name)
            run_insert_async(net, switches, table_name, n_entries, gen)
            return run_read_async(net, switches, table_name)

        # Insert
        if mode == "sync":
            return run_insert_sync(net, switches, table_name, n_entries, gen)
        return run_insert_async(net, switches, table_name, n_entries, gen)
    finally:
        net.stop()


# ---------------------------------------------------------------------------
# JSONL writers.
# ---------------------------------------------------------------------------


def _latency_config_payload(cfg: dict[str, Any], repetition: int) -> dict[str, Any]:
    return {
        "p4_program": str(cfg["p4_program"]),
        "packet_size_bytes": int(cfg["packet_size_bytes"]),
        "background_load_mbps": int(cfg["background_load_mbps"]),
        "probe_layer": "l2" if cfg["workload_type"] == WORKLOAD_LATENCY_L2 else "l3",
        "repetition": repetition,
    }


def _control_plane_config_payload(cfg: dict[str, Any], repetition: int) -> dict[str, Any]:
    return {
        "p4_program": str(cfg["p4_program"]),
        "topology": str(cfg.get("topology", "linear_n")),
        "n_switches": int(cfg["n_switches"]),
        "n_entries_per_switch": int(cfg["n_entries_per_switch"]),
        "operation": str(cfg["operation"]),
        "mode": str(cfg["mode"]),
        "repetition": repetition,
    }


def _write_latency_samples(
    fh: Any,
    samples: list[dict[str, Any]],
    run_id: str,
    cfg: dict[str, Any],
    repetition: int,
) -> None:
    rq = int(cfg["rq"])
    config_payload = _latency_config_payload(cfg, repetition)
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


def _write_control_plane_result(
    fh: Any,
    result: dict[str, Any],
    run_id: str,
    cfg: dict[str, Any],
    repetition: int,
) -> None:
    rq = int(cfg["rq"])
    config_payload = _control_plane_config_payload(cfg, repetition)
    record = {
        "run_id": run_id,
        "timestamp_utc": _utc_now_iso(),
        "rq": rq,
        "config": config_payload,
        "metric": "control_plane_wall_clock_s",
        "value": float(result["total_wall_clock_s"]),
        "extras": {
            "success_count": int(result["success_count"]),
            "failure_count": int(result["failure_count"]),
            "entries_per_second": float(result["entries_per_second"]),
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
            "workload_type": cfg.get("workload_type"),
            "p4_program": cfg.get("p4_program"),
            "repetition": repetition,
        },
        "metric": "config_failure",
        "value": f"{type(exc).__name__}: {exc}",
        "extras": {},
    }
    fh.write(json.dumps(record) + "\n")
    fh.flush()


_ = sys  # kept live for SystemExit semantics on KeyboardInterrupt

if __name__ == "__main__":
    raise SystemExit(main())
