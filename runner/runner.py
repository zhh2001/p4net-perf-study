"""Config-driven measurement runner.

Reads a YAML campaign config, executes each configuration block in
seeded-random order, and writes one JSONL record per captured sample
(RQ1, RQ3) or per repetition (RQ2) to ``data/raw/{name}_{run_id}.jsonl``.
A single ``system_info`` snapshot is written alongside, one per
runner invocation.

Config dispatch — each block must carry a ``workload_type`` field:

* ``latency_l2`` / ``latency_l3`` — RQ1 single-switch latency probes.
  L2 path uses Ethernet etherType 0x88B5; L3 path uses IPv4 protocol
  0xFD. One JSONL record per received probe; ``metric == "switch_transit_us"``.

* ``control_plane`` — RQ2 multi-switch control-plane workload against
  a linear-N topology. One JSONL record per repetition;
  ``metric == "control_plane_wall_clock_s"``.

* ``saturation_sweep`` — diagnostic (pre-RQ1 calibration). Sweeps
  background load rates and reports per-rate probe loss + iperf3 ratio.

* ``resource_only`` — RQ4 direct CPU/RSS/throughput sampling under
  background load, no latency probe alongside.

Unified warmup policy (Phase E onward)
--------------------------------------

Phase D revealed that the cold-cache 0 Mbps RQ1 baseline (216 μs)
was systematically *higher* than the under-load medians (134/142 μs)
because steady background traffic warms BMv2's CPU caches, branch
predictors, and the kernel veth path between probes. To make every
configuration's measurement window comparable, **every** config —
regardless of its measurement-window background load — gets the same
fixed-rate warmup BEFORE measurement starts:

    Phase 1 (warmup, no metrics recorded):
        BackgroundTraffic.start(rate=campaign.warmup_rate_mbps)
        sleep(campaign.warmup_seconds)
        BackgroundTraffic.stop()
    Phase 2 (measurement, metrics recorded):
        BackgroundTraffic.start(rate=cfg.background_load_mbps)  # 0 = no-op
        ResourceMonitor.__enter__()
        <primary workload>
        ResourceMonitor.__exit__()
        BackgroundTraffic.stop()

The warmup rate defaults to 1 Mbps — high enough to keep caches and
the veth path active, low enough not to consume measurable BMv2 CPU.
ResourceMonitor samples only the measurement phase so the warmup
period doesn't pollute the time-series.

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
from workloads.int_collector import run_collection as run_int_collection
from workloads.latency_probe import run_probe
from workloads.resource_monitor import ResourceMonitor
from workloads.saturation_sweep import find_sustainable_load

RESOURCE_SAMPLE_INTERVAL_S = 0.1
DEFAULT_WARMUP_RATE_MBPS = 1

logger = logging.getLogger("runner")

REPO_ROOT = Path(__file__).resolve().parent.parent

P4_PROGRAM_PATHS = {
    "l2_forward": "p4/l2_forward.p4",
    "l3_lpm": "p4/l3_lpm.p4",
    "l3_lpm_acl": "p4/l3_lpm_acl.p4",
    "l3_lpm_int": "p4/l3_lpm_int.p4",
    "l3_lpm_int_chain": "p4/l3_lpm_int_chain.p4",
}

WORKLOAD_LATENCY_L2 = "latency_l2"
WORKLOAD_LATENCY_L3 = "latency_l3"
WORKLOAD_CONTROL_PLANE = "control_plane"
WORKLOAD_SATURATION_SWEEP = "saturation_sweep"
WORKLOAD_RESOURCE_ONLY = "resource_only"
WORKLOAD_INT_MULTIHOP = "int_multihop"
KNOWN_WORKLOAD_TYPES = {
    WORKLOAD_LATENCY_L2,
    WORKLOAD_LATENCY_L3,
    WORKLOAD_CONTROL_PLANE,
    WORKLOAD_SATURATION_SWEEP,
    WORKLOAD_RESOURCE_ONLY,
    WORKLOAD_INT_MULTIHOP,
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


def _collect_bmv2_pids(net: Any) -> list[int]:
    """All live BMv2 PIDs in the network (one per switch)."""
    pids: list[int] = []
    for name in net.switches:
        sw = net.switch(name)
        bmv2 = getattr(sw, "bmv2", None)
        pid = getattr(bmv2, "pid", None) if bmv2 is not None else None
        if pid is not None:
            pids.append(int(pid))
    return pids


def _collect_switch_ifaces(topo: Any) -> list[str]:
    """Switch-side veth names from the topology (visible in root netns)."""
    switch_nodes = set(topo.switches.keys())
    ifaces: list[str] = []
    for link in topo.links:
        for endpoint in (link.a, link.b):
            if endpoint.node in switch_nodes:
                ifaces.append(endpoint.iface_name)
    return ifaces


def do_unified_warmup(
    net: Any,
    sender_host: str,
    receiver_host: str,
    sender_ip: str,
    receiver_ip: str,
    warmup_seconds: float,
    warmup_rate_mbps: int,
) -> None:
    """Run a fixed-rate background traffic burn-in before measurement.

    Idempotent and side-effect-free with respect to metrics: nothing is
    recorded during the warmup window. Returns once the warmup
    background traffic has been started, slept through, and stopped.
    No-op if ``warmup_seconds <= 0`` or ``warmup_rate_mbps <= 0``.
    """
    if warmup_seconds <= 0 or warmup_rate_mbps <= 0:
        return
    logger.info("  warmup %.1fs at %d Mbps", warmup_seconds, warmup_rate_mbps)
    warmup_bg = BackgroundTraffic(
        net=net,
        sender_host=sender_host,
        receiver_host=receiver_host,
        sender_ip=sender_ip,
        receiver_ip=receiver_ip,
        rate_mbps=warmup_rate_mbps,
    )
    warmup_bg.start()
    try:
        time.sleep(float(warmup_seconds))
    finally:
        warmup_bg.stop()


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
    warmup_rate_mbps = int(campaign["campaign"].get("warmup_rate_mbps", DEFAULT_WARMUP_RATE_MBPS))
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
                        samples, resource_samples = _run_latency(
                            cfg, rep, warmup_s, warmup_rate_mbps
                        )
                        _write_latency_samples(jsonl_fh, samples, run_id, cfg, rep)
                        _write_resource_samples(jsonl_fh, resource_samples, run_id, cfg, rep)
                        logger.info(
                            "  → %d samples written, %d resource records",
                            len(samples),
                            len(resource_samples) * 4,
                        )
                    elif wl == WORKLOAD_CONTROL_PLANE:
                        result, resource_samples = _run_control_plane(
                            cfg, rep, warmup_s, warmup_rate_mbps
                        )
                        _write_control_plane_result(jsonl_fh, result, run_id, cfg, rep)
                        _write_resource_samples(jsonl_fh, resource_samples, run_id, cfg, rep)
                        logger.info(
                            "  → wall_clock=%.3fs success=%d failure=%d, %d resource records",
                            result["total_wall_clock_s"],
                            result["success_count"],
                            result["failure_count"],
                            len(resource_samples) * 4,
                        )
                    elif wl == WORKLOAD_SATURATION_SWEEP:
                        sweep_results, resource_samples = _run_saturation_sweep(
                            cfg, warmup_s, warmup_rate_mbps
                        )
                        _write_saturation_sweep(jsonl_fh, sweep_results, run_id, cfg, rep)
                        _write_resource_samples(jsonl_fh, resource_samples, run_id, cfg, rep)
                        logger.info(
                            "  → %d sweep records, %d resource records",
                            len(sweep_results),
                            len(resource_samples) * 4,
                        )
                    elif wl == WORKLOAD_RESOURCE_ONLY:
                        _, resource_samples = _run_resource_only(cfg, warmup_s, warmup_rate_mbps)
                        _write_resource_samples(jsonl_fh, resource_samples, run_id, cfg, rep)
                        logger.info("  → %d resource records", len(resource_samples) * 4)
                    else:  # WORKLOAD_INT_MULTIHOP
                        int_samples, resource_samples = _run_int_multihop(
                            cfg, rep, warmup_s, warmup_rate_mbps
                        )
                        _write_int_samples(jsonl_fh, int_samples, run_id, cfg, rep)
                        _write_resource_samples(jsonl_fh, resource_samples, run_id, cfg, rep)
                        logger.info(
                            "  → %d INT samples, %d resource records",
                            len(int_samples),
                            len(resource_samples) * 4,
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
    warmup_rate_mbps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
            for mac, port in ((h2_mac, 2), (h1_mac, 1)):
                sw.client.insert_table_entry(
                    "MyIngress.mac_forward",
                    {"hdr.ethernet.dst_addr": mac},
                    "MyIngress.set_egress",
                    {"port": port},
                )

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
        disable_l4_offload(net, ["h1", "h2"])

        bmv2_pids = _collect_bmv2_pids(net)
        switch_ifaces = _collect_switch_ifaces(topo)

        # Phase 1: unified warmup (probes deliberately omitted)
        do_unified_warmup(
            net=net,
            sender_host="h1",
            receiver_host="h2",
            sender_ip=h1_ip,
            receiver_ip=h2_ip,
            warmup_seconds=warmup_s,
            warmup_rate_mbps=warmup_rate_mbps,
        )

        # Phase 2: measurement window at configured load
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
            with ResourceMonitor(
                sample_interval_s=RESOURCE_SAMPLE_INTERVAL_S,
                target_processes=bmv2_pids,
                target_interfaces=switch_ifaces,
            ) as mon:
                primary = run_probe(
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
            return primary, mon.samples()
        finally:
            bg.stop()
    finally:
        net.stop()


# ---------------------------------------------------------------------------
# Saturation sweep diagnostic (pre-RQ1 calibration).
# ---------------------------------------------------------------------------


def _run_saturation_sweep(
    cfg: dict[str, Any],
    warmup_s: float,
    warmup_rate_mbps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from p4net import Network

    p4_path = _p4_path(str(cfg["p4_program"]))
    rates_mbps = [int(r) for r in cfg["rates_mbps"]]
    n_probes = int(cfg.get("n_probes_per_rate", 100))
    probe_interval_ms = float(cfg.get("probe_interval_ms", 60.0))
    packet_size_bytes = int(cfg.get("packet_size_bytes", 256))
    duration_s = int(cfg.get("duration_s", 60))

    h1_ip = _bare_ip(H1_IP)
    h2_ip = _bare_ip(H2_IP)
    h1_mac = H1_MAC
    h2_mac = H2_MAC

    topo = build_single_switch(p4_path)
    net = Network(topo)
    net.start()
    try:
        sw = net.switch("s1")
        for ip, mac, port in ((h1_ip, h1_mac, 1), (h2_ip, h2_mac, 2)):
            sw.client.insert_table_entry(
                "MyIngress.ipv4_lpm",
                {"hdr.ipv4.dst_addr": f"{ip}/32"},
                "MyIngress.set_nhop",
                {"nhop_mac": mac, "port": port},
            )
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
        disable_l4_offload(net, ["h1", "h2"])
        bmv2_pids = _collect_bmv2_pids(net)
        switch_ifaces = _collect_switch_ifaces(topo)

        do_unified_warmup(
            net=net,
            sender_host="h1",
            receiver_host="h2",
            sender_ip=h1_ip,
            receiver_ip=h2_ip,
            warmup_seconds=warmup_s,
            warmup_rate_mbps=warmup_rate_mbps,
        )

        with ResourceMonitor(
            sample_interval_s=RESOURCE_SAMPLE_INTERVAL_S,
            target_processes=bmv2_pids,
            target_interfaces=switch_ifaces,
        ) as mon:
            primary = find_sustainable_load(
                net=net,
                sender_host="h1",
                receiver_host="h2",
                sender_ip=h1_ip,
                receiver_ip=h2_ip,
                sender_mac=h1_mac,
                receiver_mac=h2_mac,
                rates_mbps=rates_mbps,
                n_probes_per_rate=n_probes,
                probe_interval_ms=probe_interval_ms,
                packet_size_bytes=packet_size_bytes,
                duration_s=duration_s,
            )
        return primary, mon.samples()
    finally:
        net.stop()


# ---------------------------------------------------------------------------
# Resource-only workload (RQ4 direct).
# ---------------------------------------------------------------------------


def _run_resource_only(
    cfg: dict[str, Any],
    warmup_s: float,
    warmup_rate_mbps: int,
) -> tuple[None, list[dict[str, Any]]]:
    """Bring up the topology, optionally start background traffic, run the
    resource monitor for ``duration_s``. No latency probe, no control-plane
    operation — just resource sampling under load.
    """
    from p4net import Network

    p4_path = _p4_path(str(cfg["p4_program"]))
    topology_name = str(cfg.get("topology", "single_switch"))
    n_switches = int(cfg.get("n_switches", 1))
    rate_mbps = int(cfg.get("background_load_mbps", 0))
    duration_s = float(cfg.get("duration_s", 60))

    h1_ip = _bare_ip(H1_IP)
    h2_ip = _bare_ip(H2_IP)
    h1_mac = H1_MAC
    h2_mac = H2_MAC

    if topology_name == "linear_n":
        topo = build_linear_n(
            n_switches=n_switches,
            p4_program=p4_path,
            subnet_per_switch=False,
        )
    elif topology_name == "single_switch":
        topo = build_single_switch(p4_path)
    else:
        raise ValueError(f"unsupported topology {topology_name!r} for resource_only")

    net = Network(topo)
    net.start()
    try:
        # Only program forwarding when background traffic is needed; the
        # idle baseline case skips it so we measure BMv2 doing nothing.
        if rate_mbps > 0:
            if topology_name == "single_switch":
                for ip, mac, port in ((h1_ip, h1_mac, 1), (h2_ip, h2_mac, 2)):
                    net.switch("s1").client.insert_table_entry(
                        "MyIngress.ipv4_lpm",
                        {"hdr.ipv4.dst_addr": f"{ip}/32"},
                        "MyIngress.set_nhop",
                        {"nhop_mac": mac, "port": port},
                    )
            else:
                # Linear-N L3 forwarding for background traffic to traverse
                # the full chain: each switch points the dst toward port 2
                # if dst==h2 else port 1, with the next-hop MAC matching
                # the endpoint host (a stand-in — the action does not
                # require a real MAC for forwarding to function).
                for i in range(1, n_switches + 1):
                    sw = net.switch(f"s{i}")
                    sw.client.insert_table_entry(
                        "MyIngress.ipv4_lpm",
                        {"hdr.ipv4.dst_addr": f"{h2_ip}/32"},
                        "MyIngress.set_nhop",
                        {"nhop_mac": h2_mac, "port": 2},
                    )
                    sw.client.insert_table_entry(
                        "MyIngress.ipv4_lpm",
                        {"hdr.ipv4.dst_addr": f"{h1_ip}/32"},
                        "MyIngress.set_nhop",
                        {"nhop_mac": h1_mac, "port": 1},
                    )
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
            disable_l4_offload(net, ["h1", "h2"])

        bmv2_pids = _collect_bmv2_pids(net)
        switch_ifaces = _collect_switch_ifaces(topo)

        # Unified warmup only useful when forwarding is wired up;
        # idle-baseline configs (rate_mbps==0, no LPM/ARP) skip it.
        if rate_mbps > 0:
            do_unified_warmup(
                net=net,
                sender_host="h1",
                receiver_host="h2",
                sender_ip=h1_ip,
                receiver_ip=h2_ip,
                warmup_seconds=warmup_s,
                warmup_rate_mbps=warmup_rate_mbps,
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
            with ResourceMonitor(
                sample_interval_s=RESOURCE_SAMPLE_INTERVAL_S,
                target_processes=bmv2_pids,
                target_interfaces=switch_ifaces,
            ) as mon:
                time.sleep(duration_s)
            return None, mon.samples()
        finally:
            bg.stop()
    finally:
        net.stop()


# ---------------------------------------------------------------------------
# RQ3 multi-hop INT workload.
# ---------------------------------------------------------------------------


def _run_int_multihop(
    cfg: dict[str, Any],
    repetition: int,
    warmup_s: float,
    warmup_rate_mbps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Send INT-chain probes through a linear-N chain running
    ``l3_lpm_int_chain.p4``. Returns ``(int_samples, resource_samples)``.

    Every switch's ``MyEgress.switch_id_reg`` is pre-populated with the
    switch's small-integer ID (s1→1, s2→2, …) so the data plane can
    embed it in each shim. Forwarding entries route the L3 probe end
    to end across the full chain (h1's IP → port 1 at every switch,
    h2's IP → port 2 at every switch — the same convention used by
    :mod:`topologies.linear_n`).
    """
    from p4net import Network

    topology_name = str(cfg.get("topology", "linear_n"))
    if topology_name != "linear_n":
        raise ValueError(
            f"only topology=linear_n is supported for int_multihop, got {topology_name!r}"
        )

    p4_path = _p4_path(str(cfg["p4_program"]))
    n_switches = int(cfg["n_switches"])
    n_probes = int(cfg["n_probes"])
    probe_interval_ms = float(cfg["probe_interval_ms"])
    packet_size_bytes = int(cfg["packet_size_bytes"])
    rate_mbps = int(cfg.get("background_load_mbps", 0))

    h1_ip = _bare_ip(H1_IP)
    h2_ip = _bare_ip(H2_IP)
    h1_mac = H1_MAC
    h2_mac = H2_MAC

    topo = build_linear_n(
        n_switches=n_switches,
        p4_program=p4_path,
        subnet_per_switch=False,
    )
    net = Network(topo)
    net.start()
    try:
        switch_names = [f"s{i}" for i in range(1, n_switches + 1)]
        for idx, sw_name in enumerate(switch_names, start=1):
            sw = net.switch(sw_name)
            sw.client.write_register("MyEgress.switch_id_reg", 0, idx)
            sw.client.insert_table_entry(
                "MyIngress.ipv4_lpm",
                {"hdr.ipv4.dst_addr": f"{h1_ip}/32"},
                "MyIngress.set_nhop",
                {"nhop_mac": h1_mac, "port": 1},
            )
            sw.client.insert_table_entry(
                "MyIngress.ipv4_lpm",
                {"hdr.ipv4.dst_addr": f"{h2_ip}/32"},
                "MyIngress.set_nhop",
                {"nhop_mac": h2_mac, "port": 2},
            )
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
        disable_l4_offload(net, ["h1", "h2"])

        bmv2_pids = _collect_bmv2_pids(net)
        switch_ifaces = _collect_switch_ifaces(topo)

        do_unified_warmup(
            net=net,
            sender_host="h1",
            receiver_host="h2",
            sender_ip=h1_ip,
            receiver_ip=h2_ip,
            warmup_seconds=warmup_s,
            warmup_rate_mbps=warmup_rate_mbps,
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
            with ResourceMonitor(
                sample_interval_s=RESOURCE_SAMPLE_INTERVAL_S,
                target_processes=bmv2_pids,
                target_interfaces=switch_ifaces,
            ) as mon:
                primary = run_int_collection(
                    net=net,
                    sender_host="h1",
                    receiver_host="h2",
                    sender_mac=h1_mac,
                    receiver_mac=h2_mac,
                    sender_ip=h1_ip,
                    receiver_ip=h2_ip,
                    switch_names=switch_names,
                    n_probes=n_probes,
                    probe_interval_ms=probe_interval_ms,
                    packet_size_bytes=packet_size_bytes,
                    sequence_start=repetition * n_probes,
                )
            return primary, mon.samples()
        finally:
            bg.stop()
    finally:
        net.stop()


# ---------------------------------------------------------------------------
# Control-plane workload (RQ2).
# ---------------------------------------------------------------------------


def _run_control_plane(
    cfg: dict[str, Any],
    repetition: int,
    warmup_s: float,
    warmup_rate_mbps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        bmv2_pids = _collect_bmv2_pids(net)
        switch_ifaces = _collect_switch_ifaces(topo)

        # Control-plane workloads have no data-plane forwarding wired,
        # so an iperf3-style warmup wouldn't have anywhere to land.
        # Skip warmup; CPU caches are warmed by the gRPC bring-up itself.
        _ = warmup_s, warmup_rate_mbps

        with ResourceMonitor(
            sample_interval_s=RESOURCE_SAMPLE_INTERVAL_S,
            target_processes=bmv2_pids,
            target_interfaces=switch_ifaces,
        ) as mon:
            if operation == "read":
                if mode == "sync":
                    run_insert_sync(net, switches, table_name, n_entries, gen)
                    primary = run_read_sync(net, switches, table_name)
                else:
                    run_insert_async(net, switches, table_name, n_entries, gen)
                    primary = run_read_async(net, switches, table_name)
            else:
                if mode == "sync":
                    primary = run_insert_sync(net, switches, table_name, n_entries, gen)
                else:
                    primary = run_insert_async(net, switches, table_name, n_entries, gen)
        return primary, mon.samples()
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


def _write_saturation_sweep(
    fh: Any,
    sweep_results: list[dict[str, Any]],
    run_id: str,
    cfg: dict[str, Any],
    repetition: int,
) -> None:
    rq = int(cfg["rq"])
    for r in sweep_results:
        config_payload = {
            "p4_program": str(cfg["p4_program"]),
            "rate_mbps": int(r["rate_mbps"]),
            "n_probes": int(r["probes_sent"]),
            "duration_s": int(cfg.get("duration_s", 60)),
            "packet_size_bytes": int(cfg.get("packet_size_bytes", 256)),
            "repetition": repetition,
        }
        record = {
            "run_id": run_id,
            "timestamp_utc": _utc_now_iso(),
            "rq": rq,
            "config": config_payload,
            "metric": "saturation_probe_loss_pct",
            "value": float(r["probe_loss_pct"]),
            "extras": {
                "probes_received": int(r["probes_received"]),
                "median_us": float(r["median_us"]) if r["median_us"] == r["median_us"] else None,
                "p99_us": float(r["p99_us"]) if r["p99_us"] == r["p99_us"] else None,
                "p99_9_us": float(r["p99_9_us"]) if r["p99_9_us"] == r["p99_9_us"] else None,
                "iperf3_received_bytes_pct": (
                    float(r["iperf3_received_bytes_pct"])
                    if r["iperf3_received_bytes_pct"] == r["iperf3_received_bytes_pct"]
                    else None
                ),
                "rx_delta_bytes": int(r["rx_delta_bytes"]),
                "rx_window_seconds": float(r["rx_window_seconds"]),
            },
        }
        fh.write(json.dumps(record) + "\n")
    fh.flush()


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


def _int_config_payload(cfg: dict[str, Any], repetition: int) -> dict[str, Any]:
    return {
        "p4_program": str(cfg["p4_program"]),
        "topology": str(cfg.get("topology", "linear_n")),
        "n_switches": int(cfg["n_switches"]),
        "background_load_mbps": int(cfg.get("background_load_mbps", 0)),
        "packet_size_bytes": int(cfg["packet_size_bytes"]),
        "repetition": repetition,
    }


def _write_int_samples(
    fh: Any,
    samples: list[dict[str, Any]],
    run_id: str,
    cfg: dict[str, Any],
    repetition: int,
) -> None:
    rq = int(cfg["rq"])
    config_payload = _int_config_payload(cfg, repetition)
    for s in samples:
        record = {
            "run_id": run_id,
            "timestamp_utc": _utc_now_iso(),
            "rq": rq,
            "config": config_payload,
            "metric": "int_drift_us",
            "value": float(s["avg_drift_us"]),
            "extras": {
                "sequence": int(s["sequence"]),
                "hop_count": int(s["hop_count"]),
                "switch_ids": list(s["switch_ids"]),
                "raw_ingress_us": list(s["raw_ingress_us"]),
                "raw_egress_us": list(s["raw_egress_us"]),
                "boot_us": list(s["boot_us"]),
                "aligned_ingress_us": list(s["aligned_ingress_us"]),
                "aligned_egress_us": list(s["aligned_egress_us"]),
                "drift_us": list(s["drift_us"]),
            },
        }
        fh.write(json.dumps(record) + "\n")
    fh.flush()


def _resource_config_payload(cfg: dict[str, Any], repetition: int) -> dict[str, Any]:
    """Common per-sample config payload for RQ4 resource records."""
    return {
        "p4_program": str(cfg.get("p4_program", "")),
        "topology": str(cfg.get("topology", "single_switch")),
        "n_switches": int(cfg.get("n_switches", 1)),
        "background_load_mbps": int(cfg.get("background_load_mbps", 0)),
        "source_workload_type": str(cfg.get("workload_type", "")),
        "repetition": repetition,
    }


def _write_resource_samples(
    fh: Any,
    samples: list[dict[str, Any]],
    run_id: str,
    cfg: dict[str, Any],
    repetition: int,
) -> None:
    """Emit one JSONL record per metric per sample (4 records per sample).

    RQ4 records are tagged ``rq: 4`` regardless of the originating
    workload so analysis can pull all resource time-series uniformly.
    The original workload's ``rq`` and ``workload_type`` are preserved
    in ``config.source_workload_type`` to support cross-tagging.
    """
    base_cfg = _resource_config_payload(cfg, repetition)
    for sample_index, s in enumerate(samples):
        ts_utc = _utc_now_iso()
        cfg_with_index = {**base_cfg, "sample_index": sample_index}

        per_pid_cpu = {str(pid): float(v) for pid, v in s["cpu_percent_per_bmv2"].items()}
        per_pid_rss = {str(pid): int(v) for pid, v in s["rss_per_bmv2_bytes"].items()}
        per_iface = s["net_io_per_iface"]
        total_rx_pps = sum(float(v.get("rx_pps", 0.0)) for v in per_iface.values())

        records = [
            {
                "metric": "cpu_percent_total",
                "value": float(s["cpu_percent_total"]),
                "extras": {"timestamp_us": int(s["timestamp_us"])},
            },
            {
                "metric": "cpu_percent_per_bmv2",
                "value": float(sum(per_pid_cpu.values())),
                "extras": {
                    "timestamp_us": int(s["timestamp_us"]),
                    "per_pid": per_pid_cpu,
                },
            },
            {
                "metric": "rss_per_bmv2_bytes",
                "value": float(sum(per_pid_rss.values())),
                "extras": {
                    "timestamp_us": int(s["timestamp_us"]),
                    "per_pid": per_pid_rss,
                },
            },
            {
                "metric": "net_io_pps_per_iface",
                "value": float(total_rx_pps),
                "extras": {
                    "timestamp_us": int(s["timestamp_us"]),
                    "per_iface": per_iface,
                },
            },
        ]
        for r in records:
            r["run_id"] = run_id
            r["timestamp_utc"] = ts_utc
            r["rq"] = 4
            r["config"] = cfg_with_index
            fh.write(json.dumps(r) + "\n")
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
