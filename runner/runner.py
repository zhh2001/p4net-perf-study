"""Config-driven measurement runner.

Reads a YAML campaign config, executes each configuration block in
seeded-random order, and writes one JSONL record per captured sample
(RQ1, RQ3) or per repetition (RQ2) to ``data/raw/{name}_{run_id}.jsonl``.
A single ``system_info`` snapshot is written alongside, one per
runner invocation. Every invocation also writes an artifacts-directory
manifest containing the resolved schedule, campaign config, creation time,
and SHA-256 hashes of the repository inputs used by the campaign.

Config dispatch — each block must carry a ``workload_type`` field:

* ``latency_l2`` / ``latency_l3`` — RQ1 single-switch latency probes.
  L2 path uses Ethernet etherType 0x88B5; L3 path uses IPv4 protocol
  0xFD. One JSONL record per received probe; the legacy metric identifier
  ``"switch_transit_us"`` stores the BMv2 ingress-to-egress-start interval.

* ``control_plane`` — RQ2 multi-switch control-plane workload against
  a linear-N topology. One JSONL record per repetition;
  ``metric == "control_plane_wall_clock_s"``.

* ``saturation_sweep`` — diagnostic (pre-RQ1 calibration). Sweeps
  background load rates and reports per-rate probe loss + iperf3 ratio.

* ``resource_only`` — RQ4 direct CPU/RSS/throughput sampling under
  background load, no latency probe alongside.

Continuous-carrier methodology (Phase F onward)
-----------------------------------------------

Phase E's pilot tested a warmup-then-stop pattern (run BG at 1 Mbps
for 30 s, stop it, then measure with probes only). That made the
asymmetry *worse*: 0 Mbps post-warmup hit 523 μs while 25 / 45 Mbps
under-load measurements landed at 108-109 μs. The diagnosis: BMv2's
fast-path needs *continuous* traffic to keep CPU caches, branch
predictors, and the kernel veth path warm; the 16 pps probe rate is
too sparse to sustain warm state on its own, and the gap between
warmup-end and measurement-start lets everything cool back down.

Phase F adopts continuous carrier::

    Phase 1 (warmup, no metrics recorded):
        BackgroundTraffic.start(rate=cfg.background_load_mbps)
        sleep(campaign.warmup_seconds)
    Phase 2 (measurement, metrics recorded — carrier STILL running):
        ResourceMonitor.__enter__()
        <primary workload>
        ResourceMonitor.__exit__()
    Phase 3 (teardown):
        BackgroundTraffic.stop()

A single special case — ``cold_idle_reference: true`` in the config,
typically paired with ``background_load_mbps: 0`` — skips the
carrier entirely so we preserve a cold-baseline data point for the
paper §5.2 contrast.

The campaign-level ``warmup_rate_mbps`` key from Phase E is
**deprecated** (still parsed for back-compat, ignored at runtime,
warning emitted once). The carrier rate now follows each config's
own ``background_load_mbps``.

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
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
import uuid
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
from itertools import pairwise
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
    AsyncWorkloadLoop,
    default_lpm_entry_generator,
    run_insert_async,
    run_insert_sync,
    run_read_async,
    run_read_sync,
)
from workloads.int_collector import run_collection as run_int_collection
from workloads.latency_probe import run_probe
from workloads.resource_monitor import ResourceMonitor
from workloads.saturation_sweep import run_calibration_point

RESOURCE_SAMPLE_INTERVAL_S = 0.1

logger = logging.getLogger("runner")

REPO_ROOT = Path(__file__).resolve().parent.parent

P4_PROGRAM_PATHS = {
    "l2_forward": "p4/l2_forward.p4",
    "l3_lpm": "p4/l3_lpm.p4",
    "l3_lpm_acl": "p4/l3_lpm_acl.p4",
    "l3_lpm_int": "p4/l3_lpm_int.p4",
    "l3_lpm_int_chain": "p4/l3_lpm_int_chain.p4",
}

P4_POST_INSTRUMENT_BYTES = {
    "l3_lpm_int": 13,
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_execution_cells(
    configs: list[dict[str, Any]],
    *,
    seed: int,
    shuffle_all_cells: bool,
) -> list[tuple[dict[str, Any], int]]:
    """Expand configs into ``(config, repetition)`` execution cells.

    Existing campaigns preserve their historical behaviour: configuration
    blocks are shuffled, then repetitions of each block run consecutively.
    A campaign can opt into shuffling all configuration/repetition cells so
    temporal drift is not confounded with one configuration while retaining
    a fixed seed.
    """
    rng = random.Random(seed)
    expanded_configs: list[dict[str, Any]] = []
    for original in configs:
        rates = original.get("rates_mbps")
        if original.get("workload_type") == WORKLOAD_SATURATION_SWEEP and rates is not None:
            for rate in rates:
                cfg = dict(original)
                cfg.pop("rates_mbps", None)
                cfg["rate_mbps"] = int(rate)
                cfg["background_load_mbps"] = int(rate)
                expanded_configs.append(cfg)
        else:
            expanded_configs.append(dict(original))

    if shuffle_all_cells:
        cells = [
            (cfg, repetition)
            for cfg in expanded_configs
            for repetition in range(int(cfg.get("repetitions", 1)))
        ]
        rng.shuffle(cells)
        return cells

    rng.shuffle(expanded_configs)
    return [
        (cfg, repetition)
        for cfg in expanded_configs
        for repetition in range(int(cfg.get("repetitions", 1)))
    ]


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


def make_continuous_carrier(
    net: Any,
    sender_host: str,
    receiver_host: str,
    sender_ip: str,
    receiver_ip: str,
    rate_mbps: int,
) -> Any:
    """Start a continuous background-traffic carrier at ``rate_mbps``.

    The carrier runs from before warmup through the end of measurement
    so the BMv2 fast-path stays warm under representative load
    throughout. Returns the started :class:`BackgroundTraffic` instance
    (the caller is responsible for invoking ``.stop()`` after the
    measurement window), or ``None`` if ``rate_mbps <= 0`` — the
    cold-idle reference case.
    """
    if rate_mbps <= 0:
        return None
    bg = BackgroundTraffic(
        net=net,
        sender_host=sender_host,
        receiver_host=receiver_host,
        sender_ip=sender_ip,
        receiver_ip=receiver_ip,
        rate_mbps=rate_mbps,
    )
    bg.start()
    try:
        bg.ensure_running()
    except Exception:
        bg.stop()
        raise
    return bg


def _validate_resource_samples(
    samples: list[dict[str, Any]],
    *,
    bmv2_pids: list[int],
    switch_ifaces: list[str],
    minimum_samples: int = 1,
) -> None:
    """Reject incomplete resource-monitor output before writing raw records."""
    if len(samples) < minimum_samples:
        raise RuntimeError(
            f"resource monitor produced {len(samples)} samples; "
            f"expected at least {minimum_samples}"
        )
    expected_pids = set(bmv2_pids)
    observed_ifaces: set[str] = set()
    timestamps: list[int] = []
    for index, sample in enumerate(samples):
        cpu_pids = set(sample.get("cpu_percent_per_bmv2", {}))
        rss_pids = set(sample.get("rss_per_bmv2_bytes", {}))
        if cpu_pids != expected_pids or rss_pids != expected_pids:
            raise RuntimeError(
                f"resource sample {index} process mismatch: "
                f"cpu={sorted(cpu_pids)}, rss={sorted(rss_pids)}, "
                f"expected={sorted(expected_pids)}"
            )
        observed_ifaces.update(sample.get("net_io_per_iface", {}))
        timestamps.append(int(sample["timestamp_us"]))
    expected_ifaces = set(switch_ifaces)
    if observed_ifaces != expected_ifaces:
        raise RuntimeError(
            f"resource interface mismatch: observed={sorted(observed_ifaces)}, "
            f"expected={sorted(expected_ifaces)}"
        )
    if any(later <= earlier for earlier, later in pairwise(timestamps)):
        raise RuntimeError("resource sample timestamps are not strictly increasing")


def _write_calibration_manifest(
    *,
    path: Path,
    campaign: dict[str, Any],
    config_path: Path,
    run_id: str,
    cells: list[tuple[dict[str, Any], int]],
) -> None:
    """Freeze the pre-run calibration rule, schedule, and code hashes."""
    relevant_paths = [
        config_path.resolve(),
        Path(__file__).resolve(),
        (REPO_ROOT / "workloads" / "saturation_sweep.py").resolve(),
        (REPO_ROOT / "workloads" / "resource_monitor.py").resolve(),
        (REPO_ROOT / "workloads" / "latency_probe.py").resolve(),
        (REPO_ROOT / "p4" / "l3_lpm.p4").resolve(),
    ]
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": _utc_now_iso(),
        "campaign": campaign,
        "scheduled_cells": [
            {
                "schedule_index": index,
                "rate_mbps": int(cfg["rate_mbps"]),
                "repetition": int(repetition),
            }
            for index, (cfg, repetition) in enumerate(cells)
        ],
        "sha256": {
            str(file_path.relative_to(REPO_ROOT))
            if file_path.is_relative_to(REPO_ROOT)
            else str(file_path): _sha256_file(file_path)
            for file_path in relevant_paths
        },
    }
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _campaign_source_paths(
    campaign: dict[str, Any],
    config_path: Path,
) -> list[Path]:
    """Return the repository inputs that define a non-calibration campaign.

    The system-information snapshot records installed tool versions.  This
    list complements it by pinning the local orchestration, topology,
    workload, P4, dependency-declaration, and campaign-config sources used to
    produce a run.
    """
    workload_sources = {
        WORKLOAD_LATENCY_L2: (
            "workloads/background_traffic.py",
            "workloads/latency_probe.py",
            "workloads/resource_monitor.py",
        ),
        WORKLOAD_LATENCY_L3: (
            "workloads/background_traffic.py",
            "workloads/latency_probe.py",
            "workloads/resource_monitor.py",
        ),
        WORKLOAD_CONTROL_PLANE: (
            "workloads/control_plane_ops.py",
            "workloads/resource_monitor.py",
        ),
        WORKLOAD_RESOURCE_ONLY: (
            "workloads/background_traffic.py",
            "workloads/resource_monitor.py",
        ),
        WORKLOAD_INT_MULTIHOP: (
            "workloads/background_traffic.py",
            "workloads/int_collector.py",
            "workloads/resource_monitor.py",
        ),
    }
    relative_paths = {
        "analysis/aggregate_clean.py",
        "p4/include/instrument.p4h",
        "pyproject.toml",
        "runner/host_setup.py",
        "runner/runner.py",
        "runner/system_info.py",
        "topologies/linear_n.py",
        "topologies/single_switch.py",
    }
    for cfg in campaign["configs"]:
        workload_type = str(cfg["workload_type"])
        relative_paths.update(workload_sources.get(workload_type, ()))
        program_name = str(cfg["p4_program"])
        if program_name not in P4_PROGRAM_PATHS:
            raise ValueError(f"unknown p4_program {program_name!r}")
        relative_paths.add(P4_PROGRAM_PATHS[program_name])

    paths = {config_path.resolve()}
    paths.update((REPO_ROOT / relative_path).resolve() for relative_path in relative_paths)
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"campaign provenance inputs not found: {missing}")
    return sorted(paths, key=str)


def _write_campaign_manifest(
    *,
    path: Path,
    campaign: dict[str, Any],
    config_path: Path,
    run_id: str,
    cells: list[tuple[dict[str, Any], int]],
) -> None:
    """Write the resolved schedule and source hashes for a campaign run."""
    source_paths = _campaign_source_paths(campaign, config_path)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": _utc_now_iso(),
        "config_path": (
            str(config_path.resolve().relative_to(REPO_ROOT))
            if config_path.resolve().is_relative_to(REPO_ROOT)
            else str(config_path.resolve())
        ),
        "campaign": campaign,
        "scheduled_cells": [
            {
                "schedule_index": index,
                "repetition": int(repetition),
                "config": dict(cfg),
            }
            for index, (cfg, repetition) in enumerate(cells)
        ],
        "sha256": {
            str(file_path.relative_to(REPO_ROOT))
            if file_path.is_relative_to(REPO_ROOT)
            else str(file_path): _sha256_file(file_path)
            for file_path in source_paths
        },
    }
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _portable_artifact_path(path: Path, *, relative_to: Path) -> str:
    try:
        return str(path.resolve().relative_to(relative_to.resolve()))
    except ValueError:
        return os.path.relpath(path.resolve(), start=relative_to.resolve())


def _write_campaign_completion(
    *,
    path: Path,
    run_id: str,
    raw_path: Path,
    system_info_path: Path,
    manifest_path: Path,
    runner_log_path: Path,
    scheduled_cell_count: int,
    attempted_cell_count: int,
    failure_count: int,
) -> int:
    """Atomically bind completed outputs to the immutable pre-run manifest."""
    successful_cell_count = attempted_cell_count - failure_count
    complete = (
        attempted_cell_count == scheduled_cell_count and failure_count == 0
    )
    exit_code = 0 if complete else 1
    with raw_path.open("r", encoding="utf-8") as raw_fh:
        record_count = sum(1 for line in raw_fh if line.strip())

    def file_record(file_path: Path) -> dict[str, Any]:
        return {
            "path": _portable_artifact_path(
                file_path, relative_to=path.parent
            ),
            "sha256": _sha256_file(file_path),
            "size_bytes": file_path.stat().st_size,
        }

    completion = {
        "schema_version": 1,
        "run_id": run_id,
        "completed_utc": _utc_now_iso(),
        "status": "complete" if complete else "failed",
        "exit_code": exit_code,
        "scheduled_cell_count": scheduled_cell_count,
        "attempted_cell_count": attempted_cell_count,
        "successful_cell_count": successful_cell_count,
        "failure_count": failure_count,
        "raw_record_count": record_count,
        "files": {
            "raw_jsonl": file_record(raw_path),
            "system_info": file_record(system_info_path),
            "measurement_manifest": file_record(manifest_path),
            "runner_log": file_record(runner_log_path),
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return exit_code


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
    if "warmup_rate_mbps" in campaign["campaign"]:
        logger.warning(
            "campaign.warmup_rate_mbps is deprecated since Phase F (continuous-carrier "
            "methodology) — the carrier rate now follows each config's "
            "background_load_mbps. Ignoring the campaign-level value."
        )
    configs: list[dict[str, Any]] = list(campaign["configs"])
    shuffle_all_cells = bool(campaign["campaign"].get("shuffle_all_cells", False))
    cells = _build_execution_cells(
        configs,
        seed=seed,
        shuffle_all_cells=shuffle_all_cells,
    )
    has_saturation = any(
        cfg.get("workload_type") == WORKLOAD_SATURATION_SWEEP for cfg, _ in cells
    )
    has_async_control_plane = any(
        cfg.get("workload_type") == WORKLOAD_CONTROL_PLANE
        and cfg.get("mode") == "async"
        for cfg, _ in cells
    )

    args.output.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    jsonl_path = args.output / f"{name}_{run_id}.jsonl"
    sysinfo_path = args.output / f"system_info_{run_id}.json"
    artifacts_root = args.output / f"{name}_{run_id}_artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=False)
    runner_log_path = artifacts_root / "runner.log"
    file_handler = logging.FileHandler(runner_log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    if has_saturation:
        _write_calibration_manifest(
            path=artifacts_root / "manifest.json",
            campaign=campaign,
            config_path=args.config,
            run_id=run_id,
            cells=cells,
        )
    else:
        _write_campaign_manifest(
            path=artifacts_root / "manifest.json",
            campaign=campaign,
            config_path=args.config,
            run_id=run_id,
            cells=cells,
        )

    info = capture_system_info()
    _verify_schema(info)
    sysinfo_path.write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Run ID: %s", run_id)
    logger.info("System info: %s", sysinfo_path)
    logger.info("JSONL output: %s", jsonl_path)
    logger.info("Campaign: %s — %d execution cells:", name, len(cells))
    for i, (cfg, repetition) in enumerate(cells):
        logger.info(
            "  [%d] rate=%s repetition=%d workload=%s",
            i,
            cfg.get("rate_mbps", cfg.get("background_load_mbps", "n/a")),
            repetition,
            cfg.get("workload_type"),
        )

    failure_count = 0
    attempted_cell_count = 0
    with ExitStack() as workload_stack:
        async_workload_loop = (
            workload_stack.enter_context(AsyncWorkloadLoop())
            if has_async_control_plane
            else None
        )
        jsonl_fh = workload_stack.enter_context(
            open(jsonl_path, "a", encoding="utf-8")
        )
        for i, (base_cfg, rep) in enumerate(cells):
            attempted_cell_count += 1
            cfg = dict(base_cfg)
            cfg["schedule_index"] = i
            if cfg.get("workload_type") == WORKLOAD_SATURATION_SWEEP:
                cfg["warmup_seconds"] = int(warmup_s)
            reps = int(cfg.get("repetitions", 1))
            wl = cfg.get("workload_type")
            if wl not in KNOWN_WORKLOAD_TYPES:
                err_cfg = dict(cfg)
                err_cfg.setdefault("workload_type", wl)
                _write_failure(
                    jsonl_fh, ValueError(f"unknown workload_type {wl!r}"), run_id, err_cfg, rep
                )
                failure_count += 1
                continue
            logger.info(
                "=== cell %d/%d (rep %d/%d): %s ===",
                i + 1,
                len(cells),
                rep + 1,
                reps,
                cfg,
            )
            try:
                if wl in (WORKLOAD_LATENCY_L2, WORKLOAD_LATENCY_L3):
                    samples, resource_samples = _run_latency(cfg, rep, warmup_s)
                    _write_latency_samples(jsonl_fh, samples, run_id, cfg, rep)
                    _write_resource_samples(jsonl_fh, resource_samples, run_id, cfg, rep)
                    logger.info(
                        "  → %d samples written, %d resource records",
                        len(samples),
                        len(resource_samples) * 4,
                    )
                elif wl == WORKLOAD_CONTROL_PLANE:
                    result, resource_samples = _run_control_plane(
                        cfg,
                        rep,
                        warmup_s,
                        async_workload_loop=async_workload_loop,
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
                    artifact_dir = artifacts_root / (
                        f"cell_{i:02d}_rate_{int(cfg['rate_mbps'])}_rep_{rep}"
                    )
                    result, resource_samples = _run_saturation_point(
                        cfg,
                        rep,
                        warmup_s,
                        artifact_dir,
                    )
                    _write_saturation_point(jsonl_fh, result, run_id, cfg, rep)
                    _write_resource_samples(jsonl_fh, resource_samples, run_id, cfg, rep)
                    logger.info(
                        "  → %d probe records, 1 summary, %d resource records",
                        len(result["probe_samples"]),
                        len(resource_samples) * 4,
                    )
                elif wl == WORKLOAD_RESOURCE_ONLY:
                    _, resource_samples = _run_resource_only(cfg, warmup_s)
                    _write_resource_samples(jsonl_fh, resource_samples, run_id, cfg, rep)
                    logger.info("  → %d resource records", len(resource_samples) * 4)
                else:  # WORKLOAD_INT_MULTIHOP
                    int_samples, resource_samples = _run_int_multihop(cfg, rep, warmup_s)
                    _write_int_samples(jsonl_fh, int_samples, run_id, cfg, rep)
                    _write_resource_samples(jsonl_fh, resource_samples, run_id, cfg, rep)
                    logger.info(
                        "  → %d INT samples, %d resource records",
                        len(int_samples),
                        len(resource_samples) * 4,
                    )
            except Exception as exc:
                failure_count += 1
                logger.exception("Config failed: %s", exc)
                _write_failure(jsonl_fh, exc, run_id, cfg, rep)
            finally:
                if cooldown_s > 0:
                    logger.info("  cooldown %.1fs", cooldown_s)
                    time.sleep(cooldown_s)

    logging.getLogger().removeHandler(file_handler)
    file_handler.close()
    exit_code = _write_campaign_completion(
        path=artifacts_root / "completion.json",
        run_id=run_id,
        raw_path=jsonl_path,
        system_info_path=sysinfo_path,
        manifest_path=artifacts_root / "manifest.json",
        runner_log_path=runner_log_path,
        scheduled_cell_count=len(cells),
        attempted_cell_count=attempted_cell_count,
        failure_count=failure_count,
    )
    logger.info(
        "Campaign complete. JSONL: %s (failures=%d, completion=%s)",
        jsonl_path,
        failure_count,
        artifacts_root / "completion.json",
    )
    return exit_code


# ---------------------------------------------------------------------------
# Latency workload (RQ1).
# ---------------------------------------------------------------------------


def _run_latency(
    cfg: dict[str, Any],
    repetition: int,
    warmup_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from p4net import Network

    p4_path = _p4_path(str(cfg["p4_program"]))
    program_name = str(cfg["p4_program"])
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

        cold_idle = bool(cfg.get("cold_idle_reference", False))
        carrier_rate = 0 if cold_idle else rate_mbps
        carrier = make_continuous_carrier(
            net=net,
            sender_host="h1",
            receiver_host="h2",
            sender_ip=h1_ip,
            receiver_ip=h2_ip,
            rate_mbps=carrier_rate,
        )
        try:
            if warmup_s > 0:
                logger.info(
                    "  warmup %.1fs (carrier=%s)",
                    warmup_s,
                    "off (cold-idle)" if carrier is None else f"{carrier_rate} Mbps",
                )
                time.sleep(warmup_s)
            if carrier is not None:
                carrier.ensure_running()
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
                    post_instrument_bytes=P4_POST_INSTRUMENT_BYTES.get(
                        program_name, 0
                    ),
                )
            resource_samples = mon.samples()
            _validate_resource_samples(
                resource_samples,
                bmv2_pids=bmv2_pids,
                switch_ifaces=switch_ifaces,
            )
            if carrier is not None:
                carrier.ensure_running()
            return primary, resource_samples
        finally:
            if carrier is not None:
                carrier.stop()
    finally:
        net.stop()


# ---------------------------------------------------------------------------
# Single-point saturation calibration.
# ---------------------------------------------------------------------------


def _run_saturation_point(
    cfg: dict[str, Any],
    repetition: int,
    warmup_s: float,
    artifact_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from p4net import Network

    p4_path = _p4_path(str(cfg["p4_program"]))
    rate_mbps = int(cfg["rate_mbps"])
    n_probes = int(cfg.get("n_probes", 1000))
    probe_interval_ms = float(cfg.get("probe_interval_ms", 60.0))
    probe_packet_size_bytes = int(cfg.get("probe_packet_size_bytes", 256))
    measurement_seconds = int(cfg.get("measurement_seconds", 60))
    iperf_tail_seconds = int(cfg.get("iperf_tail_seconds", 15))
    iperf_post_omit_guard_intervals = int(
        cfg.get("iperf_post_omit_guard_intervals", 1)
    )
    iperf_udp_length_bytes = int(cfg.get("iperf_udp_length_bytes", 1448))
    if not float(warmup_s).is_integer():
        raise ValueError("saturation warmup_seconds must be an integer")

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
        return run_calibration_point(
            net=net,
            sender_host="h1",
            receiver_host="h2",
            sender_ip=h1_ip,
            receiver_ip=h2_ip,
            sender_mac=h1_mac,
            receiver_mac=h2_mac,
            rate_mbps=rate_mbps,
            n_probes=n_probes,
            probe_interval_ms=probe_interval_ms,
            probe_packet_size_bytes=probe_packet_size_bytes,
            sequence_start=repetition * n_probes,
            warmup_seconds=int(warmup_s),
            measurement_seconds=measurement_seconds,
            iperf_tail_seconds=iperf_tail_seconds,
            iperf_post_omit_guard_intervals=iperf_post_omit_guard_intervals,
            iperf_udp_length_bytes=iperf_udp_length_bytes,
            artifact_dir=artifact_dir,
            resource_sample_interval_s=RESOURCE_SAMPLE_INTERVAL_S,
            target_processes=bmv2_pids,
            target_interfaces=switch_ifaces,
        )
    finally:
        net.stop()


# ---------------------------------------------------------------------------
# Resource-only workload (RQ4 direct).
# ---------------------------------------------------------------------------


def _run_resource_only(
    cfg: dict[str, Any],
    warmup_s: float,
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

        cold_idle = bool(cfg.get("cold_idle_reference", False))
        carrier_rate = 0 if cold_idle else rate_mbps
        carrier = make_continuous_carrier(
            net=net,
            sender_host="h1",
            receiver_host="h2",
            sender_ip=h1_ip,
            receiver_ip=h2_ip,
            rate_mbps=carrier_rate,
        )
        try:
            if warmup_s > 0:
                logger.info(
                    "  warmup %.1fs (carrier=%s)",
                    warmup_s,
                    "off (cold-idle)" if carrier is None else f"{carrier_rate} Mbps",
                )
                time.sleep(warmup_s)
            if carrier is not None:
                carrier.ensure_running()
            with ResourceMonitor(
                sample_interval_s=RESOURCE_SAMPLE_INTERVAL_S,
                target_processes=bmv2_pids,
                target_interfaces=switch_ifaces,
            ) as mon:
                time.sleep(duration_s)
            resource_samples = mon.samples()
            minimum_samples = max(
                1,
                math.floor(
                    duration_s / RESOURCE_SAMPLE_INTERVAL_S * 0.9
                ),
            )
            _validate_resource_samples(
                resource_samples,
                bmv2_pids=bmv2_pids,
                switch_ifaces=switch_ifaces,
                minimum_samples=minimum_samples,
            )
            if carrier is not None:
                carrier.ensure_running()
            return None, resource_samples
        finally:
            if carrier is not None:
                carrier.stop()
    finally:
        net.stop()


# ---------------------------------------------------------------------------
# RQ3 multi-hop INT workload.
# ---------------------------------------------------------------------------


def _run_int_multihop(
    cfg: dict[str, Any],
    repetition: int,
    warmup_s: float,
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

        cold_idle = bool(cfg.get("cold_idle_reference", False))
        carrier_rate = 0 if cold_idle else rate_mbps
        carrier = make_continuous_carrier(
            net=net,
            sender_host="h1",
            receiver_host="h2",
            sender_ip=h1_ip,
            receiver_ip=h2_ip,
            rate_mbps=carrier_rate,
        )
        try:
            if warmup_s > 0:
                logger.info(
                    "  warmup %.1fs (carrier=%s)",
                    warmup_s,
                    "off (cold-idle)" if carrier is None else f"{carrier_rate} Mbps",
                )
                time.sleep(warmup_s)
            if carrier is not None:
                carrier.ensure_running()
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
            resource_samples = mon.samples()
            _validate_resource_samples(
                resource_samples,
                bmv2_pids=bmv2_pids,
                switch_ifaces=switch_ifaces,
            )
            if carrier is not None:
                carrier.ensure_running()
            return primary, resource_samples
        finally:
            if carrier is not None:
                carrier.stop()
    finally:
        net.stop()


# ---------------------------------------------------------------------------
# Control-plane workload (RQ2).
# ---------------------------------------------------------------------------


def _run_control_plane(
    cfg: dict[str, Any],
    repetition: int,
    warmup_s: float,
    *,
    async_workload_loop: AsyncWorkloadLoop | None = None,
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
    expected_operations = n_switches * n_entries
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
        # so a continuous-carrier warmup wouldn't have anywhere to land.
        # Honour the campaign warmup_seconds as a plain wait so the gRPC
        # stack has time to settle, but skip the carrier entirely.
        if warmup_s > 0:
            logger.info("  warmup %.1fs (no carrier — control-plane workload)", warmup_s)
            time.sleep(warmup_s)

        loop_context = (
            AsyncWorkloadLoop()
            if mode == "async" and async_workload_loop is None
            else nullcontext(async_workload_loop)
        )
        with loop_context as active_async_loop, ResourceMonitor(
            sample_interval_s=RESOURCE_SAMPLE_INTERVAL_S,
            target_processes=bmv2_pids,
            target_interfaces=switch_ifaces,
        ) as mon:
            if operation == "read":
                if mode == "sync":
                    prefill = run_insert_sync(
                        net, switches, table_name, n_entries, gen
                    )
                    _validate_control_plane_result(
                        prefill,
                        expected_operations=expected_operations,
                        phase="synchronous read prefill",
                    )
                    primary = run_read_sync(net, switches, table_name)
                else:
                    assert active_async_loop is not None
                    prefill = run_insert_async(
                        net,
                        switches,
                        table_name,
                        n_entries,
                        gen,
                        workload_loop=active_async_loop,
                    )
                    _validate_control_plane_result(
                        prefill,
                        expected_operations=expected_operations,
                        phase="asynchronous read prefill",
                    )
                    primary = run_read_async(
                        net,
                        switches,
                        table_name,
                        workload_loop=active_async_loop,
                    )
            else:
                if mode == "sync":
                    primary = run_insert_sync(
                        net, switches, table_name, n_entries, gen
                    )
                else:
                    assert active_async_loop is not None
                    primary = run_insert_async(
                        net,
                        switches,
                        table_name,
                        n_entries,
                        gen,
                        workload_loop=active_async_loop,
                    )
            _validate_control_plane_result(
                primary,
                expected_operations=expected_operations,
                phase=f"{mode} {operation}",
            )
        resource_samples = mon.samples()
        _validate_resource_samples(
            resource_samples,
            bmv2_pids=bmv2_pids,
            switch_ifaces=switch_ifaces,
        )
        return primary, resource_samples
    finally:
        net.stop()


# ---------------------------------------------------------------------------
# JSONL writers.
# ---------------------------------------------------------------------------


def _validate_control_plane_result(
    result: dict[str, Any], *, expected_operations: int, phase: str
) -> None:
    """Reject partial P4Runtime batches before they become RQ2 records."""
    success_count = int(result.get("success_count", -1))
    failure_count = int(result.get("failure_count", -1))
    if failure_count != 0 or success_count != expected_operations:
        raise RuntimeError(
            f"{phase} batch incomplete: success={success_count}, "
            f"failure={failure_count}, expected={expected_operations}"
        )
    wall_clock_s = float(result.get("total_wall_clock_s", 0.0))
    entries_per_second = float(result.get("entries_per_second", -1.0))
    if not math.isfinite(wall_clock_s) or wall_clock_s <= 0:
        raise RuntimeError(f"{phase} batch has invalid wall clock {wall_clock_s}")
    expected_rate = success_count / wall_clock_s
    if not math.isfinite(entries_per_second) or not math.isclose(
        entries_per_second, expected_rate, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise RuntimeError(
            f"{phase} batch rate {entries_per_second} does not match "
            f"success/wall-clock {expected_rate}"
        )


def _latency_config_payload(cfg: dict[str, Any], repetition: int) -> dict[str, Any]:
    return {
        "workload_type": str(cfg["workload_type"]),
        "p4_program": str(cfg["p4_program"]),
        "topology": str(cfg.get("topology", "single_switch")),
        "packet_size_bytes": int(cfg["packet_size_bytes"]),
        "background_load_mbps": int(cfg["background_load_mbps"]),
        "probe_layer": "l2" if cfg["workload_type"] == WORKLOAD_LATENCY_L2 else "l3",
        "cold_idle_reference": bool(cfg.get("cold_idle_reference", False)),
        "n_probes": int(cfg["n_probes"]),
        "probe_interval_ms": float(cfg["probe_interval_ms"]),
        "repetition": repetition,
        "schedule_index": int(cfg.get("schedule_index", -1)),
        "post_instrument_bytes": P4_POST_INSTRUMENT_BYTES.get(
            str(cfg["p4_program"]), 0
        ),
    }


def _control_plane_config_payload(cfg: dict[str, Any], repetition: int) -> dict[str, Any]:
    return {
        "workload_type": str(cfg["workload_type"]),
        "p4_program": str(cfg["p4_program"]),
        "topology": str(cfg.get("topology", "linear_n")),
        "n_switches": int(cfg["n_switches"]),
        "n_entries_per_switch": int(cfg["n_entries_per_switch"]),
        "operation": str(cfg["operation"]),
        "mode": str(cfg["mode"]),
        "repetition": repetition,
        "schedule_index": int(cfg.get("schedule_index", -1)),
    }


def _saturation_config_payload(cfg: dict[str, Any], repetition: int) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "p4_program": str(cfg["p4_program"]),
        "topology": str(cfg.get("topology", "single_switch")),
        "n_switches": int(cfg.get("n_switches", 1)),
        "rate_mbps": int(cfg["rate_mbps"]),
        "background_load_mbps": int(cfg["background_load_mbps"]),
        "n_probes": int(cfg.get("n_probes", 1000)),
        "probe_interval_ms": float(cfg.get("probe_interval_ms", 60.0)),
        "probe_packet_size_bytes": int(cfg.get("probe_packet_size_bytes", 256)),
        "warmup_seconds": int(cfg.get("warmup_seconds", 30)),
        "measurement_seconds": int(cfg.get("measurement_seconds", 60)),
        "iperf_tail_seconds": int(cfg.get("iperf_tail_seconds", 15)),
        "iperf_post_omit_guard_intervals": int(
            cfg.get("iperf_post_omit_guard_intervals", 1)
        ),
        "iperf_udp_length_bytes": int(cfg.get("iperf_udp_length_bytes", 1448)),
        "repetition": repetition,
        "schedule_index": int(cfg.get("schedule_index", -1)),
    }


def _write_saturation_point(
    fh: Any,
    result: dict[str, Any],
    run_id: str,
    cfg: dict[str, Any],
    repetition: int,
) -> None:
    rq = int(cfg["rq"])
    config_payload = _saturation_config_payload(cfg, repetition)
    summary_extras = dict(result)
    probe_samples = list(summary_extras.pop("probe_samples"))
    summary_record = {
        "run_id": run_id,
        "timestamp_utc": _utc_now_iso(),
        "rq": rq,
        "config": config_payload,
        "metric": "saturation_probe_loss_pct",
        "value": float(result["probe_loss_pct"]),
        "extras": summary_extras,
    }
    fh.write(json.dumps(summary_record, allow_nan=False) + "\n")

    for sample in probe_samples:
        record = {
            "run_id": run_id,
            "timestamp_utc": _utc_now_iso(),
            "rq": rq,
            "config": config_payload,
            "metric": "saturation_ingress_to_egress_start_us",
            "value": float(sample["switch_transit_us"]),
            "extras": {
                "sequence": int(sample["sequence"]),
                "ingress_ts_us": int(sample["ingress_ts_us"]),
                "egress_ts_us": int(sample["egress_ts_us"]),
            },
        }
        fh.write(json.dumps(record, allow_nan=False) + "\n")
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
        "workload_type": str(cfg["workload_type"]),
        "p4_program": str(cfg["p4_program"]),
        "topology": str(cfg.get("topology", "linear_n")),
        "n_switches": int(cfg["n_switches"]),
        "background_load_mbps": int(cfg.get("background_load_mbps", 0)),
        "packet_size_bytes": int(cfg["packet_size_bytes"]),
        "n_probes": int(cfg["n_probes"]),
        "probe_interval_ms": float(cfg["probe_interval_ms"]),
        "repetition": repetition,
        "schedule_index": int(cfg.get("schedule_index", -1)),
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
    payload = {
        "p4_program": str(cfg.get("p4_program", "")),
        "topology": str(cfg.get("topology", "single_switch")),
        "n_switches": int(cfg.get("n_switches", 1)),
        "background_load_mbps": int(cfg.get("background_load_mbps", 0)),
        "rate_mbps": int(cfg.get("rate_mbps", cfg.get("background_load_mbps", 0))),
        "source_workload_type": str(cfg.get("workload_type", "")),
        "source_rq": int(cfg.get("rq", 0)),
        "resource_sample_interval_s": RESOURCE_SAMPLE_INTERVAL_S,
        "repetition": repetition,
        "schedule_index": int(cfg.get("schedule_index", -1)),
    }
    if "duration_s" in cfg:
        payload["duration_s"] = float(cfg["duration_s"])
    if "n_probes" in cfg:
        payload["n_probes"] = int(cfg["n_probes"])
    if "probe_interval_ms" in cfg:
        payload["probe_interval_ms"] = float(cfg["probe_interval_ms"])
    if "packet_size_bytes" in cfg:
        payload["packet_size_bytes"] = int(cfg["packet_size_bytes"])
    if "n_entries_per_switch" in cfg:
        payload["n_entries_per_switch"] = int(cfg["n_entries_per_switch"])
    if "operation" in cfg:
        payload["operation"] = str(cfg["operation"])
    if "mode" in cfg:
        payload["mode"] = str(cfg["mode"])
    if "cold_idle_reference" in cfg:
        payload["cold_idle_reference"] = bool(cfg["cold_idle_reference"])
    return payload


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
            "schedule_index": int(cfg.get("schedule_index", -1)),
            "cell": dict(cfg),
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
