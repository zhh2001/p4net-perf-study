"""Aggregate raw JSONL measurement records into per-RQ summary CSVs.

The runner emits one JSONL record per measurement sample (RQ1, RQ3) or
per repetition (RQ2) or per resource-monitor tick (RQ4). For analysis
and plotting, the canonical inputs are *summary* tables — one row per
(config, metric) cell with descriptive statistics computed from the
per-sample records.

This module reads ``data/raw/*.jsonl``, dispatches each record to the
right per-RQ aggregator based on the ``rq`` field, and writes::

    data/summaries/rq1_summary.csv          one row per RQ1 config
    data/summaries/rq2_summary.csv          one row per RQ2 config
    data/summaries/rq3_summary.csv          one row per RQ3 config
    data/summaries/rq4_summary.csv          one row per (RQ4 config, metric)
    data/summaries/experiment_log.csv       one row per JSONL file

The experiment log records ``run_id``, source file, timestamp range,
and the total record count per RQ for traceability.

CLI::

    python -m analysis.aggregate --raw data/raw/ --summary data/summaries/
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Metric names emitted by the runner.
METRIC_RQ1 = "switch_transit_us"
METRIC_RQ2 = "control_plane_wall_clock_s"
METRIC_RQ3 = "int_drift_us"
METRIC_RQ4_SET = (
    "cpu_percent_total",
    "cpu_percent_per_bmv2",
    "rss_per_bmv2_bytes",
    "net_io_pps_per_iface",
)


def _percentile(values: np.ndarray, q: float) -> float:
    """``np.percentile`` with empty-array → NaN guard."""
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def _iter_records(jsonl_path: Path) -> Iterable[dict[str, Any]]:
    """Stream JSONL records from a single file."""
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            yield json.loads(line)


def aggregate_rq1(records: list[dict[str, Any]]) -> pd.DataFrame:
    """RQ1 switch-transit latency summary, one row per (program, size,
    load, cold_idle_reference) tuple."""
    rows: dict[tuple, list[float]] = {}
    for r in records:
        if r["metric"] != METRIC_RQ1:
            continue
        c = r["config"]
        key = (
            c["p4_program"],
            int(c["packet_size_bytes"]),
            int(c["background_load_mbps"]),
            bool(c.get("cold_idle_reference", False)),
        )
        rows.setdefault(key, []).append(float(r["value"]))
    out: list[dict[str, Any]] = []
    for (prog, size, load, cold), vs in sorted(rows.items()):
        arr = np.asarray(vs, dtype=float)
        out.append(
            {
                "p4_program": prog,
                "packet_size_bytes": size,
                "background_load_mbps": load,
                "cold_idle_reference": cold,
                "n_samples": int(arr.size),
                "mean_us": float(arr.mean()) if arr.size else float("nan"),
                "std_us": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
                "median_us": _percentile(arr, 50),
                "p25_us": _percentile(arr, 25),
                "p75_us": _percentile(arr, 75),
                "p99_us": _percentile(arr, 99),
                "p999_us": _percentile(arr, 99.9),
            }
        )
    return pd.DataFrame(out)


def aggregate_rq2(records: list[dict[str, Any]]) -> pd.DataFrame:
    """RQ2 control-plane wall-clock summary, one row per (n_switches,
    n_entries_per_switch, operation, mode) tuple. Each row aggregates
    the per-repetition records for that config."""
    rows: dict[tuple, list[tuple[float, float]]] = {}
    for r in records:
        if r["metric"] != METRIC_RQ2:
            continue
        c = r["config"]
        key = (
            int(c["n_switches"]),
            int(c["n_entries_per_switch"]),
            str(c["operation"]),
            str(c["mode"]),
        )
        eps = float(r["extras"].get("entries_per_second", float("nan")))
        rows.setdefault(key, []).append((float(r["value"]), eps))
    out: list[dict[str, Any]] = []
    for (n, k, op, mode), runs in sorted(rows.items()):
        wall = np.asarray([w for w, _ in runs], dtype=float)
        eps = np.asarray([e for _, e in runs], dtype=float)
        out.append(
            {
                "n_switches": n,
                "n_entries_per_switch": k,
                "operation": op,
                "mode": mode,
                "n_reps": int(wall.size),
                "mean_s": float(wall.mean()) if wall.size else float("nan"),
                "std_s": float(wall.std(ddof=1)) if wall.size > 1 else float("nan"),
                "median_s": _percentile(wall, 50),
                "p25_s": _percentile(wall, 25),
                "p75_s": _percentile(wall, 75),
                "median_entries_per_sec": _percentile(eps, 50),
            }
        )
    return pd.DataFrame(out)


def aggregate_rq3(records: list[dict[str, Any]]) -> pd.DataFrame:
    """RQ3 INT drift summary, one row per (n_switches, load) tuple.

    Drift values can be negative because per-switch ``boot_timestamp_us``
    precision dominates the real inter-hop propagation; the analytical
    target is the *noise envelope* (std, |mean|, |p99|), not the sign.
    """
    rows: dict[tuple, list[float]] = {}
    per_hop: dict[tuple, list[list[float]]] = {}
    for r in records:
        if r["metric"] != METRIC_RQ3:
            continue
        c = r["config"]
        key = (int(c["n_switches"]), int(c["background_load_mbps"]))
        rows.setdefault(key, []).append(float(r["value"]))
        per_hop.setdefault(key, []).append([float(d) for d in r["extras"].get("drift_us", [])])
    out: list[dict[str, Any]] = []
    for (n, load), vs in sorted(rows.items()):
        arr = np.asarray(vs, dtype=float)
        abs_arr = np.abs(arr)
        # per_hop[i][j] = j-th hop's drift on i-th packet; flatten
        flat_hops = [d for packet in per_hop[(n, load)] for d in packet]
        hop_arr = np.asarray(flat_hops, dtype=float) if flat_hops else np.asarray([])
        out.append(
            {
                "n_switches": n,
                "background_load_mbps": load,
                "n_samples": int(arr.size),
                "mean_us": float(arr.mean()) if arr.size else float("nan"),
                "std_us": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
                "median_us": _percentile(arr, 50),
                "p1_us": _percentile(arr, 1),
                "p99_us": _percentile(arr, 99),
                "abs_mean_us": float(abs_arr.mean()) if abs_arr.size else float("nan"),
                "abs_p99_us": _percentile(abs_arr, 99),
                "per_hop_n": int(hop_arr.size),
                "per_hop_abs_mean_us": (
                    float(np.abs(hop_arr).mean()) if hop_arr.size else float("nan")
                ),
            }
        )
    return pd.DataFrame(out)


def aggregate_rq4(records: list[dict[str, Any]]) -> pd.DataFrame:
    """RQ4 resource summary, one row per (config, metric) tuple.

    Each row's stats come from the time-series of per-100ms samples for
    that (config, metric). RQ4 records are tagged ``rq: 4`` even when
    the originating workload is something else (e.g., control_plane);
    the row's ``source_workload_type`` field preserves that.
    """
    rows: dict[tuple, list[float]] = {}
    for r in records:
        if r["rq"] != 4 or r["metric"] not in METRIC_RQ4_SET:
            continue
        c = r["config"]
        key = (
            str(c.get("p4_program", "")),
            str(c.get("topology", "")),
            int(c.get("n_switches", 0)),
            int(c.get("background_load_mbps", 0)),
            str(c.get("source_workload_type", "")),
            str(r["metric"]),
        )
        rows.setdefault(key, []).append(float(r["value"]))
    out: list[dict[str, Any]] = []
    for (prog, topo, n, load, src_wl, metric), vs in sorted(rows.items()):
        arr = np.asarray(vs, dtype=float)
        out.append(
            {
                "p4_program": prog,
                "topology": topo,
                "n_switches": n,
                "background_load_mbps": load,
                "source_workload_type": src_wl,
                "metric": metric,
                "n_samples": int(arr.size),
                "mean": float(arr.mean()) if arr.size else float("nan"),
                "max": float(arr.max()) if arr.size else float("nan"),
                "std": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
                "p5": _percentile(arr, 5),
                "p95": _percentile(arr, 95),
            }
        )
    return pd.DataFrame(out)


def aggregate_experiment_log(
    jsonl_paths: list[Path], all_records: list[tuple[Path, list[dict[str, Any]]]]
) -> pd.DataFrame:
    """One row per JSONL file: counts per RQ, first/last timestamp, run_id."""
    out: list[dict[str, Any]] = []
    for path, records in all_records:
        run_ids = {r.get("run_id", "") for r in records}
        timestamps = [r.get("timestamp_utc", "") for r in records if r.get("timestamp_utc")]
        counts = {"rq1_records": 0, "rq2_records": 0, "rq3_records": 0, "rq4_records": 0}
        for r in records:
            counts[f"rq{r['rq']}_records"] = counts.get(f"rq{r['rq']}_records", 0) + 1
        out.append(
            {
                "source_file": path.name,
                "run_id": next(iter(run_ids), ""),
                "first_timestamp_utc": min(timestamps) if timestamps else "",
                "last_timestamp_utc": max(timestamps) if timestamps else "",
                "total_records": len(records),
                **counts,
            }
        )
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate raw JSONL into per-RQ summary CSVs.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--summary", type=Path, default=Path("data/summaries"))
    args = parser.parse_args(argv)

    args.summary.mkdir(parents=True, exist_ok=True)
    jsonl_paths = sorted(args.raw.glob("*.jsonl"))
    if not jsonl_paths:
        print(f"no JSONL files found under {args.raw}")
        return 0

    all_records: list[dict[str, Any]] = []
    file_records: list[tuple[Path, list[dict[str, Any]]]] = []
    for path in jsonl_paths:
        recs = list(_iter_records(path))
        all_records.extend(recs)
        file_records.append((path, recs))
    print(f"loaded {len(all_records)} records from {len(jsonl_paths)} JSONL files")

    summaries: dict[str, pd.DataFrame] = {
        "rq1_summary": aggregate_rq1(all_records),
        "rq2_summary": aggregate_rq2(all_records),
        "rq3_summary": aggregate_rq3(all_records),
        "rq4_summary": aggregate_rq4(all_records),
        "experiment_log": aggregate_experiment_log(jsonl_paths, file_records),
    }
    for name, df in summaries.items():
        out_path = args.summary / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  wrote {out_path} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
