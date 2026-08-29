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

Output labeling (Phase H onward)
---------------------------------

The ``--label`` CLI argument appends a suffix to each output filename so
multiple replication runs can coexist without overwriting. For
example, ``--label rep2`` produces ``rq1_summary_rep2.csv``,
``rq2_summary_rep2.csv``, etc. With no ``--label`` (default empty
string), the unsuffixed names are used. Phase G's outputs were
renamed to ``*_rep1.csv`` retroactively when ``--label`` landed so
Phase H's rep2 outputs don't clobber them.

CLI::

    python -m analysis.aggregate --raw data/raw/ --summary data/summaries/
    python -m analysis.aggregate --raw data/raw/ --summary data/summaries/ --label rep2
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
METRIC_SATURATION_SUMMARY = "saturation_probe_loss_pct"
METRIC_SATURATION_LATENCY = "saturation_ingress_to_egress_start_us"
METRIC_RQ4_SET = (
    "cpu_percent_total",
    "cpu_percent_per_bmv2",
    "rss_per_bmv2_bytes",
    "net_io_pps_per_iface",
)

SATURATION_LOSS_CUTOFFS_PCT = (1.0, 5.0, 10.0)
SATURATION_THROUGHPUT_FLOOR_PCT = 95.0
SATURATION_MIN_PASSING_REPETITIONS = 3

SATURATION_RUN_COLUMNS = (
    "run_id",
    "rate_mbps",
    "repetition",
    "schedule_index",
    "nominal_offered_mbps",
    "actual_offered_mbps",
    "achieved_mbps",
    "achieved_to_actual_offered_pct",
    "achieved_to_nominal_pct",
    "sender_seconds",
    "receiver_seconds",
    "sender_datagrams",
    "receiver_total_datagrams",
    "receiver_lost_datagrams",
    "receiver_datagrams",
    "sender_pps",
    "receiver_pps",
    "iperf_receiver_loss_pct",
    "probes_sent",
    "probes_received",
    "probe_loss_pct",
    "duplicate_probes",
    "out_of_range_probes",
    "latency_n_samples",
    "latency_median_us",
    "latency_p99_us",
    "bmv2_cpu_n_samples",
    "bmv2_cpu_mean_pct",
    "bmv2_cpu_p95_pct",
    "system_cpu_n_samples",
    "system_cpu_mean_pct",
    "system_cpu_p95_pct",
    "measurement_start_monotonic_us",
    "measurement_end_monotonic_us",
    "probe_campaign_seconds",
    "iperf_udp_length_bytes",
    "iperf3_version",
    "iperf_client_json_path",
    "iperf_server_json_path",
)

SATURATION_SUMMARY_METRICS = (
    "actual_offered_mbps",
    "achieved_mbps",
    "achieved_to_actual_offered_pct",
    "achieved_to_nominal_pct",
    "sender_pps",
    "receiver_pps",
    "iperf_receiver_loss_pct",
    "probe_loss_pct",
    "latency_median_us",
    "latency_p99_us",
    "bmv2_cpu_mean_pct",
    "bmv2_cpu_p95_pct",
    "system_cpu_mean_pct",
    "system_cpu_p95_pct",
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
    """RQ1 BMv2 ingress-to-egress-start latency summary, one row per (program, size,
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


def _saturation_key(record: dict[str, Any]) -> tuple[str, int, int]:
    config = record["config"]
    return (
        str(record["run_id"]),
        int(config["rate_mbps"]),
        int(config["repetition"]),
    )


def _finite_float(value: Any, field: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"saturation field {field} is not finite: {value!r}")
    return parsed


def aggregate_saturation_runs(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build one calibration row per ``(run_id, rate, repetition)``.

    Latency percentiles and CPU statistics are computed inside each run.
    Samples from different repetitions are never pooled at this stage.
    """
    summaries: dict[tuple[str, int, int], dict[str, Any]] = {}
    latencies: dict[tuple[str, int, int], list[float]] = {}
    cpu_total: dict[tuple[str, int, int], list[float]] = {}
    cpu_bmv2: dict[tuple[str, int, int], list[float]] = {}

    for record in records:
        metric = str(record.get("metric", ""))
        config = record.get("config", {})
        if metric == METRIC_SATURATION_SUMMARY:
            key = _saturation_key(record)
            if key in summaries:
                raise ValueError(f"duplicate saturation summary for {key}")
            summaries[key] = record
        elif metric == METRIC_SATURATION_LATENCY:
            latencies.setdefault(_saturation_key(record), []).append(
                _finite_float(record["value"], "latency")
            )
        elif (
            int(record.get("rq", -1)) == 4
            and config.get("source_workload_type") == "saturation_sweep"
            and metric in {"cpu_percent_total", "cpu_percent_per_bmv2"}
        ):
            key = _saturation_key(record)
            target = cpu_total if metric == "cpu_percent_total" else cpu_bmv2
            target.setdefault(key, []).append(_finite_float(record["value"], metric))

    if not summaries:
        return pd.DataFrame(columns=SATURATION_RUN_COLUMNS)

    rows: list[dict[str, Any]] = []
    for key, record in sorted(summaries.items(), key=lambda item: (item[0][1], item[0][2])):
        config = record["config"]
        extras = record["extras"]
        latency_values = np.asarray(latencies.get(key, []), dtype=float)
        total_cpu_values = np.asarray(cpu_total.get(key, []), dtype=float)
        bmv2_cpu_values = np.asarray(cpu_bmv2.get(key, []), dtype=float)

        probes_sent = int(extras["probes_sent"])
        probes_received = int(extras["probes_received"])
        if probes_sent <= 0 or not 0 <= probes_received <= probes_sent:
            raise ValueError(f"invalid probe counts for saturation cell {key}")
        if latency_values.size != probes_received:
            raise ValueError(
                f"saturation cell {key} has {latency_values.size} latency records "
                f"but reports {probes_received} received probes"
            )
        if total_cpu_values.size == 0 or bmv2_cpu_values.size == 0:
            raise ValueError(f"saturation cell {key} has no in-window CPU samples")

        recomputed_loss = 100.0 * (1.0 - probes_received / probes_sent)
        if not np.isclose(recomputed_loss, float(record["value"]), rtol=0.0, atol=1e-9):
            raise ValueError(f"probe-loss formula mismatch for saturation cell {key}")

        row = {
            "run_id": key[0],
            "rate_mbps": key[1],
            "repetition": key[2],
            "schedule_index": int(config.get("schedule_index", -1)),
            "nominal_offered_mbps": _finite_float(
                extras["nominal_offered_mbps"], "nominal_offered_mbps"
            ),
            "actual_offered_mbps": _finite_float(
                extras["actual_offered_mbps"], "actual_offered_mbps"
            ),
            "achieved_mbps": _finite_float(extras["achieved_mbps"], "achieved_mbps"),
            "achieved_to_actual_offered_pct": _finite_float(
                extras["achieved_to_actual_offered_pct"],
                "achieved_to_actual_offered_pct",
            ),
            "achieved_to_nominal_pct": _finite_float(
                extras["achieved_to_nominal_pct"], "achieved_to_nominal_pct"
            ),
            "sender_seconds": _finite_float(extras["sender_seconds"], "sender_seconds"),
            "receiver_seconds": _finite_float(
                extras["receiver_seconds"], "receiver_seconds"
            ),
            "sender_datagrams": int(extras["sender_datagrams"]),
            "receiver_total_datagrams": int(extras["receiver_total_datagrams"]),
            "receiver_lost_datagrams": int(extras["receiver_lost_datagrams"]),
            "receiver_datagrams": int(extras["receiver_datagrams"]),
            "sender_pps": _finite_float(extras["sender_pps"], "sender_pps"),
            "receiver_pps": _finite_float(extras["receiver_pps"], "receiver_pps"),
            "iperf_receiver_loss_pct": _finite_float(
                extras["iperf_receiver_loss_pct"], "iperf_receiver_loss_pct"
            ),
            "probes_sent": probes_sent,
            "probes_received": probes_received,
            "probe_loss_pct": recomputed_loss,
            "duplicate_probes": int(extras["duplicate_probes"]),
            "out_of_range_probes": int(extras["out_of_range_probes"]),
            "latency_n_samples": int(latency_values.size),
            "latency_median_us": _percentile(latency_values, 50),
            "latency_p99_us": _percentile(latency_values, 99),
            "bmv2_cpu_n_samples": int(bmv2_cpu_values.size),
            "bmv2_cpu_mean_pct": float(bmv2_cpu_values.mean()),
            "bmv2_cpu_p95_pct": _percentile(bmv2_cpu_values, 95),
            "system_cpu_n_samples": int(total_cpu_values.size),
            "system_cpu_mean_pct": float(total_cpu_values.mean()),
            "system_cpu_p95_pct": _percentile(total_cpu_values, 95),
            "measurement_start_monotonic_us": int(
                extras["measurement_start_monotonic_us"]
            ),
            "measurement_end_monotonic_us": int(extras["measurement_end_monotonic_us"]),
            "probe_campaign_seconds": _finite_float(
                extras["probe_campaign_seconds"], "probe_campaign_seconds"
            ),
            "iperf_udp_length_bytes": int(extras["iperf_udp_length_bytes"]),
            "iperf3_version": str(extras["iperf3_version"]),
            "iperf_client_json_path": str(extras["iperf_client_json_path"]),
            "iperf_server_json_path": str(extras["iperf_server_json_path"]),
        }
        if row["measurement_end_monotonic_us"] <= row["measurement_start_monotonic_us"]:
            raise ValueError(f"invalid measurement window for saturation cell {key}")
        if row["duplicate_probes"] or row["out_of_range_probes"]:
            raise ValueError(f"invalid probe identities for saturation cell {key}")
        rows.append(row)
    return pd.DataFrame(rows, columns=SATURATION_RUN_COLUMNS)


def aggregate_saturation(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate run-level calibration rows into one row per offered rate."""
    columns = [
        "rate_mbps",
        "n_reps",
        "probes_sent_total",
        "probes_received_total",
        "probe_loss_pooled_pct",
    ]
    for metric in SATURATION_SUMMARY_METRICS:
        columns.extend((f"{metric}_median", f"{metric}_min", f"{metric}_max"))
    if run_summary.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for rate, group in run_summary.groupby("rate_mbps", sort=True):
        sent_total = int(group["probes_sent"].sum())
        received_total = int(group["probes_received"].sum())
        row: dict[str, Any] = {
            "rate_mbps": int(rate),
            "n_reps": len(group),
            "probes_sent_total": sent_total,
            "probes_received_total": received_total,
            "probe_loss_pooled_pct": 100.0 * (1.0 - received_total / sent_total),
        }
        for metric in SATURATION_SUMMARY_METRICS:
            values = np.asarray(group[metric], dtype=float)
            row[f"{metric}_median"] = _percentile(values, 50)
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def aggregate_saturation_sensitivity(
    run_summary: pd.DataFrame,
    *,
    loss_cutoffs_pct: tuple[float, ...] = SATURATION_LOSS_CUTOFFS_PCT,
    throughput_floor_pct: float = SATURATION_THROUGHPUT_FLOOR_PCT,
    minimum_passing_repetitions: int = SATURATION_MIN_PASSING_REPETITIONS,
) -> pd.DataFrame:
    """Apply the pre-run joint criterion at each sensitivity cutoff."""
    columns = [
        "loss_cutoff_pct",
        "throughput_floor_pct",
        "rate_mbps",
        "passing_reps",
        "n_reps",
        "rate_qualifies",
        "rmax_mbps",
    ]
    if run_summary.empty:
        return pd.DataFrame(columns=columns)
    if minimum_passing_repetitions < 1:
        raise ValueError("minimum_passing_repetitions must be >= 1")

    rows: list[dict[str, Any]] = []
    for cutoff in loss_cutoffs_pct:
        cutoff_rows: list[dict[str, Any]] = []
        for rate, group in run_summary.groupby("rate_mbps", sort=True):
            passes = (
                (group["probe_loss_pct"] <= float(cutoff))
                & (group["achieved_to_nominal_pct"] >= float(throughput_floor_pct))
            )
            passing_reps = int(passes.sum())
            cutoff_rows.append(
                {
                    "loss_cutoff_pct": float(cutoff),
                    "throughput_floor_pct": float(throughput_floor_pct),
                    "rate_mbps": int(rate),
                    "passing_reps": passing_reps,
                    "n_reps": len(group),
                    "rate_qualifies": passing_reps >= minimum_passing_repetitions,
                }
            )
        qualifying = [row["rate_mbps"] for row in cutoff_rows if row["rate_qualifies"]]
        rmax = max(qualifying) if qualifying else None
        for row in cutoff_rows:
            row["rmax_mbps"] = rmax
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


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
    parser.add_argument(
        "--label",
        default="",
        help=(
            "Optional suffix for output CSV filenames, e.g. '--label rep2' "
            "writes rq1_summary_rep2.csv. Default empty: writes rq1_summary.csv "
            "(unsuffixed). Used to keep multiple replication runs side-by-side."
        ),
    )
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

    saturation_run_summary = aggregate_saturation_runs(all_records)
    summaries: dict[str, pd.DataFrame] = {
        "rq1_summary": aggregate_rq1(all_records),
        "rq2_summary": aggregate_rq2(all_records),
        "rq3_summary": aggregate_rq3(all_records),
        "rq4_summary": aggregate_rq4(all_records),
        "saturation_run_summary": saturation_run_summary,
        "saturation_summary": aggregate_saturation(saturation_run_summary),
        "saturation_sensitivity": aggregate_saturation_sensitivity(saturation_run_summary),
        "experiment_log": aggregate_experiment_log(jsonl_paths, file_records),
    }
    suffix = f"_{args.label}" if args.label else ""
    for name, df in summaries.items():
        out_path = args.summary / f"{name}{suffix}.csv"
        df.to_csv(out_path, index=False)
        print(f"  wrote {out_path} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
