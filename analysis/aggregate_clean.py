"""Two-level aggregation for independently restarted clean RQ campaigns.

The legacy :mod:`analysis.aggregate` functions pool packet or time-series
samples across repetitions.  This module deliberately implements a separate
path for the replacement campaigns used in Reviewer 1, Concern 4:

1. samples are summarized inside each ``(run_id, config, repetition)`` cell;
2. the five run-level values are summarized as median, minimum, and maximum.

The input to one invocation must be exactly one raw campaign JSONL file for
one research question.  Records from the resource monitor that accompany an
RQ1--RQ3 workload are allowed, but primary measurements from another research
question, multiple run IDs, failed cells, missing repetitions, and duplicate
cells are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.aggregate import METRIC_RQ1, METRIC_RQ2, METRIC_RQ3, METRIC_RQ4_SET

EXPECTED_REPETITIONS = 5
REPO_ROOT = Path(__file__).resolve().parent.parent

RQ1_CONFIG_COLUMNS = (
    "p4_program",
    "packet_size_bytes",
    "background_load_mbps",
    "probe_layer",
    "cold_idle_reference",
    "n_probes",
    "probe_interval_ms",
)
RQ2_CONFIG_COLUMNS = (
    "p4_program",
    "topology",
    "n_switches",
    "n_entries_per_switch",
    "operation",
    "mode",
)
RQ3_CONFIG_COLUMNS = (
    "p4_program",
    "topology",
    "n_switches",
    "background_load_mbps",
    "packet_size_bytes",
    "n_probes",
    "probe_interval_ms",
)
RQ4_CONFIG_COLUMNS = (
    "p4_program",
    "topology",
    "n_switches",
    "background_load_mbps",
    "rate_mbps",
    "source_workload_type",
    "duration_s",
    "resource_sample_interval_s",
    "metric",
)

RQ1_RUN_METRICS = (
    "probes_sent",
    "probes_received",
    "probes_lost",
    "probe_loss_pct",
    "median_us",
    "iqr_us",
    "p99_us",
)
RQ2_RUN_METRICS = ("wall_clock_s", "entries_per_second")
RQ3_RUN_METRICS = (
    "probes_sent",
    "probes_received",
    "probes_lost",
    "probe_loss_pct",
    "n_samples",
    "mean_us",
    "std_us",
    "median_us",
    "iqr_us",
    "p1_us",
    "p99_us",
    "abs_mean_us",
    "abs_std_us",
    "abs_median_us",
    "abs_iqr_us",
    "abs_p99_us",
    "per_hop_n",
    "per_hop_abs_mean_us",
)
RQ4_RUN_METRICS = (
    "n_samples",
    "mean",
    "std",
    "median",
    "iqr",
    "p5",
    "p95",
    "p99",
    "max",
)

_PRIMARY_METRICS = {METRIC_RQ1: 1, METRIC_RQ2: 2, METRIC_RQ3: 3}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"record at {path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{field} is not finite: {value!r}")
    return parsed


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} is not an integer: {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not an integer: {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} is not an integer: {value!r}")
    return parsed


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} is not a boolean: {value!r}")
    return value


def _required(config: dict[str, Any], field: str) -> Any:
    if field not in config:
        raise ValueError(f"clean campaign record is missing config.{field}")
    return config[field]


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile))


def _sample_statistics(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot summarize an empty clean-campaign cell")
    if not np.isfinite(array).all():
        raise ValueError("clean-campaign cell contains a non-finite sample")
    p25 = _percentile(array, 25)
    p75 = _percentile(array, 75)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": _percentile(array, 50),
        "iqr": p75 - p25,
        "p1": _percentile(array, 1),
        "p5": _percentile(array, 5),
        "p95": _percentile(array, 95),
        "p99": _percentile(array, 99),
        "max": float(array.max()),
    }


def _prepare_campaign(records: list[dict[str, Any]], rq: int) -> str:
    if not records:
        raise ValueError("clean campaign contains no records")
    if rq not in {1, 2, 3, 4}:
        raise ValueError(f"rq must be 1, 2, 3, or 4; got {rq}")

    missing_run_id = [index for index, record in enumerate(records) if not record.get("run_id")]
    if missing_run_id:
        raise ValueError(f"clean campaign records missing run_id at indexes {missing_run_id[:5]}")
    run_ids = {str(record["run_id"]) for record in records}
    if len(run_ids) != 1:
        raise ValueError(f"mixed campaign run_ids are not allowed: {sorted(run_ids)}")
    run_id = next(iter(run_ids))

    failures = [record for record in records if record.get("metric") == "config_failure"]
    if failures:
        failed_repetitions = [
            record.get("config", {}).get("repetition") for record in failures
        ]
        raise ValueError(
            "clean campaign contains config_failure records for repetitions "
            f"{failed_repetitions}"
        )

    allowed = set(METRIC_RQ4_SET)
    if rq < 4:
        target_metric = {1: METRIC_RQ1, 2: METRIC_RQ2, 3: METRIC_RQ3}[rq]
        allowed.add(target_metric)
    unexpected = sorted(
        {str(record.get("metric", "")) for record in records} - allowed
    )
    if unexpected:
        raise ValueError(
            f"clean RQ{rq} campaign contains unexpected metrics: {unexpected}"
        )

    for index, record in enumerate(records):
        metric = str(record.get("metric", ""))
        record_rq = _integer(record.get("rq"), "record.rq")
        if metric in _PRIMARY_METRICS and record_rq != rq:
            raise ValueError(
                f"clean RQ{rq} primary record {index} is tagged rq={record_rq}"
            )
        if metric in METRIC_RQ4_SET:
            if record_rq != 4:
                raise ValueError(
                    f"clean resource record {index} is tagged rq={record_rq}"
                )
            config = record.get("config")
            if not isinstance(config, dict):
                raise ValueError(f"clean resource record {index} has no config object")
            source_rq = _integer(config.get("source_rq"), "config.source_rq")
            if source_rq != rq:
                raise ValueError(
                    f"clean RQ{rq} resource record {index} has source_rq={source_rq}"
                )

    primary_rqs = {
        _PRIMARY_METRICS[str(record["metric"])]
        for record in records
        if str(record.get("metric", "")) in _PRIMARY_METRICS
    }
    expected_primary = set() if rq == 4 else {rq}
    if primary_rqs != expected_primary:
        raise ValueError(
            f"clean RQ{rq} campaign primary measurements are {sorted(primary_rqs)}"
        )

    if rq == 4:
        resource_records = [
            record for record in records if record.get("metric") in METRIC_RQ4_SET
        ]
        if not resource_records:
            raise ValueError("clean RQ4 campaign contains no resource records")
        sources = {
            str(record.get("config", {}).get("source_workload_type", ""))
            for record in resource_records
        }
        if sources != {"resource_only"}:
            raise ValueError(
                "clean RQ4 aggregation accepts only source_workload_type=resource_only; "
                f"found {sorted(sources)}"
            )

    return run_id


def _rq1_config(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(_required(config, "p4_program")),
        _integer(_required(config, "packet_size_bytes"), "config.packet_size_bytes"),
        _integer(
            _required(config, "background_load_mbps"),
            "config.background_load_mbps",
        ),
        str(_required(config, "probe_layer")),
        _boolean(
            _required(config, "cold_idle_reference"),
            "config.cold_idle_reference",
        ),
        _integer(_required(config, "n_probes"), "config.n_probes"),
        _finite_float(
            _required(config, "probe_interval_ms"), "config.probe_interval_ms"
        ),
    )


def _rq2_config(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(_required(config, "p4_program")),
        str(_required(config, "topology")),
        _integer(_required(config, "n_switches"), "config.n_switches"),
        _integer(
            _required(config, "n_entries_per_switch"),
            "config.n_entries_per_switch",
        ),
        str(_required(config, "operation")),
        str(_required(config, "mode")),
    )


def _rq3_config(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(_required(config, "p4_program")),
        str(_required(config, "topology")),
        _integer(_required(config, "n_switches"), "config.n_switches"),
        _integer(
            _required(config, "background_load_mbps"),
            "config.background_load_mbps",
        ),
        _integer(_required(config, "packet_size_bytes"), "config.packet_size_bytes"),
        _integer(_required(config, "n_probes"), "config.n_probes"),
        _finite_float(
            _required(config, "probe_interval_ms"), "config.probe_interval_ms"
        ),
    )


def _rq4_config(config: dict[str, Any], metric: str) -> tuple[Any, ...]:
    return (
        str(_required(config, "p4_program")),
        str(_required(config, "topology")),
        _integer(_required(config, "n_switches"), "config.n_switches"),
        _integer(
            _required(config, "background_load_mbps"),
            "config.background_load_mbps",
        ),
        _integer(_required(config, "rate_mbps"), "config.rate_mbps"),
        str(_required(config, "source_workload_type")),
        _finite_float(_required(config, "duration_s"), "config.duration_s"),
        _finite_float(
            _required(config, "resource_sample_interval_s"),
            "config.resource_sample_interval_s",
        ),
        metric,
    )


def _validate_repetitions(
    run_summary: pd.DataFrame,
    config_columns: Sequence[str],
    *,
    expected_repetitions: int,
) -> None:
    if expected_repetitions < 1:
        raise ValueError("expected_repetitions must be at least one")
    expected = set(range(expected_repetitions))
    for config, group in run_summary.groupby(list(config_columns), sort=False, dropna=False):
        actual_values = [int(value) for value in group["repetition"]]
        actual = set(actual_values)
        if actual != expected or len(actual_values) != expected_repetitions:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            duplicates = sorted(
                repetition
                for repetition in actual
                if actual_values.count(repetition) > 1
            )
            raise ValueError(
                f"clean config {config!r} repetition mismatch: "
                f"missing={missing}, extra={extra}, duplicates={duplicates}"
            )


def _across_repetitions(
    run_summary: pd.DataFrame,
    config_columns: Sequence[str],
    metric_columns: Sequence[str],
    *,
    expected_repetitions: int,
) -> pd.DataFrame:
    _validate_repetitions(
        run_summary,
        config_columns,
        expected_repetitions=expected_repetitions,
    )
    output_columns = ["run_id", *config_columns, "n_reps"]
    for metric in metric_columns:
        output_columns.extend((f"{metric}_median", f"{metric}_min", f"{metric}_max"))

    rows: list[dict[str, Any]] = []
    for config, group in run_summary.groupby(list(config_columns), sort=True, dropna=False):
        config_values = config if isinstance(config, tuple) else (config,)
        run_ids = {str(value) for value in group["run_id"]}
        if len(run_ids) != 1:
            raise ValueError(f"config {config!r} contains mixed run_ids: {sorted(run_ids)}")
        row: dict[str, Any] = {
            "run_id": next(iter(run_ids)),
            **dict(zip(config_columns, config_values, strict=True)),
            "n_reps": len(group),
        }
        for metric in metric_columns:
            values = np.asarray(group[metric], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"config {config!r} has non-finite run metric {metric}")
            row[f"{metric}_median"] = _percentile(values, 50)
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows, columns=output_columns)


def _packet_run_rows(
    records: Iterable[dict[str, Any]],
    *,
    metric: str,
    config_parser: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("metric") != metric:
            continue
        config = record.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"{metric} record has no config object")
        repetition = _integer(_required(config, "repetition"), "config.repetition")
        key = (str(record["run_id"]), *config_parser(config), repetition)
        groups[key].append(record)
    if not groups:
        raise ValueError(f"clean campaign contains no {metric} records")
    return groups


def _cell_schedule_index(
    cell_records: Sequence[dict[str, Any]], cell: tuple[Any, ...]
) -> int:
    indexes = {
        _integer(
            _required(record.get("config", {}), "schedule_index"),
            "config.schedule_index",
        )
        for record in cell_records
    }
    if len(indexes) != 1:
        raise ValueError(f"clean cell {cell!r} contains schedule indexes {sorted(indexes)}")
    schedule_index = next(iter(indexes))
    if schedule_index < 0:
        raise ValueError(f"clean cell {cell!r} has negative schedule_index={schedule_index}")
    return schedule_index


def aggregate_clean_rq1_runs(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Return one received-probe latency row per clean RQ1 repetition."""
    _prepare_campaign(records, 1)
    groups = _packet_run_rows(records, metric=METRIC_RQ1, config_parser=_rq1_config)
    columns = [
        "run_id",
        *RQ1_CONFIG_COLUMNS,
        "repetition",
        "schedule_index",
        *RQ1_RUN_METRICS,
    ]
    rows: list[dict[str, Any]] = []
    for key, cell_records in sorted(groups.items()):
        run_id, *config_values, repetition = key
        config_row = dict(zip(RQ1_CONFIG_COLUMNS, config_values, strict=True))
        n_probes = int(config_row["n_probes"])
        if n_probes <= 0:
            raise ValueError(f"RQ1 config has non-positive n_probes={n_probes}")

        sequences = [
            _integer(record.get("extras", {}).get("sequence"), "extras.sequence")
            for record in cell_records
        ]
        if len(set(sequences)) != len(sequences):
            raise ValueError(f"RQ1 cell {key!r} contains duplicate probe sequences")
        expected_start = int(repetition) * n_probes
        expected_stop = expected_start + n_probes
        outside = [seq for seq in sequences if not expected_start <= seq < expected_stop]
        if outside:
            raise ValueError(
                f"RQ1 cell {key!r} has sequences outside "
                f"[{expected_start}, {expected_stop}): {outside[:5]}"
            )
        if len(sequences) > n_probes:
            raise ValueError(f"RQ1 cell {key!r} received more probes than intended")

        values = [
            _finite_float(record.get("value"), f"RQ1 cell {key!r} value")
            for record in cell_records
        ]
        stats = _sample_statistics(values)
        received = len(values)
        lost = n_probes - received
        rows.append(
            {
                "run_id": run_id,
                **config_row,
                "repetition": int(repetition),
                "schedule_index": _cell_schedule_index(cell_records, key),
                "probes_sent": n_probes,
                "probes_received": received,
                "probes_lost": lost,
                "probe_loss_pct": 100.0 * lost / n_probes,
                "median_us": stats["median"],
                "iqr_us": stats["iqr"],
                "p99_us": stats["p99"],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def aggregate_clean_rq1(
    run_summary: pd.DataFrame, *, expected_repetitions: int = EXPECTED_REPETITIONS
) -> pd.DataFrame:
    """Summarize five RQ1 run-level rows per configuration."""
    return _across_repetitions(
        run_summary,
        RQ1_CONFIG_COLUMNS,
        RQ1_RUN_METRICS,
        expected_repetitions=expected_repetitions,
    )


def aggregate_clean_rq2_runs(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Return one control-plane wall-time/throughput row per RQ2 repetition."""
    _prepare_campaign(records, 2)
    columns = [
        "run_id",
        *RQ2_CONFIG_COLUMNS,
        "repetition",
        "schedule_index",
        "success_count",
        "failure_count",
        *RQ2_RUN_METRICS,
    ]
    seen: set[tuple[Any, ...]] = set()
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("metric") != METRIC_RQ2:
            continue
        config = record.get("config")
        if not isinstance(config, dict):
            raise ValueError("RQ2 record has no config object")
        config_values = _rq2_config(config)
        repetition = _integer(_required(config, "repetition"), "config.repetition")
        key = (str(record["run_id"]), *config_values, repetition)
        if key in seen:
            raise ValueError(f"duplicate RQ2 run-level record for {key!r}")
        seen.add(key)
        extras = record.get("extras")
        if not isinstance(extras, dict):
            raise ValueError(f"RQ2 cell {key!r} has no extras object")
        success_count = _integer(extras.get("success_count"), "extras.success_count")
        failure_count = _integer(extras.get("failure_count"), "extras.failure_count")
        if failure_count != 0:
            raise ValueError(f"RQ2 cell {key!r} reports failure_count={failure_count}")
        if success_count <= 0:
            raise ValueError(f"RQ2 cell {key!r} has no successful operations")
        expected_operations = int(config_values[2]) * int(config_values[3])
        if success_count != expected_operations:
            raise ValueError(
                f"RQ2 cell {key!r} reports success_count={success_count}; "
                f"expected {expected_operations}"
            )
        wall_clock_s = _finite_float(record.get("value"), "RQ2 wall clock")
        if wall_clock_s <= 0:
            raise ValueError(f"RQ2 cell {key!r} has non-positive wall clock")
        entries_per_second = _finite_float(
            extras.get("entries_per_second"), "RQ2 entries_per_second"
        )
        expected_rate = success_count / wall_clock_s
        if not np.isclose(
            entries_per_second, expected_rate, rtol=1e-12, atol=1e-9
        ):
            raise ValueError(
                f"RQ2 cell {key!r} entries_per_second={entries_per_second} "
                f"does not match success/wall-clock={expected_rate}"
            )
        rows.append(
            {
                "run_id": key[0],
                **dict(zip(RQ2_CONFIG_COLUMNS, config_values, strict=True)),
                "repetition": repetition,
                "schedule_index": _cell_schedule_index([record], key),
                "success_count": success_count,
                "failure_count": failure_count,
                "wall_clock_s": wall_clock_s,
                "entries_per_second": entries_per_second,
            }
        )
    if not rows:
        raise ValueError(f"clean campaign contains no {METRIC_RQ2} records")
    ordered = sorted(
        rows,
        key=lambda row: tuple(row[column] for column in [*RQ2_CONFIG_COLUMNS, "repetition"]),
    )
    return pd.DataFrame(ordered, columns=columns)


def aggregate_clean_rq2(
    run_summary: pd.DataFrame, *, expected_repetitions: int = EXPECTED_REPETITIONS
) -> pd.DataFrame:
    """Summarize five RQ2 run-level rows per configuration."""
    return _across_repetitions(
        run_summary,
        RQ2_CONFIG_COLUMNS,
        RQ2_RUN_METRICS,
        expected_repetitions=expected_repetitions,
    )


def aggregate_clean_rq3_runs(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Return signed and absolute INT-drift statistics per RQ3 repetition."""
    _prepare_campaign(records, 3)
    groups = _packet_run_rows(records, metric=METRIC_RQ3, config_parser=_rq3_config)
    columns = [
        "run_id",
        *RQ3_CONFIG_COLUMNS,
        "repetition",
        "schedule_index",
        *RQ3_RUN_METRICS,
    ]
    rows: list[dict[str, Any]] = []
    for key, cell_records in sorted(groups.items()):
        run_id, *config_values, repetition = key
        config_row = dict(zip(RQ3_CONFIG_COLUMNS, config_values, strict=True))
        n_probes = int(config_row["n_probes"])
        n_switches = int(config_row["n_switches"])
        if n_probes <= 0:
            raise ValueError(f"RQ3 config has non-positive n_probes={n_probes}")
        if n_switches < 2:
            raise ValueError(f"RQ3 config has fewer than two switches: {n_switches}")
        sequences = [
            _integer(record.get("extras", {}).get("sequence"), "extras.sequence")
            for record in cell_records
        ]
        if len(set(sequences)) != len(sequences):
            raise ValueError(f"RQ3 cell {key!r} contains duplicate probe sequences")
        expected_start = int(repetition) * n_probes
        expected_stop = expected_start + n_probes
        outside = [seq for seq in sequences if not expected_start <= seq < expected_stop]
        if outside:
            raise ValueError(
                f"RQ3 cell {key!r} has sequences outside "
                f"[{expected_start}, {expected_stop}): {outside[:5]}"
            )
        if len(sequences) > n_probes:
            raise ValueError(f"RQ3 cell {key!r} received more probes than intended")

        values: list[float] = []
        per_hop_values: list[float] = []
        for record in cell_records:
            extras = record.get("extras")
            if not isinstance(extras, dict):
                raise ValueError(f"RQ3 cell {key!r} record has no extras object")
            hop_count = _integer(extras.get("hop_count"), "extras.hop_count")
            if hop_count != n_switches:
                raise ValueError(
                    f"RQ3 cell {key!r} reports hop_count={hop_count}; "
                    f"expected {n_switches}"
                )
            expected_hop_fields = (
                "switch_ids",
                "raw_ingress_us",
                "raw_egress_us",
                "boot_us",
                "aligned_ingress_us",
                "aligned_egress_us",
            )
            for field in expected_hop_fields:
                hop_values = extras.get(field)
                if not isinstance(hop_values, list) or len(hop_values) != n_switches:
                    raise ValueError(
                        f"RQ3 cell {key!r} extras.{field} has "
                        f"{len(hop_values) if isinstance(hop_values, list) else 'no'} "
                        f"hop values; expected {n_switches}"
                    )
            switch_ids = [
                _integer(value, "extras.switch_ids")
                for value in extras["switch_ids"]
            ]
            if switch_ids != list(range(1, n_switches + 1)):
                raise ValueError(
                    f"RQ3 cell {key!r} has unexpected switch order {switch_ids}"
                )
            drift = extras.get("drift_us")
            if not isinstance(drift, list) or len(drift) != n_switches - 1:
                raise ValueError(
                    f"RQ3 cell {key!r} record has "
                    f"{len(drift) if isinstance(drift, list) else 'no'} per-hop "
                    f"drift_us values; expected {n_switches - 1}"
                )
            parsed_drift = [
                _finite_float(value, "extras.drift_us") for value in drift
            ]
            observed_average = _finite_float(
                record.get("value"), f"RQ3 cell {key!r} value"
            )
            expected_average = float(np.mean(parsed_drift))
            if not np.isclose(observed_average, expected_average, rtol=1e-12, atol=1e-9):
                raise ValueError(
                    f"RQ3 cell {key!r} average drift {observed_average} does not "
                    f"match per-hop mean {expected_average}"
                )
            values.append(observed_average)
            per_hop_values.extend(parsed_drift)
        signed = _sample_statistics(values)
        absolute = _sample_statistics(np.abs(np.asarray(values, dtype=float)).tolist())
        received = len(values)
        lost = n_probes - received
        rows.append(
            {
                "run_id": run_id,
                **config_row,
                "repetition": int(repetition),
                "schedule_index": _cell_schedule_index(cell_records, key),
                "probes_sent": n_probes,
                "probes_received": received,
                "probes_lost": lost,
                "probe_loss_pct": 100.0 * lost / n_probes,
                "n_samples": received,
                "mean_us": signed["mean"],
                "std_us": signed["std"],
                "median_us": signed["median"],
                "iqr_us": signed["iqr"],
                "p1_us": signed["p1"],
                "p99_us": signed["p99"],
                "abs_mean_us": absolute["mean"],
                "abs_std_us": absolute["std"],
                "abs_median_us": absolute["median"],
                "abs_iqr_us": absolute["iqr"],
                "abs_p99_us": absolute["p99"],
                "per_hop_n": len(per_hop_values),
                "per_hop_abs_mean_us": float(
                    np.abs(np.asarray(per_hop_values, dtype=float)).mean()
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def aggregate_clean_rq3(
    run_summary: pd.DataFrame, *, expected_repetitions: int = EXPECTED_REPETITIONS
) -> pd.DataFrame:
    """Summarize five RQ3 run-level rows per configuration."""
    return _across_repetitions(
        run_summary,
        RQ3_CONFIG_COLUMNS,
        RQ3_RUN_METRICS,
        expected_repetitions=expected_repetitions,
    )


def aggregate_clean_rq4_runs(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Return one time-series summary per RQ4 config/repetition/metric."""
    _prepare_campaign(records, 4)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        metric = str(record.get("metric", ""))
        if metric not in METRIC_RQ4_SET:
            continue
        config = record.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"RQ4 {metric} record has no config object")
        repetition = _integer(_required(config, "repetition"), "config.repetition")
        key = (str(record["run_id"]), *_rq4_config(config, metric), repetition)
        groups[key].append(record)
    if not groups:
        raise ValueError("clean campaign contains no RQ4 resource records")

    by_cell: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for key, cell_records in groups.items():
        metric = str(key[-2])
        base_key = (*key[:-2], key[-1])
        by_cell[base_key][metric] = cell_records
    expected_metrics = set(METRIC_RQ4_SET)
    for base_key, metric_records in by_cell.items():
        actual_metrics = set(metric_records)
        if actual_metrics != expected_metrics:
            raise ValueError(
                f"RQ4 cell {base_key!r} metric mismatch: "
                f"missing={sorted(expected_metrics - actual_metrics)}, "
                f"extra={sorted(actual_metrics - expected_metrics)}"
            )
        index_sets: dict[str, set[int]] = {}
        timestamps_by_metric: dict[str, dict[int, int]] = {}
        schedule_indexes: set[int] = set()
        for metric, samples in metric_records.items():
            index_sets[metric] = {
                _integer(
                    _required(record["config"], "sample_index"),
                    "config.sample_index",
                )
                for record in samples
            }
            timestamps_by_metric[metric] = {
                _integer(
                    _required(record["config"], "sample_index"),
                    "config.sample_index",
                ): _integer(
                    _required(record.get("extras", {}), "timestamp_us"),
                    "extras.timestamp_us",
                )
                for record in samples
            }
            if len(timestamps_by_metric[metric]) != len(samples):
                raise ValueError(
                    f"RQ4 cell {base_key!r} contains duplicate sample indexes"
                )
            schedule_indexes.add(_cell_schedule_index(samples, base_key))
        if len(schedule_indexes) != 1:
            raise ValueError(
                f"RQ4 cell {base_key!r} metrics have schedule indexes "
                f"{sorted(schedule_indexes)}"
            )
        reference_indexes = next(iter(index_sets.values()))
        if reference_indexes != set(range(len(reference_indexes))):
            raise ValueError(f"RQ4 cell {base_key!r} sample indexes are not contiguous")
        if any(indexes != reference_indexes for indexes in index_sets.values()):
            raise ValueError(
                f"RQ4 cell {base_key!r} resource metrics have different sample indexes"
            )
        reference_timestamps = next(iter(timestamps_by_metric.values()))
        if any(
            timestamps != reference_timestamps
            for timestamps in timestamps_by_metric.values()
        ):
            raise ValueError(
                f"RQ4 cell {base_key!r} resource metrics have different timestamps"
            )
        ordered_timestamps = [
            reference_timestamps[index] for index in sorted(reference_timestamps)
        ]
        if any(
            later <= earlier
            for earlier, later in pairwise(ordered_timestamps)
        ):
            raise ValueError(
                f"RQ4 cell {base_key!r} timestamps are not strictly increasing"
            )

    columns = [
        "run_id",
        *RQ4_CONFIG_COLUMNS,
        "repetition",
        "schedule_index",
        *RQ4_RUN_METRICS,
    ]
    rows: list[dict[str, Any]] = []
    for key, cell_records in sorted(groups.items()):
        run_id, *config_values, repetition = key
        n_switches = int(config_values[2])
        metric = str(config_values[-1])
        for record in cell_records:
            extras = record.get("extras")
            if not isinstance(extras, dict):
                raise ValueError(f"RQ4 cell {key!r} record has no extras object")
            if metric in {"cpu_percent_per_bmv2", "rss_per_bmv2_bytes"}:
                per_pid = extras.get("per_pid")
                if not isinstance(per_pid, dict) or len(per_pid) != n_switches:
                    raise ValueError(
                        f"RQ4 cell {key!r} has "
                        f"{len(per_pid) if isinstance(per_pid, dict) else 'no'} "
                        f"per-process values; expected {n_switches}"
                    )
            if metric == "net_io_pps_per_iface":
                per_iface = extras.get("per_iface")
                if not isinstance(per_iface, dict) or not per_iface:
                    raise ValueError(
                        f"RQ4 cell {key!r} has no per-interface measurements"
                    )
        sample_indexes = [
            _integer(
                _required(record["config"], "sample_index"),
                "config.sample_index",
            )
            for record in cell_records
        ]
        if len(set(sample_indexes)) != len(sample_indexes):
            raise ValueError(f"RQ4 cell {key!r} contains duplicate sample_index values")
        values = [
            _finite_float(record.get("value"), f"RQ4 cell {key!r} value")
            for record in cell_records
        ]
        stats = _sample_statistics(values)
        rows.append(
            {
                "run_id": run_id,
                **dict(zip(RQ4_CONFIG_COLUMNS, config_values, strict=True)),
                "repetition": int(repetition),
                "schedule_index": _cell_schedule_index(cell_records, key),
                "n_samples": len(values),
                "mean": stats["mean"],
                "std": stats["std"],
                "median": stats["median"],
                "iqr": stats["iqr"],
                "p5": stats["p5"],
                "p95": stats["p95"],
                "p99": stats["p99"],
                "max": stats["max"],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def aggregate_clean_rq4(
    run_summary: pd.DataFrame, *, expected_repetitions: int = EXPECTED_REPETITIONS
) -> pd.DataFrame:
    """Summarize five RQ4 run-level rows per configuration and metric."""
    return _across_repetitions(
        run_summary,
        RQ4_CONFIG_COLUMNS,
        RQ4_RUN_METRICS,
        expected_repetitions=expected_repetitions,
    )


def aggregate_clean_campaign(
    records: list[dict[str, Any]],
    rq: int,
    *,
    expected_repetitions: int = EXPECTED_REPETITIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build run-level and across-run tables for one clean campaign."""
    run_aggregators = {
        1: aggregate_clean_rq1_runs,
        2: aggregate_clean_rq2_runs,
        3: aggregate_clean_rq3_runs,
        4: aggregate_clean_rq4_runs,
    }
    config_aggregators = {
        1: aggregate_clean_rq1,
        2: aggregate_clean_rq2,
        3: aggregate_clean_rq3,
        4: aggregate_clean_rq4,
    }
    if rq not in run_aggregators:
        raise ValueError(f"rq must be 1, 2, 3, or 4; got {rq}")
    run_summary = run_aggregators[rq](records)
    config_summary = config_aggregators[rq](
        run_summary, expected_repetitions=expected_repetitions
    )
    return run_summary, config_summary


def _scheduled_config_identity(rq: int, config: dict[str, Any]) -> tuple[Any, ...]:
    if rq == 1:
        workload_type = str(_required(config, "workload_type"))
        if workload_type not in {"latency_l2", "latency_l3"}:
            raise ValueError(f"unexpected RQ1 workload_type={workload_type!r}")
        return (
            str(_required(config, "p4_program")),
            _integer(_required(config, "packet_size_bytes"), "packet_size_bytes"),
            _integer(
                _required(config, "background_load_mbps"),
                "background_load_mbps",
            ),
            "l2" if workload_type == "latency_l2" else "l3",
            _boolean(config.get("cold_idle_reference", False), "cold_idle_reference"),
            _integer(_required(config, "n_probes"), "n_probes"),
            _finite_float(_required(config, "probe_interval_ms"), "probe_interval_ms"),
        )
    if rq == 2:
        return _rq2_config(config)
    if rq == 3:
        return _rq3_config(config)
    if rq == 4:
        background_load = _integer(
            config.get("background_load_mbps", 0), "background_load_mbps"
        )
        return (
            str(_required(config, "p4_program")),
            str(config.get("topology", "single_switch")),
            _integer(config.get("n_switches", 1), "n_switches"),
            background_load,
            _integer(config.get("rate_mbps", background_load), "rate_mbps"),
            str(_required(config, "workload_type")),
            _finite_float(_required(config, "duration_s"), "duration_s"),
        )
    raise ValueError(f"rq must be 1, 2, 3, or 4; got {rq}")


def _observed_config_identity(rq: int, row: Any) -> tuple[Any, ...]:
    if rq == 1:
        return tuple(getattr(row, column) for column in RQ1_CONFIG_COLUMNS)
    if rq == 2:
        return tuple(getattr(row, column) for column in RQ2_CONFIG_COLUMNS)
    if rq == 3:
        return tuple(getattr(row, column) for column in RQ3_CONFIG_COLUMNS)
    if rq == 4:
        return (
            row.p4_program,
            row.topology,
            int(row.n_switches),
            int(row.background_load_mbps),
            int(row.rate_mbps),
            row.source_workload_type,
            float(row.duration_s),
        )
    raise ValueError(f"rq must be 1, 2, 3, or 4; got {rq}")


def validate_clean_manifest(
    manifest: dict[str, Any],
    rq: int,
    run_summary: pd.DataFrame,
    *,
    expected_repetitions: int = EXPECTED_REPETITIONS,
) -> None:
    """Match every observed run-level cell to the frozen randomized schedule."""
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("clean campaign manifest must have schema_version=1")
    manifest_run_id = str(manifest.get("run_id", ""))
    observed_run_ids = {str(value) for value in run_summary["run_id"]}
    if observed_run_ids != {manifest_run_id}:
        raise ValueError(
            f"manifest/raw run_id mismatch: manifest={manifest_run_id!r}, "
            f"raw={sorted(observed_run_ids)}"
        )

    campaign = manifest.get("campaign")
    if not isinstance(campaign, dict):
        raise ValueError("clean campaign manifest has no campaign object")
    campaign_options = campaign.get("campaign")
    if not isinstance(campaign_options, dict):
        raise ValueError("clean campaign manifest has no campaign.campaign object")
    if campaign_options.get("shuffle_all_cells") is not True:
        raise ValueError("clean campaign must set campaign.shuffle_all_cells=true")
    campaign_configs = campaign.get("configs")
    if not isinstance(campaign_configs, list) or not campaign_configs:
        raise ValueError("clean campaign manifest has no campaign configs")
    for config in campaign_configs:
        if not isinstance(config, dict):
            raise ValueError("clean campaign manifest contains a non-object config")
        if _integer(config.get("rq"), "config.rq") != rq:
            raise ValueError(f"clean RQ{rq} manifest contains another RQ config")
        if _integer(config.get("repetitions"), "config.repetitions") != expected_repetitions:
            raise ValueError(
                f"clean RQ{rq} config does not request {expected_repetitions} repetitions"
            )

    scheduled_cells = manifest.get("scheduled_cells")
    if not isinstance(scheduled_cells, list) or not scheduled_cells:
        raise ValueError("clean campaign manifest has no scheduled_cells")
    expected_cell_count = len(campaign_configs) * expected_repetitions
    if len(scheduled_cells) != expected_cell_count:
        raise ValueError(
            f"clean campaign schedules {len(scheduled_cells)} cells; "
            f"expected {expected_cell_count}"
        )
    schedule_indexes = [
        _integer(cell.get("schedule_index"), "scheduled_cells.schedule_index")
        for cell in scheduled_cells
        if isinstance(cell, dict)
    ]
    if len(schedule_indexes) != len(scheduled_cells):
        raise ValueError("clean campaign manifest contains a non-object scheduled cell")
    if sorted(schedule_indexes) != list(range(len(scheduled_cells))):
        raise ValueError("clean campaign schedule indexes are not unique and contiguous")

    expected_cells: set[tuple[Any, ...]] = set()
    scheduled_config_repetitions: set[tuple[Any, ...]] = set()
    for cell in scheduled_cells:
        config = cell.get("config")
        if not isinstance(config, dict):
            raise ValueError("scheduled cell has no config object")
        repetition = _integer(cell.get("repetition"), "scheduled_cells.repetition")
        if repetition not in range(expected_repetitions):
            raise ValueError(f"scheduled cell has unexpected repetition={repetition}")
        identity = (
            *_scheduled_config_identity(rq, config),
            repetition,
            _integer(cell.get("schedule_index"), "scheduled_cells.schedule_index"),
        )
        if identity in expected_cells:
            raise ValueError(f"manifest contains duplicate scheduled cell {identity!r}")
        expected_cells.add(identity)
        scheduled_config_repetitions.add((*identity[:-2], repetition))

    requested_config_repetitions = {
        (*_scheduled_config_identity(rq, config), repetition)
        for config in campaign_configs
        for repetition in range(expected_repetitions)
    }
    if scheduled_config_repetitions != requested_config_repetitions:
        missing = sorted(
            requested_config_repetitions - scheduled_config_repetitions, key=repr
        )
        extra = sorted(
            scheduled_config_repetitions - requested_config_repetitions, key=repr
        )
        raise ValueError(
            "clean manifest schedule does not match requested configs: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    observed_cells = {
        (
            *_observed_config_identity(rq, row),
            int(row.repetition),
            int(row.schedule_index),
        )
        for row in run_summary.itertuples(index=False)
    }
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells, key=repr)
        extra = sorted(observed_cells - expected_cells, key=repr)
        raise ValueError(
            "clean campaign raw/manifest cell mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid manifest JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"manifest at {path} is not a JSON object")
    return value


def validate_manifest_source_hashes(
    manifest: dict[str, Any], *, repo_root: Path = REPO_ROOT
) -> None:
    """Verify that every source recorded before the campaign is unchanged."""
    recorded = manifest.get("sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("clean campaign manifest has no source SHA-256 hashes")
    for source_name, expected in recorded.items():
        if not isinstance(source_name, str) or not source_name:
            raise ValueError("clean campaign manifest has an invalid source path")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError(
                f"clean campaign manifest has an invalid SHA-256 for {source_name!r}"
            )
        source_path = Path(source_name)
        if not source_path.is_absolute():
            source_path = repo_root / source_path
        if not source_path.is_file():
            raise ValueError(f"manifest source not found: {source_path}")
        observed = _sha256_file(source_path)
        if observed != expected:
            raise ValueError(
                f"manifest source hash mismatch for {source_name}: "
                f"expected {expected}, observed {observed}"
            )


def _completion_file(
    completion: dict[str, Any], completion_path: Path, label: str
) -> Path:
    files = completion.get("files")
    if not isinstance(files, dict) or not isinstance(files.get(label), dict):
        raise ValueError(f"clean campaign completion has no files.{label} record")
    record = files[label]
    relative_path = record.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"clean campaign completion has no path for {label}")
    path = (completion_path.parent / relative_path).resolve()
    if not path.is_file():
        raise ValueError(f"clean campaign completion file not found: {path}")
    expected_size = _integer(record.get("size_bytes"), f"files.{label}.size_bytes")
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"completion size mismatch for {label}: "
            f"expected {expected_size}, observed {path.stat().st_size}"
        )
    expected_hash = record.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"clean campaign completion has invalid SHA-256 for {label}")
    observed_hash = _sha256_file(path)
    if observed_hash != expected_hash:
        raise ValueError(
            f"completion hash mismatch for {label}: "
            f"expected {expected_hash}, observed {observed_hash}"
        )
    return path


def validate_clean_completion(
    completion: dict[str, Any],
    *,
    completion_path: Path,
    raw_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    run_summary: pd.DataFrame,
) -> Path:
    """Require a successful final record and verify all bound campaign files."""
    if int(completion.get("schema_version", 0)) != 1:
        raise ValueError("clean campaign completion must have schema_version=1")
    run_id = str(manifest.get("run_id", ""))
    if str(completion.get("run_id", "")) != run_id:
        raise ValueError("clean campaign completion run_id does not match manifest")
    if completion.get("status") != "complete" or _integer(
        completion.get("exit_code"), "completion.exit_code"
    ) != 0:
        raise ValueError(
            f"clean campaign completion is not successful: "
            f"status={completion.get('status')!r}, "
            f"exit_code={completion.get('exit_code')!r}"
        )
    scheduled = len(manifest.get("scheduled_cells", []))
    count_fields = {
        "scheduled_cell_count": scheduled,
        "attempted_cell_count": scheduled,
        "successful_cell_count": scheduled,
        "failure_count": 0,
    }
    for field, expected in count_fields.items():
        observed = _integer(completion.get(field), f"completion.{field}")
        if observed != expected:
            raise ValueError(
                f"clean campaign completion {field}={observed}; expected {expected}"
            )
    observed_run_ids = {str(value) for value in run_summary["run_id"]}
    if observed_run_ids != {run_id}:
        raise ValueError("clean campaign completion/raw run_id mismatch")

    bound_raw = _completion_file(completion, completion_path, "raw_jsonl")
    bound_system = _completion_file(completion, completion_path, "system_info")
    bound_manifest = _completion_file(
        completion, completion_path, "measurement_manifest"
    )
    _completion_file(completion, completion_path, "runner_log")
    if bound_raw != raw_path.resolve():
        raise ValueError("completion raw_jsonl path does not match requested raw file")
    if bound_manifest != manifest_path.resolve():
        raise ValueError("completion manifest path does not match requested manifest")
    with raw_path.open("r", encoding="utf-8") as raw_fh:
        record_count = sum(1 for line in raw_fh if line.strip())
    expected_records = _integer(
        completion.get("raw_record_count"), "completion.raw_record_count"
    )
    if record_count != expected_records:
        raise ValueError(
            f"completion raw record count mismatch: "
            f"expected {expected_records}, observed {record_count}"
        )
    try:
        system_info = json.loads(bound_system.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid bound system-info JSON: {exc}") from exc
    if not isinstance(system_info, dict) or not system_info:
        raise ValueError("bound system-info file is not a nonempty JSON object")
    return bound_system


def _provenance_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _write_analysis_provenance(
    *,
    path: Path,
    rq: int,
    run_id: str,
    raw_path: Path,
    system_info_path: Path,
    manifest_path: Path,
    completion_path: Path,
    runner_log_path: Path,
    run_summary_path: Path,
    config_summary_path: Path,
) -> None:
    files = {
        "raw_jsonl": raw_path,
        "system_info": system_info_path,
        "measurement_manifest": manifest_path,
        "campaign_completion": completion_path,
        "runner_log": runner_log_path,
        "aggregation_script": Path(__file__).resolve(),
        "run_summary": run_summary_path,
        "config_summary": config_summary_path,
    }
    provenance = {
        "schema_version": 1,
        "run_id": run_id,
        "rq": rq,
        "experimental_unit": "campaign run_id, configuration, repetition",
        "expected_repetitions": EXPECTED_REPETITIONS,
        "across_run_summary": "median, minimum, maximum",
        "files": {
            label: {
                "path": _provenance_path(file_path),
                "sha256": _sha256_file(file_path),
                "size_bytes": file_path.stat().st_size,
            }
            for label, file_path in files.items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Two-level aggregation for one clean five-repetition RQ campaign."
    )
    parser.add_argument("--raw-file", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Campaign manifest. By default, use "
            "<raw-file-stem>_artifacts/manifest.json next to the raw file."
        ),
    )
    parser.add_argument(
        "--completion",
        type=Path,
        help="Final campaign completion record; defaults next to the manifest.",
    )
    parser.add_argument("--rq", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--summary", type=Path, default=Path("data/summaries"))
    parser.add_argument(
        "--label",
        default="clean",
        help="Output suffix; default 'clean' avoids overwriting legacy summary CSVs.",
    )
    args = parser.parse_args(argv)

    manifest_path = args.manifest or (
        args.raw_file.parent / f"{args.raw_file.stem}_artifacts" / "manifest.json"
    )
    if not manifest_path.is_file():
        raise ValueError(f"clean campaign manifest not found: {manifest_path}")
    manifest = _read_manifest(manifest_path)
    validate_manifest_source_hashes(manifest)
    records = _read_jsonl(args.raw_file)
    run_summary, config_summary = aggregate_clean_campaign(
        records,
        args.rq,
        expected_repetitions=EXPECTED_REPETITIONS,
    )
    validate_clean_manifest(
        manifest,
        args.rq,
        run_summary,
        expected_repetitions=EXPECTED_REPETITIONS,
    )
    completion_path = args.completion or manifest_path.parent / "completion.json"
    if not completion_path.is_file():
        raise ValueError(f"clean campaign completion not found: {completion_path}")
    completion = _read_manifest(completion_path)
    system_info_path = validate_clean_completion(
        completion,
        completion_path=completion_path,
        raw_path=args.raw_file,
        manifest_path=manifest_path,
        manifest=manifest,
        run_summary=run_summary,
    )
    runner_log_path = _completion_file(
        completion, completion_path, "runner_log"
    )
    args.summary.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.label}" if args.label else ""
    run_path = args.summary / f"rq{args.rq}_run_summary{suffix}.csv"
    config_path = args.summary / f"rq{args.rq}_summary{suffix}.csv"
    run_summary.to_csv(run_path, index=False)
    config_summary.to_csv(config_path, index=False)
    provenance_path = args.summary / f"rq{args.rq}_provenance{suffix}.json"
    _write_analysis_provenance(
        path=provenance_path,
        rq=args.rq,
        run_id=str(manifest["run_id"]),
        raw_path=args.raw_file,
        system_info_path=system_info_path,
        manifest_path=manifest_path,
        completion_path=completion_path,
        runner_log_path=runner_log_path,
        run_summary_path=run_path,
        config_summary_path=config_path,
    )
    print(
        f"VALID: RQ{args.rq} clean campaign; "
        f"{len(run_summary)} run rows; {len(config_summary)} config rows"
    )
    print(f"wrote {run_path}")
    print(f"wrote {config_path}")
    print(f"wrote {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
