"""Generate LaTeX tables from the clean five-restart RQ summaries.

The inputs are the configuration-level files written by
``analysis.aggregate_clean``. Every numeric table cell is therefore a
run-level statistic summarized across five independent restarts as
``median [minimum--maximum]``. Legacy rep1/rep2, cross-day divergence, and
cross-phase data are deliberately outside this generator.

CLI::

    python -m analysis.tables --summary data/summaries \
        --output paper/tables --label c4_clean
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

PROGRAMS = ("l2_forward", "l3_lpm", "l3_lpm_acl", "l3_lpm_int")
SIZES = (64, 256, 1500)
LOADS = (1, 25, 45)
RQ2_NS = (1, 2, 4, 8)
RQ2_KS = (10, 100, 1000)
RQ3_NS = (2, 3)
RQ3_LOADS = (0, 25, 45)
EXPECTED_REPETITIONS = 5
RQ4_CONFIGS = (
    ("l3_lpm", "single_switch", 1, 1),
    ("l3_lpm", "single_switch", 1, 45),
    ("l3_lpm", "linear_n", 4, 1),
    ("l3_lpm", "linear_n", 4, 45),
    ("l3_lpm", "linear_n", 8, 1),
    ("l3_lpm", "linear_n", 8, 45),
    ("l3_lpm_acl", "linear_n", 4, 1),
    ("l3_lpm_int", "linear_n", 4, 1),
)


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body + "\n", encoding="utf-8")


def _latex_texttt(value: Any) -> str:
    return "\\texttt{" + str(value).replace("_", "\\_") + "}"


def _require_columns(df: pd.DataFrame, source: str, columns: set[str]) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError(f"{source} is empty")
    repetitions = pd.to_numeric(df["n_reps"], errors="coerce")
    if repetitions.isna().any() or not (repetitions == EXPECTED_REPETITIONS).all():
        observed = sorted({str(value) for value in df["n_reps"].tolist()})
        raise ValueError(
            f"{source} must contain n_reps={EXPECTED_REPETITIONS} for every row; "
            f"observed {observed}"
        )


def _one_row(df: pd.DataFrame, source: str, **criteria: Any) -> pd.Series:
    selected = df
    for column, expected in criteria.items():
        selected = selected[selected[column] == expected]
    if len(selected) != 1:
        rendered = ", ".join(f"{key}={value!r}" for key, value in criteria.items())
        raise ValueError(
            f"{source} must contain exactly one row for {rendered}; found {len(selected)}"
        )
    return selected.iloc[0]


def _format_range(
    row: pd.Series,
    metric: str,
    decimals: int,
    *,
    scale: float = 1.0,
) -> str:
    values = [
        float(row[f"{metric}_{suffix}"]) / scale
        for suffix in ("median", "min", "max")
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite {metric} range: {values}")
    median, minimum, maximum = values
    if minimum > median or median > maximum:
        raise ValueError(
            f"invalid {metric} range: median={median}, minimum={minimum}, maximum={maximum}"
        )
    rendered = [f"{value:.{decimals}f}" for value in values]
    return f"{rendered[0]} [{rendered[1]}\\,--\\,{rendered[2]}]"


def tab_rq1_main_matrix(out_dir: Path, rq1: pd.DataFrame) -> None:
    """Write RQ1 run-median latency with its five-restart range."""
    source = "RQ1 clean summary"
    required = {
        "p4_program",
        "packet_size_bytes",
        "background_load_mbps",
        "cold_idle_reference",
        "n_reps",
        "median_us_median",
        "median_us_min",
        "median_us_max",
    }
    _require_columns(rq1, source, required)

    body = [
        "\\begin{tabular}{ll" + "l" * len(LOADS) + "}",
        "\\toprule",
        "\\textbf{Program} & \\textbf{Size (B)} & "
        + " & ".join(f"\\textbf{{{load}\\,Mbps}}" for load in LOADS)
        + " \\\\",
        "\\midrule",
    ]
    for program in PROGRAMS:
        for size in SIZES:
            cells = []
            for load in LOADS:
                row = _one_row(
                    rq1,
                    source,
                    p4_program=program,
                    packet_size_bytes=size,
                    background_load_mbps=load,
                    cold_idle_reference=False,
                )
                cells.append(_format_range(row, "median_us", 1))
            body.append(
                f"{_latex_texttt(program)} & {size} & "
                + " & ".join(cells)
                + " \\\\"
            )

    cold = _one_row(
        rq1,
        source,
        p4_program="l3_lpm",
        packet_size_bytes=256,
        background_load_mbps=0,
        cold_idle_reference=True,
    )
    body.extend(
        [
            "\\midrule",
            "\\multicolumn{2}{l}{\\textit{cold-idle reference}} & "
            + _format_range(cold, "median_us", 1)
            + " & --- & --- \\\\ ",
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    _write(out_dir / "tab_rq1_main_matrix.tex", "\n".join(body))


def tab_rq2_n_k_grid(out_dir: Path, rq2: pd.DataFrame) -> None:
    """Write INSERT/READ blocks of wall-clock run medians and ranges."""
    source = "RQ2 clean summary"
    required = {
        "n_switches",
        "n_entries_per_switch",
        "operation",
        "mode",
        "n_reps",
        "wall_clock_s_median",
        "wall_clock_s_min",
        "wall_clock_s_max",
    }
    _require_columns(rq2, source, required)

    body = ["\\begin{tabular}{rrlll}", "\\toprule"]
    for block_index, operation in enumerate(("insert", "read")):
        if block_index:
            body.append("\\midrule")
        body.extend(
            [
                f"\\multicolumn{{5}}{{l}}{{\\textbf{{{operation.upper()}}}}} \\\\ ",
                "\\textbf{$N$} & \\textbf{Mode} & "
                + " & ".join(f"\\textbf{{$K={k}$}}" for k in RQ2_KS)
                + " \\\\",
                "\\cmidrule(lr){1-5}",
            ]
        )
        for n_switches in RQ2_NS:
            modes = ("sync",) if n_switches == 1 else ("sync", "async")
            for mode in modes:
                cells = []
                for n_entries in RQ2_KS:
                    row = _one_row(
                        rq2,
                        source,
                        n_switches=n_switches,
                        n_entries_per_switch=n_entries,
                        operation=operation,
                        mode=mode,
                    )
                    cells.append(_format_range(row, "wall_clock_s", 4))
                body.append(
                    f"{n_switches} & {mode} & " + " & ".join(cells) + " \\\\"
                )
    body.extend(["\\bottomrule", "\\end{tabular}"])
    _write(out_dir / "tab_rq2_n_k_grid.tex", "\n".join(body))


def tab_rq3_drift_summary(out_dir: Path, rq3: pd.DataFrame) -> None:
    """Write signed, absolute, and within-run-IQR drift summaries."""
    source = "RQ3 clean summary"
    required = {"n_switches", "background_load_mbps", "n_reps"}
    for metric in ("median_us", "abs_median_us", "iqr_us"):
        required.update(f"{metric}_{suffix}" for suffix in ("median", "min", "max"))
    _require_columns(rq3, source, required)

    body = [
        "\\begin{tabular}{rrlll}",
        "\\toprule",
        "\\textbf{$N$} & \\textbf{Load (Mbps)} & \\textbf{Signed median ($\\mu$s)} & "
        "\\textbf{Absolute median ($\\mu$s)} & \\textbf{Within-run IQR ($\\mu$s)} \\\\",
        "\\midrule",
    ]
    for n_switches in RQ3_NS:
        for load in RQ3_LOADS:
            row = _one_row(
                rq3,
                source,
                n_switches=n_switches,
                background_load_mbps=load,
            )
            body.append(
                f"{n_switches} & {load} & {_format_range(row, 'median_us', 1)} & "
                f"{_format_range(row, 'abs_median_us', 1)} & "
                f"{_format_range(row, 'iqr_us', 1)} \\\\"
            )
    body.extend(["\\bottomrule", "\\end{tabular}"])
    _write(out_dir / "tab_rq3_drift_summary.tex", "\n".join(body))


def tab_rq4_resource_summary(out_dir: Path, rq4: pd.DataFrame) -> None:
    """Write selected RQ4 run-level resource statistics across restarts."""
    source = "RQ4 clean summary"
    required = {
        "p4_program",
        "topology",
        "n_switches",
        "background_load_mbps",
        "source_workload_type",
        "metric",
        "n_reps",
    }
    for metric in ("mean", "p95", "max"):
        required.update(f"{metric}_{suffix}" for suffix in ("median", "min", "max"))
    _require_columns(rq4, source, required)
    resource = rq4[rq4["source_workload_type"] == "resource_only"].copy()
    if resource.empty:
        raise ValueError("RQ4 clean summary has no resource_only rows")

    config_columns = ["p4_program", "topology", "n_switches", "background_load_mbps"]
    configs = sorted(
        set(resource[config_columns].itertuples(index=False, name=None)),
        key=lambda item: (str(item[0]), str(item[1]), int(item[2]), float(item[3])),
    )
    expected_configs = set(RQ4_CONFIGS)
    actual_configs = set(configs)
    if actual_configs != expected_configs:
        missing = sorted(expected_configs - actual_configs)
        extra = sorted(actual_configs - expected_configs)
        raise ValueError(
            f"RQ4 clean summary configuration mismatch: missing={missing}, extra={extra}"
        )
    body = [
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "\\textbf{Configuration} & \\textbf{CPU mean (\\%)} & "
        "\\textbf{CPU p95 (\\%)} & \\textbf{RSS max (MB)} & "
        "\\textbf{Aggregate RX mean (pps)} \\\\",
        "\\midrule",
    ]
    for program, topology, n_switches, load in RQ4_CONFIGS:
        criteria = {
            "p4_program": program,
            "topology": topology,
            "n_switches": n_switches,
            "background_load_mbps": load,
            "source_workload_type": "resource_only",
        }
        cpu = _one_row(resource, source, **criteria, metric="cpu_percent_per_bmv2")
        rss = _one_row(resource, source, **criteria, metric="rss_per_bmv2_bytes")
        rx = _one_row(resource, source, **criteria, metric="net_io_pps_per_iface")
        label = (
            f"{_latex_texttt(program)} {_latex_texttt(topology)} "
            f"$N{{=}}{int(n_switches)}$ {float(load):g}\\,Mbps"
        )
        body.append(
            f"{label} & {_format_range(cpu, 'mean', 1)} & "
            f"{_format_range(cpu, 'p95', 1)} & "
            f"{_format_range(rss, 'max', 1, scale=1_000_000.0)} & "
            f"{_format_range(rx, 'mean', 1)} \\\\"
        )
    body.extend(["\\bottomrule", "\\end{tabular}"])
    _write(out_dir / "tab_rq4_resource_summary.tex", "\n".join(body))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from clean five-restart summary CSVs."
    )
    parser.add_argument("--summary", type=Path, default=Path("data/summaries"))
    parser.add_argument("--output", type=Path, default=Path("paper/tables"))
    parser.add_argument(
        "--label",
        default="c4_clean",
        help="summary filename label (default: c4_clean)",
    )
    args = parser.parse_args(argv)

    if not args.label or Path(args.label).name != args.label:
        parser.error("--label must be a non-empty filename component")
    paths = {
        rq: args.summary / f"rq{rq}_summary_{args.label}.csv" for rq in range(1, 5)
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        parser.error(
            "missing clean summary file(s): " + ", ".join(str(path) for path in missing)
        )

    summaries = {rq: pd.read_csv(path) for rq, path in paths.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"writing clean five-restart tables to {args.output} (label={args.label})")
    tab_rq1_main_matrix(args.output, summaries[1])
    tab_rq2_n_k_grid(args.output, summaries[2])
    tab_rq3_drift_summary(args.output, summaries[3])
    tab_rq4_resource_summary(args.output, summaries[4])
    print("  produced 4 LaTeX snippets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
