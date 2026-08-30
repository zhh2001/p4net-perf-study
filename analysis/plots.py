"""Generate paper figures from clean, independently restarted campaigns.

The active plotting pipeline reads one configuration-level summary for each
research question. These files are produced by :mod:`analysis.aggregate_clean`
and contain a median, minimum, and maximum across five independently restarted
runs for every run-level statistic. Figure markers/bars show the across-run
median and whiskers show the observed run-level minimum--maximum range. The
whiskers are descriptive ranges, not confidence intervals.

CLI::

    python -m analysis.plots \
        --summary data/summaries \
        --label c4_clean \
        --output paper/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_REPETITIONS = 5

# Paper-friendly style. Times New Roman matches the manuscript's Times-based
# typography; STIX supplies compatible mathematical glyphs. ColorBrewer Set1
# is used for qualitative palettes and viridis for sequential displays.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 100,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)

PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
PROGRAMS = ("l2_forward", "l3_lpm", "l3_lpm_acl", "l3_lpm_int")
PROGRAM_COLORS = {
    "l2_forward": PALETTE[1],
    "l3_lpm": PALETTE[2],
    "l3_lpm_acl": PALETTE[3],
    "l3_lpm_int": PALETTE[4],
}
LOAD_COLORS = {1: PALETTE[1], 45: PALETTE[0]}

FIGURE_NAMES = (
    "fig_rq1_latency_by_load",
    "fig_rq1_packet_size_independence",
    "fig_rq1_cold_warm_regime",
    "fig_rq2_async_vs_sync_speedup",
    "fig_rq2_scaling_curves",
    "fig_rq3_drift_envelope",
    "fig_rq4_resource_scaling",
    "fig_rq4_pipeline_overhead",
)


def _save(fig: matplotlib.figure.Figure, out_dir: Path, name: str) -> None:
    """Write ``out_dir/<name>.pdf`` and ``out_dir/<name>.png`` at 300 dpi."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=300)
    plt.close(fig)


def _require_columns(df: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{label} clean summary is missing columns: {missing}")
    if df.empty:
        raise ValueError(f"{label} clean summary is empty")
    repetitions = pd.to_numeric(df["n_reps"], errors="coerce")
    if repetitions.isna().any() or not (repetitions == EXPECTED_REPETITIONS).all():
        observed = sorted(repetitions.dropna().unique().tolist())
        raise ValueError(
            f"{label} clean summary must contain exactly {EXPECTED_REPETITIONS} "
            f"restarts per row; observed {observed}"
        )


def _cold_idle_mask(df: pd.DataFrame) -> pd.Series:
    values = df["cold_idle_reference"]
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if normalized.isna().any():
        invalid = sorted(values[normalized.isna()].astype(str).unique().tolist())
        raise ValueError(f"invalid cold_idle_reference values: {invalid}")
    return normalized.astype(bool)


def _range_arrays(
    frame: pd.DataFrame, statistic: str, *, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = frame[f"{statistic}_median"].to_numpy(dtype=float) / scale
    minimum = frame[f"{statistic}_min"].to_numpy(dtype=float) / scale
    maximum = frame[f"{statistic}_max"].to_numpy(dtype=float) / scale
    if not (
        np.isfinite(median).all()
        and np.isfinite(minimum).all()
        and np.isfinite(maximum).all()
    ):
        raise ValueError(f"{statistic} contains non-finite clean-summary values")
    if np.any(minimum > median) or np.any(median > maximum):
        raise ValueError(f"{statistic} does not satisfy minimum <= median <= maximum")
    return median, minimum, maximum


def _errorbars(
    median: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
) -> np.ndarray:
    return np.vstack((median - minimum, maximum - median))


def _one_row(frame: pd.DataFrame, description: str) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f"expected one {description} row, found {len(frame)}")
    return frame.iloc[0]


def _row_range(row: pd.Series, statistic: str) -> tuple[float, float, float]:
    median = float(row[f"{statistic}_median"])
    minimum = float(row[f"{statistic}_min"])
    maximum = float(row[f"{statistic}_max"])
    if not np.isfinite([median, minimum, maximum]).all():
        raise ValueError(f"{statistic} contains non-finite clean-summary values")
    if minimum > median or median > maximum:
        raise ValueError(f"{statistic} does not satisfy minimum <= median <= maximum")
    return median, minimum, maximum


# ---------------------------------------------------------------------------
# RQ1 figures.
# ---------------------------------------------------------------------------


def fig_rq1_latency_by_load(out_dir: Path, rq1: pd.DataFrame) -> None:
    """Plot run-level latency medians across loads for 256 B and 1500 B probes."""
    df = rq1[~_cold_idle_mask(rq1)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    panel_sizes = (256, 1500)
    for ax, size in zip(axes, panel_sizes, strict=True):
        panel = df[df["packet_size_bytes"] == size]
        for program in PROGRAMS:
            rows = panel[panel["p4_program"] == program].sort_values(
                "background_load_mbps"
            )
            if rows.empty:
                raise ValueError(f"RQ1 has no {program}, {size} B rows")
            median, minimum, maximum = _range_arrays(rows, "median_us")
            ax.errorbar(
                rows["background_load_mbps"].to_numpy(),
                median,
                yerr=_errorbars(median, minimum, maximum),
                marker="o",
                capsize=3,
                label=program,
                color=PROGRAM_COLORS[program],
                linewidth=1.5,
                markersize=6,
            )
        ax.set_xticks([1, 25, 45])
        ax.set_xlabel("Continuous-carrier load (Mbps)")
        ax.set_title(f"{size} B probes")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Across-run median of run-level latency medians (μs)")
    axes[0].legend(loc="lower right", framealpha=0.95)
    fig.suptitle("RQ1 latency across background loads")
    _save(fig, out_dir, "fig_rq1_latency_by_load")


def fig_rq1_latency_by_packet_size(out_dir: Path, rq1: pd.DataFrame) -> None:
    """Plot run-level latency medians by packet size at the 1 Mbps load."""
    cold_idle = _cold_idle_mask(rq1)
    df = rq1[(~cold_idle) & (rq1["background_load_mbps"] == 1)].copy()
    sizes = (64, 256, 1500)
    width = 0.18
    x = np.arange(len(sizes))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for index, program in enumerate(PROGRAMS):
        medians: list[float] = []
        minima: list[float] = []
        maxima: list[float] = []
        for size in sizes:
            row = _one_row(
                df[(df["p4_program"] == program) & (df["packet_size_bytes"] == size)],
                f"RQ1 {program}, {size} B, 1 Mbps",
            )
            median, minimum, maximum = _row_range(row, "median_us")
            medians.append(median)
            minima.append(minimum)
            maxima.append(maximum)
        median_array = np.asarray(medians)
        minimum_array = np.asarray(minima)
        maximum_array = np.asarray(maxima)
        positions = x + (index - (len(PROGRAMS) - 1) / 2) * width
        ax.bar(
            positions,
            median_array,
            width,
            yerr=_errorbars(median_array, minimum_array, maximum_array),
            capsize=3,
            label=program,
            color=PROGRAM_COLORS[program],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.set_xlabel("Packet size (B)")
    ax.set_ylabel("Across-run median of run-level latency medians (μs)")
    ax.set_title("RQ1 latency by packet size at 1 Mbps")
    ax.legend(loc="upper right", framealpha=0.95, ncol=2)
    ax.set_ylim(bottom=0)
    _save(fig, out_dir, "fig_rq1_packet_size_independence")


def fig_rq1_cold_warm_regime(out_dir: Path, rq1: pd.DataFrame) -> None:
    """Compare clean l3_lpm/256 B cold-idle and continuous-carrier runs."""
    cold_idle = _cold_idle_mask(rq1)
    base = rq1[(rq1["p4_program"] == "l3_lpm") & (rq1["packet_size_bytes"] == 256)]
    base_cold_idle = cold_idle.loc[base.index]
    selections = (
        ("Cold-idle\n(no carrier)", base[base_cold_idle], PALETTE[0]),
        (
            "Continuous carrier\n(1 Mbps)",
            base[(~base_cold_idle) & (base["background_load_mbps"] == 1)],
            PALETTE[1],
        ),
        (
            "Continuous carrier\n(25 Mbps)",
            base[(~base_cold_idle) & (base["background_load_mbps"] == 25)],
            PALETTE[2],
        ),
        (
            "Continuous carrier\n(45 Mbps)",
            base[(~base_cold_idle) & (base["background_load_mbps"] == 45)],
            PALETTE[2],
        ),
    )
    labels: list[str] = []
    colors: list[str] = []
    medians: list[float] = []
    minima: list[float] = []
    maxima: list[float] = []
    for label, selection, color in selections:
        row = _one_row(selection, f"RQ1 {label.replace(chr(10), ' ')}")
        median, minimum, maximum = _row_range(row, "median_us")
        labels.append(label)
        colors.append(color)
        medians.append(median)
        minima.append(minimum)
        maxima.append(maximum)
    median_array = np.asarray(medians)
    minimum_array = np.asarray(minima)
    maximum_array = np.asarray(maxima)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(
        np.arange(len(labels)),
        median_array,
        yerr=_errorbars(median_array, minimum_array, maximum_array),
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Across-run median of run-level latency medians (μs)")
    ax.set_title("RQ1 cold-idle and continuous-carrier measurements")
    ax.set_ylim(bottom=0)
    _save(fig, out_dir, "fig_rq1_cold_warm_regime")


# ---------------------------------------------------------------------------
# RQ2 figures.
# ---------------------------------------------------------------------------


def _build_rq2_ratio_grids(
    rq2: pd.DataFrame, operation: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    """Build ratio-of-medians and conservative observed-range envelopes.

    The envelope is ``sync_min / async_max`` to ``sync_max / async_min``.
    It is not a confidence interval and does not pair repetition identifiers.
    """
    switch_counts = sorted(int(value) for value in rq2["n_switches"].unique())
    entry_counts = sorted(int(value) for value in rq2["n_entries_per_switch"].unique())
    central = np.full((len(switch_counts), len(entry_counts)), np.nan)
    lower = np.full_like(central, np.nan)
    upper = np.full_like(central, np.nan)
    for row_index, n_switches in enumerate(switch_counts):
        for column_index, n_entries in enumerate(entry_counts):
            base = rq2[
                (rq2["n_switches"] == n_switches)
                & (rq2["n_entries_per_switch"] == n_entries)
                & (rq2["operation"] == operation)
            ]
            sync = base[base["mode"] == "sync"]
            asynchronous = base[base["mode"] == "async"]
            if sync.empty or asynchronous.empty:
                continue
            sync_row = _one_row(sync, f"RQ2 {operation} sync N={n_switches}, K={n_entries}")
            async_row = _one_row(
                asynchronous,
                f"RQ2 {operation} async N={n_switches}, K={n_entries}",
            )
            sync_median, sync_minimum, sync_maximum = _row_range(
                sync_row, "wall_clock_s"
            )
            async_median, async_minimum, async_maximum = _row_range(
                async_row, "wall_clock_s"
            )
            if min(sync_minimum, async_minimum) <= 0:
                raise ValueError("RQ2 wall-clock ranges must be strictly positive")
            central[row_index, column_index] = sync_median / async_median
            lower[row_index, column_index] = sync_minimum / async_maximum
            upper[row_index, column_index] = sync_maximum / async_minimum
    return central, lower, upper, switch_counts, entry_counts


def fig_rq2_async_vs_sync_speedup(out_dir: Path, rq2: pd.DataFrame) -> None:
    """Plot sync/async ratio-of-medians with conservative observed envelopes."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 8.6))
    for axis_index, (ax, operation) in enumerate(
        zip(axes, ("insert", "read"), strict=True)
    ):
        central, lower, upper, switch_counts, entry_counts = _build_rq2_ratio_grids(
            rq2, operation
        )
        finite = central[np.isfinite(central)]
        if finite.size == 0:
            raise ValueError(f"RQ2 has no paired sync/async {operation} configurations")
        norm = matplotlib.colors.TwoSlopeNorm(
            vmin=min(0.5, float(finite.min())),
            vcenter=1.0,
            vmax=max(1.5, float(finite.max())),
        )
        image = ax.imshow(
            central,
            aspect="auto",
            cmap="RdYlGn",
            norm=norm,
            origin="lower",
        )
        ax.set_xticks(range(len(entry_counts)))
        ax.set_xticklabels([str(value) for value in entry_counts])
        ax.set_yticks(range(len(switch_counts)))
        ax.set_yticklabels([f"N={value}" for value in switch_counts])
        if axis_index == len(axes) - 1:
            ax.set_xlabel("K (entries per switch)")
        ax.set_title(f"{operation.capitalize()} operations")
        for row_index in range(central.shape[0]):
            for column_index in range(central.shape[1]):
                value = central[row_index, column_index]
                if np.isnan(value):
                    annotation = "—"
                    color = "grey"
                    weight = "normal"
                else:
                    annotation = (
                        f"{value:.2f}×\n"
                        f"[{lower[row_index, column_index]:.2f}–"
                        f"{upper[row_index, column_index]:.2f}]"
                    )
                    color = "black"
                    weight = "bold"
                ax.text(
                    column_index,
                    row_index,
                    annotation,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                    weight=weight,
                )
        fig.colorbar(image, ax=ax, pad=0.02, label="sync / async wall-clock ratio")
    fig.suptitle("RQ2 synchronous-to-asynchronous wall-clock ratios")
    fig.text(
        0.5,
        0.018,
        "Brackets are conservative observed-range envelopes, not confidence "
        "intervals or paired-run ratios.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.10, right=0.91, bottom=0.10, top=0.93, hspace=0.32)
    _save(fig, out_dir, "fig_rq2_async_vs_sync_speedup")


def fig_rq2_scaling_curves(out_dir: Path, rq2: pd.DataFrame) -> None:
    """Plot insert wall-clock time versus K with five-restart ranges."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    inserts = rq2[rq2["operation"] == "insert"].copy()
    switch_counts = sorted(int(value) for value in inserts["n_switches"].unique())
    colors = matplotlib.colormaps["viridis"](
        np.linspace(0.2, 0.8, len(switch_counts))
    )
    for color, n_switches in zip(colors, switch_counts, strict=True):
        for mode, line_style, marker in (("sync", "-", "o"), ("async", "--", "s")):
            rows = inserts[
                (inserts["n_switches"] == n_switches) & (inserts["mode"] == mode)
            ].sort_values("n_entries_per_switch")
            if rows.empty:
                continue
            median, minimum, maximum = _range_arrays(rows, "wall_clock_s")
            ax.errorbar(
                rows["n_entries_per_switch"].to_numpy(),
                median,
                yerr=_errorbars(median, minimum, maximum),
                marker=marker,
                linestyle=line_style,
                capsize=3,
                color=color,
                label=f"N={n_switches} {mode}",
                markersize=5,
                linewidth=1.5,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K (entries per switch, log scale)")
    ax.set_ylabel("Across-run median wall-clock time (s, log scale)")
    ax.set_title("RQ2 insert wall-clock time across switch and entry counts")
    ax.legend(loc="upper left", ncol=2, framealpha=0.95, fontsize=8)
    _save(fig, out_dir, "fig_rq2_scaling_curves")


# ---------------------------------------------------------------------------
# RQ3 figures.
# ---------------------------------------------------------------------------


def fig_rq3_drift_envelope(out_dir: Path, rq3: pd.DataFrame) -> None:
    """Plot absolute run medians and within-run IQRs across five restarts."""
    rows = rq3.sort_values(["n_switches", "background_load_mbps"]).copy()
    labels = [
        f"N={int(n_switches)}\n{int(load)} Mbps"
        for n_switches, load in zip(
            rows["n_switches"], rows["background_load_mbps"], strict=True
        )
    ]
    x = np.arange(len(labels))
    absolute_median, absolute_minimum, absolute_maximum = _range_arrays(
        rows, "abs_median_us"
    )
    iqr_median, iqr_minimum, iqr_maximum = _range_arrays(rows, "iqr_us")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].errorbar(
        x,
        absolute_median,
        yerr=_errorbars(absolute_median, absolute_minimum, absolute_maximum),
        marker="o",
        linestyle="none",
        capsize=4,
        color=PALETTE[0],
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Across-run median of absolute run medians (μs)")
    axes[0].set_title("Absolute packet-drift median")

    axes[1].errorbar(
        x,
        iqr_median,
        yerr=_errorbars(iqr_median, iqr_minimum, iqr_maximum),
        marker="s",
        linestyle="none",
        capsize=4,
        color=PALETTE[1],
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Across-run median of within-run IQRs (μs)")
    axes[1].set_title("Within-run packet-drift IQR")
    fig.suptitle("RQ3 INT timestamp drift across configurations")
    _save(fig, out_dir, "fig_rq3_drift_envelope")


# ---------------------------------------------------------------------------
# RQ4 figures.
# ---------------------------------------------------------------------------


def fig_rq4_resource_scaling(out_dir: Path, rq4: pd.DataFrame) -> None:
    """Plot BMv2 CPU mean and aggregate RSS peak across switch counts."""
    data = rq4[
        (rq4["source_workload_type"] == "resource_only")
        & (rq4["p4_program"] == "l3_lpm")
    ].copy()
    cpu = data[data["metric"] == "cpu_percent_per_bmv2"]
    rss = data[data["metric"] == "rss_per_bmv2_bytes"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    observed_switch_counts: set[int] = set()
    for load in (1, 45):
        cpu_rows = cpu[cpu["background_load_mbps"] == load].sort_values("n_switches")
        rss_rows = rss[rss["background_load_mbps"] == load].sort_values("n_switches")
        if cpu_rows.empty or rss_rows.empty:
            raise ValueError(f"RQ4 l3_lpm resource scaling is missing the {load} Mbps load")
        observed_switch_counts.update(int(value) for value in cpu_rows["n_switches"])
        cpu_median, cpu_minimum, cpu_maximum = _range_arrays(cpu_rows, "mean")
        axes[0].errorbar(
            cpu_rows["n_switches"].to_numpy(),
            cpu_median,
            yerr=_errorbars(cpu_median, cpu_minimum, cpu_maximum),
            marker="o",
            capsize=3,
            color=LOAD_COLORS[load],
            label=f"{load} Mbps",
            linewidth=2,
            markersize=7,
        )
        rss_median, rss_minimum, rss_maximum = _range_arrays(
            rss_rows, "max", scale=1e6
        )
        axes[1].errorbar(
            rss_rows["n_switches"].to_numpy(),
            rss_median,
            yerr=_errorbars(rss_median, rss_minimum, rss_maximum),
            marker="s",
            capsize=3,
            color=LOAD_COLORS[load],
            label=f"{load} Mbps",
            linewidth=2,
            markersize=7,
        )
    ticks = sorted(observed_switch_counts)
    axes[0].set_xticks(ticks)
    axes[0].set_xlabel("N (switch count)")
    axes[0].set_ylabel("Across-run median of run-level mean BMv2 CPU (%)")
    axes[0].set_title("Aggregate BMv2 CPU")
    axes[0].legend(framealpha=0.95)

    axes[1].set_xticks(ticks)
    axes[1].set_xlabel("N (switch count)")
    axes[1].set_ylabel("Across-run median of run-level peak RSS (MB)")
    axes[1].set_title("Aggregate BMv2 resident memory")
    axes[1].legend(framealpha=0.95)
    fig.suptitle("RQ4 resource measurements across switch counts and loads")
    _save(fig, out_dir, "fig_rq4_resource_scaling")


def fig_rq4_pipeline_overhead(out_dir: Path, rq4: pd.DataFrame) -> None:
    """Plot mean aggregate BMv2 CPU for three pipelines at N=4 and 1 Mbps."""
    data = rq4[
        (rq4["source_workload_type"] == "resource_only")
        & (rq4["metric"] == "cpu_percent_per_bmv2")
        & (rq4["n_switches"] == 4)
        & (rq4["background_load_mbps"] == 1)
    ]
    medians: list[float] = []
    minima: list[float] = []
    maxima: list[float] = []
    programs = ("l3_lpm", "l3_lpm_acl", "l3_lpm_int")
    for program in programs:
        row = _one_row(data[data["p4_program"] == program], f"RQ4 {program}, N=4, 1 Mbps")
        median, minimum, maximum = _row_range(row, "mean")
        medians.append(median)
        minima.append(minimum)
        maxima.append(maximum)
    median_array = np.asarray(medians)
    minimum_array = np.asarray(minima)
    maximum_array = np.asarray(maxima)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        programs,
        median_array,
        yerr=_errorbars(median_array, minimum_array, maximum_array),
        capsize=4,
        color=[PROGRAM_COLORS[program] for program in programs],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Across-run median of run-level mean BMv2 CPU (%)")
    ax.set_title("RQ4 pipeline comparison at N=4 and 1 Mbps")
    ax.set_ylim(bottom=0)
    _save(fig, out_dir, "fig_rq4_pipeline_overhead")


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------


def _summary_path(summary_dir: Path, rq: int, label: str) -> Path:
    return summary_dir / f"rq{rq}_summary_{label}.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate paper figures from clean five-restart summary CSVs."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "data" / "summaries",
    )
    parser.add_argument(
        "--label",
        default="c4_clean",
        help="Suffix in rqN_summary_<label>.csv; default: c4_clean.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "paper" / "figures",
    )
    args = parser.parse_args(argv)
    label = args.label.strip()
    if not label:
        parser.error("--label must be non-empty")

    frames = {
        rq: pd.read_csv(_summary_path(args.summary, rq, label)) for rq in range(1, 5)
    }
    _require_columns(
        frames[1],
        {
            "n_reps",
            "p4_program",
            "packet_size_bytes",
            "background_load_mbps",
            "cold_idle_reference",
            "median_us_median",
            "median_us_min",
            "median_us_max",
        },
        "RQ1",
    )
    _require_columns(
        frames[2],
        {
            "n_reps",
            "n_switches",
            "n_entries_per_switch",
            "operation",
            "mode",
            "wall_clock_s_median",
            "wall_clock_s_min",
            "wall_clock_s_max",
        },
        "RQ2",
    )
    _require_columns(
        frames[3],
        {
            "n_reps",
            "n_switches",
            "background_load_mbps",
            "abs_median_us_median",
            "abs_median_us_min",
            "abs_median_us_max",
            "iqr_us_median",
            "iqr_us_min",
            "iqr_us_max",
        },
        "RQ3",
    )
    _require_columns(
        frames[4],
        {
            "n_reps",
            "p4_program",
            "n_switches",
            "background_load_mbps",
            "source_workload_type",
            "metric",
            "mean_median",
            "mean_min",
            "mean_max",
            "max_median",
            "max_min",
            "max_max",
        },
        "RQ4",
    )

    print(f"writing clean-campaign figures to {args.output}")
    fig_rq1_latency_by_load(args.output, frames[1])
    fig_rq1_latency_by_packet_size(args.output, frames[1])
    fig_rq1_cold_warm_regime(args.output, frames[1])
    fig_rq2_async_vs_sync_speedup(args.output, frames[2])
    fig_rq2_scaling_curves(args.output, frames[2])
    fig_rq3_drift_envelope(args.output, frames[3])
    fig_rq4_resource_scaling(args.output, frames[4])
    fig_rq4_pipeline_overhead(args.output, frames[4])
    print(f"produced {len(FIGURE_NAMES)} PDFs and {len(FIGURE_NAMES)} PNGs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
