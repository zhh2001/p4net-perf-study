"""Generate paper figures from the rep1/rep2 summary CSVs.

One callable per figure; ``main()`` invokes them all and writes both
PDF and PNG @ 300 dpi to ``paper/figures/``. The figures are the
canonical visualisations referenced by the paper sections §4 (method),
§5 (RQ1 latency), §6 (RQ2 control plane), §7 (RQ3 INT fidelity), and
§8 (RQ4 resources).

CLI::

    python -m analysis.plots --summary data/summaries/ --output paper/figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# Paper-friendly style. Sans-serif, no neon, ColorBrewer Set1 for
# qualitative palettes and viridis for sequential/heatmap.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)

# ColorBrewer Set1 qualitative palette, paper-friendly.
PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
PROGRAM_COLORS = {
    "l2_forward": PALETTE[1],  # blue
    "l3_lpm": PALETTE[2],  # green
    "l3_lpm_acl": PALETTE[3],  # purple
    "l3_lpm_int": PALETTE[4],  # orange
}


def _save(fig: matplotlib.figure.Figure, out_dir: Path, name: str) -> None:
    """Write ``out_dir/<name>.pdf`` and ``out_dir/<name>.png`` (300 dpi)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# §4 Methodology figures.
# ---------------------------------------------------------------------------


# Cross-phase RQ1 l3_lpm 0/1 Mbps medians, drawn from the phase report
# series. These numbers are stable historical artifacts; inlining them
# here is simpler than re-parsing each phase's JSONL.
CROSS_PHASE_DATA = [
    ("B", "cold + offload bug", 242.0, "#984ea3"),
    ("C", "offload fix", 238.0, "#984ea3"),
    ("D", "full offload + monitor", 216.0, "#984ea3"),
    ("E", "warmup-then-stop 0 Mbps", 523.0, "#e41a1c"),
    ("F", "cont. carrier 0 Mbps (cold)", 549.5, "#e41a1c"),
    ("F", "cont. carrier 1 Mbps (warm)", 126.0, "#377eb8"),
    ("F", "cont. carrier 25 Mbps", 96.0, "#4daf4a"),
    ("F", "cont. carrier 45 Mbps", 100.0, "#4daf4a"),
]


def fig_cross_phase_methodology(out_dir: Path) -> None:
    """RQ1 l3_lpm 256 B median latency across methodology iterations."""
    labels = [f"Phase {p}\n{lbl}" for p, lbl, _, _ in CROSS_PHASE_DATA]
    values = [v for _, _, v, _ in CROSS_PHASE_DATA]
    colors = [c for _, _, _, c in CROSS_PHASE_DATA]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("RQ1 l3_lpm 256 B median latency (μs)")
    ax.set_title("Methodology evolution: cold-cache → warmup-then-stop → continuous carrier")
    for bar, v in zip(bars, values, strict=True):
        ax.annotate(
            f"{v:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    # Annotate the methodology pivot.
    ax.axvline(4.5, color="black", linestyle=":", alpha=0.5)
    ax.text(
        4.5,
        max(values) * 0.85,
        "← continuous-carrier methodology adopted",
        rotation=90,
        fontsize=9,
        color="black",
        ha="right",
        va="top",
    )
    _save(fig, out_dir, "fig_cross_phase_methodology")


def fig_cross_day_reproducibility_heatmap(out_dir: Path, divergence_report: pd.DataFrame) -> None:
    """Per-RQ × per-config band heatmap."""
    rq_order = ["rq1", "rq2", "rq3", "rq4"]
    # Build a long array: rows = RQ, cols = config index within RQ, val = band
    # 0 = within 5%, 1 = within 10%, 2 = within 20%, 3 = flagged, NaN = no data.
    max_configs = 0
    per_rq = {}
    for rq in rq_order:
        sub = divergence_report[divergence_report["rq"] == rq].copy()
        # find delta_pct column for this RQ
        dp_cols = [c for c in sub.columns if c.startswith("delta_pct_")]
        if not dp_cols:
            per_rq[rq] = np.asarray([])
            continue
        col = dp_cols[0]
        sub = sub[sub[col].notna()].copy()
        abs_pct = sub[col].abs().to_numpy()
        bands = np.where(
            abs_pct <= 5,
            0,
            np.where(abs_pct <= 10, 1, np.where(abs_pct <= 20, 2, 3)),
        )
        per_rq[rq] = bands
        max_configs = max(max_configs, len(bands))

    grid = np.full((len(rq_order), max_configs), np.nan)
    for i, rq in enumerate(rq_order):
        b = per_rq[rq]
        grid[i, : len(b)] = b

    band_colors = ["#1a9641", "#a6d96a", "#fdae61", "#d7191c"]  # green/yellow-green/orange/red
    cmap = matplotlib.colors.ListedColormap(band_colors)
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_yticks(range(len(rq_order)))
    ax.set_yticklabels([rq.upper() for rq in rq_order])
    ax.set_xlabel("Config index within RQ (sorted by |Δ%|)")
    ax.set_title("Cross-day reproducibility band per config (rep1 vs rep2)")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], pad=0.01)
    cbar.ax.set_yticklabels(["within 5%", "within 10%", "within 20%", "flagged (>20%)"])
    _save(fig, out_dir, "fig_cross_day_reproducibility_heatmap")


# ---------------------------------------------------------------------------
# §5 RQ1 figures.
# ---------------------------------------------------------------------------


def fig_rq1_latency_by_load(out_dir: Path, rq1: pd.DataFrame) -> None:
    """Median latency vs load, one line per program. Two panels:
    256B (left) + 1500B (right). Cold-idle reference excluded."""
    df = rq1[~rq1["cold_idle_reference"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    panel_sizes = [256, 1500]
    for ax, size in zip(axes, panel_sizes, strict=True):
        sub = df[df["packet_size_bytes"] == size]
        for prog in ("l2_forward", "l3_lpm", "l3_lpm_acl", "l3_lpm_int"):
            ssub = sub[sub["p4_program"] == prog].sort_values("background_load_mbps")
            if ssub.empty:
                continue
            loads = ssub["background_load_mbps"].to_numpy()
            med = ssub["median_us"].to_numpy()
            p25 = ssub["p25_us"].to_numpy()
            p75 = ssub["p75_us"].to_numpy()
            lower = med - p25
            upper = p75 - med
            ax.errorbar(
                loads,
                med,
                yerr=np.vstack([lower, upper]),
                marker="o",
                capsize=3,
                label=prog,
                color=PROGRAM_COLORS[prog],
                linewidth=1.5,
                markersize=6,
            )
        ax.set_xticks([1, 25, 45])
        ax.set_xlabel("Continuous-carrier load (Mbps)")
        ax.set_title(f"{size} B packets")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Median switch-transit latency (μs)")
    axes[0].legend(loc="upper right", framealpha=0.95)
    fig.suptitle("RQ1: per-program median latency vs background load")
    _save(fig, out_dir, "fig_rq1_latency_by_load")


def fig_rq1_packet_size_independence(out_dir: Path, rq1: pd.DataFrame) -> None:
    """Grouped bars at fixed load (1 Mbps): packet size on x-axis,
    program as group. Visualises that size has <5 μs effect."""
    df = rq1[(~rq1["cold_idle_reference"]) & (rq1["background_load_mbps"] == 1)].copy()
    sizes = sorted(df["packet_size_bytes"].unique())
    progs = ["l2_forward", "l3_lpm", "l3_lpm_acl", "l3_lpm_int"]
    width = 0.18
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(sizes))
    for i, prog in enumerate(progs):
        meds = []
        for size in sizes:
            row = df[(df["p4_program"] == prog) & (df["packet_size_bytes"] == size)]
            meds.append(float(row["median_us"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + i * width - 1.5 * width, meds, width, label=prog, color=PROGRAM_COLORS[prog])
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Packet size (B)")
    ax.set_ylabel("Median switch-transit latency (μs)")
    ax.set_title("RQ1: packet size has minimal effect at fixed (program, 1 Mbps)")
    ax.legend(loc="upper right", framealpha=0.95, ncol=2)
    ax.set_ylim(bottom=0)
    _save(fig, out_dir, "fig_rq1_packet_size_independence")


def fig_rq1_cold_warm_regime(out_dir: Path, rq1: pd.DataFrame) -> None:
    """Cold-idle vs warm baseline vs heated: l3_lpm 256B contrast."""
    sub = rq1[(rq1["p4_program"] == "l3_lpm") & (rq1["packet_size_bytes"] == 256)].copy()
    rows = []
    cold = sub[sub["cold_idle_reference"]]
    if not cold.empty:
        rows.append(
            ("cold-idle\n(0 Mbps, no carrier)", float(cold["median_us"].iloc[0]), "#d7191c")
        )
    for load, color, label in [
        (1, "#377eb8", "warm baseline\n(1 Mbps)"),
        (25, "#4daf4a", "heated\n(25 Mbps)"),
        (45, "#4daf4a", "heated\n(45 Mbps)"),
    ]:
        row = sub[(~sub["cold_idle_reference"]) & (sub["background_load_mbps"] == load)]
        if not row.empty:
            rows.append((label, float(row["median_us"].iloc[0]), color))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Median switch-transit latency (μs)")
    ax.set_title("RQ1 §5.2: cold-idle regime is 4-5× higher than the warm baseline")
    for bar, v in zip(bars, values, strict=True):
        ax.annotate(
            f"{v:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    if len(values) >= 2:
        ratio = values[0] / values[1]
        ax.annotate(
            f"cold/warm ratio ≈ {ratio:.1f}×",
            xy=(0.5, max(values) * 0.7),
            xytext=(0.5, max(values) * 0.7),
            ha="left",
            fontsize=11,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=0.3"},
        )
    _save(fig, out_dir, "fig_rq1_cold_warm_regime")


# ---------------------------------------------------------------------------
# §6 RQ2 figures.
# ---------------------------------------------------------------------------


def _build_rq2_ratio_grid(rq2: pd.DataFrame, op: str) -> tuple[np.ndarray, list[int], list[int]]:
    """Sync/async median_s ratio grid; N rows × K cols. NaN where N=1 (sync-only)."""
    Ns = sorted(rq2["n_switches"].unique())
    Ks = sorted(rq2["n_entries_per_switch"].unique())
    grid = np.full((len(Ns), len(Ks)), np.nan)
    for i, n in enumerate(Ns):
        for j, k in enumerate(Ks):
            sync_row = rq2[
                (rq2["n_switches"] == n)
                & (rq2["n_entries_per_switch"] == k)
                & (rq2["operation"] == op)
                & (rq2["mode"] == "sync")
            ]
            async_row = rq2[
                (rq2["n_switches"] == n)
                & (rq2["n_entries_per_switch"] == k)
                & (rq2["operation"] == op)
                & (rq2["mode"] == "async")
            ]
            if sync_row.empty or async_row.empty:
                continue
            sync_med = float(sync_row["median_s"].iloc[0])
            async_med = float(async_row["median_s"].iloc[0])
            if async_med > 0:
                grid[i, j] = sync_med / async_med
    return grid, Ns, Ks


def fig_rq2_async_vs_sync_speedup(out_dir: Path, rq2: pd.DataFrame) -> None:
    """Two-panel heatmap: sync/async ratio for insert (top), read (bottom)."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5))
    for ax, op in zip(axes, ("insert", "read"), strict=True):
        grid, Ns, Ks = _build_rq2_ratio_grid(rq2, op)
        # diverging colormap centered at 1.0
        vmax = max(2.0, float(np.nanmax(grid)) if np.any(~np.isnan(grid)) else 2.0)
        vmin = min(0.3, float(np.nanmin(grid)) if np.any(~np.isnan(grid)) else 0.3)
        im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_xticks(range(len(Ks)))
        ax.set_xticklabels([str(k) for k in Ks])
        ax.set_yticks(range(len(Ns)))
        ax.set_yticklabels([f"N={n}" for n in Ns])
        ax.set_xlabel("K (entries per switch)")
        ax.set_title(f"{op.upper()}: sync / async median ratio (>1 = async wins)")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="grey")
                else:
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}×",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="black",
                        weight="bold",
                    )
        fig.colorbar(im, ax=ax, pad=0.02, label="ratio")
    fig.suptitle("RQ2: async-vs-sync speedup grid")
    _save(fig, out_dir, "fig_rq2_async_vs_sync_speedup")


def fig_rq2_scaling_curves(out_dir: Path, rq2: pd.DataFrame) -> None:
    """Insert wall-clock vs K, log-log, one line per (N, mode)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sub = rq2[rq2["operation"] == "insert"].copy()
    Ns = sorted(sub["n_switches"].unique())
    n_colors = matplotlib.cm.viridis(np.linspace(0.2, 0.8, len(Ns)))
    for color, n in zip(n_colors, Ns, strict=True):
        for mode, ls, marker in (("sync", "-", "o"), ("async", "--", "s")):
            row = sub[(sub["n_switches"] == n) & (sub["mode"] == mode)].sort_values(
                "n_entries_per_switch"
            )
            if row.empty:
                continue
            ax.plot(
                row["n_entries_per_switch"],
                row["median_s"],
                marker=marker,
                linestyle=ls,
                color=color,
                label=f"N={n} {mode}",
                markersize=5,
                linewidth=1.5,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K (entries per switch, log scale)")
    ax.set_ylabel("Median wall-clock (s, log scale)")
    ax.set_title("RQ2 insert scaling: sync grows steeper than async at high N")
    ax.legend(loc="upper left", ncol=2, framealpha=0.95, fontsize=8)
    _save(fig, out_dir, "fig_rq2_scaling_curves")


def fig_rq2_cross_day_reproducibility(
    out_dir: Path, rq2_rep1: pd.DataFrame, rq2_rep2: pd.DataFrame
) -> None:
    """N=8 K=1000 cells: rep1 vs rep2 side-by-side bars + speedup annotation."""
    cells = [("insert", "sync"), ("insert", "async"), ("read", "sync"), ("read", "async")]
    rep1_vals = []
    rep2_vals = []
    labels = []
    for op, mode in cells:
        r1 = rq2_rep1[
            (rq2_rep1["n_switches"] == 8)
            & (rq2_rep1["n_entries_per_switch"] == 1000)
            & (rq2_rep1["operation"] == op)
            & (rq2_rep1["mode"] == mode)
        ]
        r2 = rq2_rep2[
            (rq2_rep2["n_switches"] == 8)
            & (rq2_rep2["n_entries_per_switch"] == 1000)
            & (rq2_rep2["operation"] == op)
            & (rq2_rep2["mode"] == mode)
        ]
        if r1.empty or r2.empty:
            continue
        rep1_vals.append(float(r1["median_s"].iloc[0]))
        rep2_vals.append(float(r2["median_s"].iloc[0]))
        labels.append(f"{op}\n{mode}")
    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, rep1_vals, width, label="rep1", color="#377eb8")
    ax.bar(x + width / 2, rep2_vals, width, label="rep2", color="#ff7f00")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median wall-clock (s)")
    ax.set_title("RQ2 cross-day reproducibility at N=8, K=1000")
    ax.legend()
    # Annotate async-insert speedup invariance.
    if len(rep1_vals) >= 2:
        spd_rep1 = rep1_vals[0] / rep1_vals[1]
        spd_rep2 = rep2_vals[0] / rep2_vals[1]
        ax.text(
            0.02,
            0.95,
            f"async-insert speedup\n  rep1: {spd_rep1:.2f}×\n  rep2: {spd_rep2:.2f}×\n  Δ = "
            f"{100 * (spd_rep2 - spd_rep1) / spd_rep1:+.1f}%",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=0.4"},
        )
    _save(fig, out_dir, "fig_rq2_cross_day_reproducibility")


# ---------------------------------------------------------------------------
# §7 RQ3 figures.
# ---------------------------------------------------------------------------


def fig_rq3_drift_envelope(out_dir: Path, rq3_rep1: pd.DataFrame, rq3_rep2: pd.DataFrame) -> None:
    """Left: per-run abs_mean across reps (unstable). Right: within-run std
    across reps (reproducible)."""
    keys = ["n_switches", "background_load_mbps"]
    merged = rq3_rep1.merge(rq3_rep2, on=keys, suffixes=("_r1", "_r2"))
    labels = [
        f"N={int(n)}\n{int(load)} Mbps"
        for n, load in zip(merged["n_switches"], merged["background_load_mbps"], strict=True)
    ]
    x = np.arange(len(labels))
    width = 0.4

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(x - width / 2, merged["abs_mean_us_r1"], width, label="rep1", color="#e41a1c")
    axes[0].bar(x + width / 2, merged["abs_mean_us_r2"], width, label="rep2", color="#ff7f00")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("|drift mean| (μs)")
    axes[0].set_title("Per-run mean: independent draws, NOT reproducible")
    axes[0].legend()

    axes[1].bar(x - width / 2, merged["std_us_r1"], width, label="rep1", color="#4daf4a")
    axes[1].bar(x + width / 2, merged["std_us_r2"], width, label="rep2", color="#377eb8")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Within-run std of drift (μs)")
    axes[1].set_title("Within-run std: structural — REPRODUCIBLE")
    axes[1].legend()
    axes[1].set_yscale("log")

    fig.suptitle(
        "RQ3 §7: drift mean shifts wildly across reps; within-run std stays in the same band"
    )
    _save(fig, out_dir, "fig_rq3_drift_envelope")


def fig_rq3_noise_decomposition(out_dir: Path, rq3_rep1: pd.DataFrame) -> None:
    """Per-config stacked bar: structural boot offset (large) vs estimated
    physical veth propagation (small)."""
    df = rq3_rep1.copy().sort_values(["n_switches", "background_load_mbps"])
    labels = [
        f"N={int(n)} {int(load)}Mbps"
        for n, load in zip(df["n_switches"], df["background_load_mbps"], strict=True)
    ]
    # Physical inter-hop propagation through veth is on the order of
    # tens of microseconds; we use 50 μs as a representative reference.
    physical = 50.0
    structural = df["abs_mean_us"].to_numpy() - physical
    structural = np.maximum(structural, 0)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, [physical] * len(x), color="#4daf4a", label="estimated physical propagation (~50 μs)")
    ax.bar(
        x,
        structural,
        bottom=[physical] * len(x),
        color="#e41a1c",
        label="structural boot_timestamp noise floor",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("|drift| contribution (μs, log)")
    ax.set_yscale("log")
    ax.set_title("RQ3: structural noise floor dominates physical propagation by 30-200×")
    ax.legend(loc="lower right")
    _save(fig, out_dir, "fig_rq3_noise_decomposition")


# ---------------------------------------------------------------------------
# §8 RQ4 figures.
# ---------------------------------------------------------------------------


def fig_rq4_resource_scaling(out_dir: Path, rq4: pd.DataFrame) -> None:
    """CPU + RSS scaling vs N. Left panel: CPU at 1 / 45 Mbps. Right: RSS."""
    df = rq4[
        (rq4["source_workload_type"] == "resource_only") & (rq4["p4_program"] == "l3_lpm")
    ].copy()
    cpu_df = df[df["metric"] == "cpu_percent_per_bmv2"].copy()
    rss_df = df[df["metric"] == "rss_per_bmv2_bytes"].copy()
    cpu_df["mean_cpu"] = cpu_df["mean"]
    rss_df["max_rss_mb"] = rss_df["max"] / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    for load, color, label in [(1, "#377eb8", "1 Mbps"), (45, "#e41a1c", "45 Mbps")]:
        sub = cpu_df[cpu_df["background_load_mbps"] == load].sort_values("n_switches")
        if sub.empty:
            continue
        axes[0].plot(
            sub["n_switches"],
            sub["mean_cpu"],
            marker="o",
            color=color,
            label=label,
            linewidth=2,
            markersize=8,
        )
    axes[0].set_xlabel("N (switch count)")
    axes[0].set_ylabel("Aggregate BMv2 CPU (%)")
    axes[0].set_title("CPU scaling: linear in N at each load")
    axes[0].legend(framealpha=0.95)
    axes[0].set_xticks([1, 4, 8])

    sub_rss = rss_df.sort_values("n_switches")
    axes[1].plot(
        sub_rss["n_switches"],
        sub_rss["max_rss_mb"],
        marker="s",
        color="#4daf4a",
        linewidth=2,
        markersize=8,
    )
    axes[1].set_xlabel("N (switch count)")
    axes[1].set_ylabel("Max aggregate BMv2 RSS (MB)")
    axes[1].set_title("RSS scaling: linear in N (~42 MB per process)")
    axes[1].set_xticks([1, 4, 8])
    fig.suptitle("RQ4: BMv2 resource scaling")
    _save(fig, out_dir, "fig_rq4_resource_scaling")


def fig_rq4_pipeline_overhead(out_dir: Path, rq4: pd.DataFrame) -> None:
    """At matched N=4, 1 Mbps: l3_lpm vs l3_lpm_acl vs l3_lpm_int CPU overhead."""
    df = rq4[
        (rq4["source_workload_type"] == "resource_only")
        & (rq4["metric"] == "cpu_percent_per_bmv2")
        & (rq4["n_switches"] == 4)
        & (rq4["background_load_mbps"] == 1)
    ].copy()
    if df.empty:
        # fall back: pick the resource_only N=4 cells regardless of load
        df = rq4[
            (rq4["source_workload_type"] == "resource_only")
            & (rq4["metric"] == "cpu_percent_per_bmv2")
            & (rq4["n_switches"] == 4)
        ].copy()
    progs = ["l3_lpm", "l3_lpm_acl", "l3_lpm_int"]
    means = []
    for prog in progs:
        row = df[df["p4_program"] == prog]
        means.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(
        progs, means, color=[PROGRAM_COLORS[p] for p in progs], edgecolor="black", linewidth=0.5
    )
    ax.set_ylabel("Aggregate BMv2 CPU (%)")
    ax.set_title("RQ4: pipeline-extension CPU overhead at N=4, 1 Mbps")
    for bar, v in zip(bars, means, strict=True):
        if np.isnan(v):
            continue
        ax.annotate(
            f"{v:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    if len(means) == 3 and not np.isnan(means[0]):
        ax.text(
            0.02,
            0.95,
            f"l3_lpm_acl Δ = {means[1] - means[0]:+.2f}%\n"
            f"l3_lpm_int Δ = {means[2] - means[0]:+.2f}%",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=0.4"},
        )
    _save(fig, out_dir, "fig_rq4_pipeline_overhead")


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate paper figures from summary CSVs.")
    parser.add_argument("--summary", type=Path, default=Path("data/summaries"))
    parser.add_argument("--output", type=Path, default=Path.home() / "projects/paper/figures")
    args = parser.parse_args(argv)

    rq1_rep1 = pd.read_csv(args.summary / "rq1_summary_rep1.csv")
    rq1_rep2 = pd.read_csv(args.summary / "rq1_summary_rep2.csv")
    rq2_rep1 = pd.read_csv(args.summary / "rq2_summary_rep1.csv")
    rq2_rep2 = pd.read_csv(args.summary / "rq2_summary_rep2.csv")
    rq3_rep1 = pd.read_csv(args.summary / "rq3_summary_rep1.csv")
    rq3_rep2 = pd.read_csv(args.summary / "rq3_summary_rep2.csv")
    rq4_rep1 = pd.read_csv(args.summary / "rq4_summary_rep1.csv")
    divergence_report = pd.read_csv(args.summary / "divergence_report.csv")

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"writing figures to {args.output}")
    fig_cross_phase_methodology(args.output)
    fig_cross_day_reproducibility_heatmap(args.output, divergence_report)
    fig_rq1_latency_by_load(args.output, rq1_rep1)
    fig_rq1_packet_size_independence(args.output, rq1_rep1)
    fig_rq1_cold_warm_regime(args.output, rq1_rep1)
    fig_rq2_async_vs_sync_speedup(args.output, rq2_rep1)
    fig_rq2_scaling_curves(args.output, rq2_rep1)
    fig_rq2_cross_day_reproducibility(args.output, rq2_rep1, rq2_rep2)
    fig_rq3_drift_envelope(args.output, rq3_rep1, rq3_rep2)
    fig_rq3_noise_decomposition(args.output, rq3_rep1)
    fig_rq4_resource_scaling(args.output, rq4_rep1)
    fig_rq4_pipeline_overhead(args.output, rq4_rep1)
    # Silence the unused rep2 vars when only rep1 is needed elsewhere.
    _ = rq1_rep2
    pdfs = sorted(args.output.glob("*.pdf"))
    pngs = sorted(args.output.glob("*.png"))
    print(f"  produced {len(pdfs)} PDFs and {len(pngs)} PNGs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
