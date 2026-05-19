"""Generate LaTeX (booktabs) table snippets for paper §3-§8.

Each callable writes a single ``\\begin{tabular}{...}...\\end{tabular}``
block (no surrounding ``table`` / ``caption`` — the paper writer adds
those around ``\\input{tables/foo.tex}``). Style: booktabs only
(``\\toprule`` / ``\\midrule`` / ``\\bottomrule``), no ``\\hline``, no
vertical rules.

CLI::

    python -m analysis.tables --summary data/summaries/ --output paper/tables/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROGRAMS = ["l2_forward", "l3_lpm", "l3_lpm_acl", "l3_lpm_int"]
SIZES = [64, 256, 1500]
LOADS = [1, 25, 45]


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body + "\n", encoding="utf-8")


def _lookup_rq1(df: pd.DataFrame, prog: str, size: int, load: int, cold: bool = False) -> float:
    sub = df[
        (df["p4_program"] == prog)
        & (df["packet_size_bytes"] == size)
        & (df["background_load_mbps"] == load)
        & (df["cold_idle_reference"] == cold)
    ]
    if sub.empty:
        return float("nan")
    return float(sub["median_us"].iloc[0])


def _fmt(x: float, fmt: str = ".0f") -> str:
    """NaN-safe formatter — emits an em-dash for missing data."""
    if x != x:  # NaN
        return "---"
    return format(x, fmt)


# ---------------------------------------------------------------------------
# §3 platform.
# ---------------------------------------------------------------------------


def tab_hardware_software(out_dir: Path, summary_dir: Path) -> None:
    # Find any system_info_*.json — they all have the same hardware
    # fingerprint on this single rig. Fall back to a recorded reference
    # set if none are present (e.g., tests).
    candidates = sorted(summary_dir.parent.glob("raw_rep1/system_info_*.json"))
    if not candidates:
        candidates = sorted(summary_dir.parent.glob("raw_rep2/system_info_*.json"))
    info = {}
    if candidates:
        info = json.loads(candidates[0].read_text())
    # Project a stable set of fields onto the table.
    rows = [
        ("CPU model", info.get("cpu_model", "13th Gen Intel(R) Core(TM) i5-13500H")),
        ("CPU physical cores", str(info.get("cpu_cores_physical", 8))),
        ("CPU logical cores", str(info.get("cpu_cores_logical", 16))),
        ("RAM (GB)", f"{info.get('ram_total_gb', 11.68):.2f}"),
        ("Distro", info.get("distro", "Ubuntu 24.04.4 LTS")),
        ("Kernel", info.get("kernel_version", "6.6.87.2-microsoft-standard-WSL2")),
        ("Python", info.get("python_version", "3.12.3")),
        ("p4net", info.get("p4net_version", "1.7.0")),
        ("p4c", info.get("p4c_version", "p4c 1.2.5.10")),
        ("BMv2", info.get("bmv2_version", "1.15.0-2bdd0b7b")),
    ]
    body = [
        "\\begin{tabular}{ll}",
        "\\toprule",
        "\\textbf{Component} & \\textbf{Value} \\\\",
        "\\midrule",
    ]
    for k, v in rows:
        body.append(f"{k} & \\texttt{{{v}}} \\\\")
    body.append("\\bottomrule")
    body.append("\\end{tabular}")
    _write(out_dir / "tab_hardware_software.tex", "\n".join(body))


# ---------------------------------------------------------------------------
# §4 experimental matrix.
# ---------------------------------------------------------------------------


def tab_experimental_matrix(out_dir: Path) -> None:
    rows = [
        (
            "RQ1",
            "single\\_switch",
            "$4 \\times 3 \\times 3 = 36 + 1$ cold-idle",
            "1\\,000 probes/cell",
            "1",
        ),
        (
            "RQ2",
            "linear\\_n ($N \\in \\{1,2,4,8\\}$)",
            "$42$ ($N{=}1$ sync-only)",
            "10 reps/cell",
            "1",
        ),
        ("RQ3", "linear\\_n (2- and 3-hop)", "$2 \\times 3 = 6$", "1\\,000 INT probes/cell", "1"),
        ("RQ4", "single + linear\\_n", "8 explicit", "60\\,s @ 100\\,ms cadence", "1"),
    ]
    body = [
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "\\textbf{RQ} & \\textbf{Topology} & \\textbf{Configs} & "
        "\\textbf{Sampling} & \\textbf{Reps} \\\\",
        "\\midrule",
    ]
    for r in rows:
        body.append(" & ".join(r) + " \\\\")
    body.append("\\bottomrule")
    body.append("\\end{tabular}")
    _write(out_dir / "tab_experimental_matrix.tex", "\n".join(body))


# ---------------------------------------------------------------------------
# §5 RQ1 main matrix.
# ---------------------------------------------------------------------------


def tab_rq1_main_matrix(out_dir: Path, rq1_r1: pd.DataFrame, rq1_r2: pd.DataFrame) -> None:
    body = [
        "\\begin{tabular}{ll" + "r" * len(LOADS) + "}",
        "\\toprule",
        "\\textbf{Program} & \\textbf{Size (B)} & "
        + " & ".join(f"\\textbf{{{load}\\,Mbps}}" for load in LOADS)
        + " \\\\",
        "\\midrule",
    ]
    for prog in PROGRAMS:
        for size in SIZES:
            cells = []
            for load in LOADS:
                m1 = _lookup_rq1(rq1_r1, prog, size, load)
                m2 = _lookup_rq1(rq1_r2, prog, size, load)
                if m1 != m1:
                    cells.append("---")
                else:
                    cells.append(f"{m1:.0f} \\textit{{\\small ({m2:.0f})}}")
            body.append(f"\\texttt{{{prog}}} & {size} & " + " & ".join(cells) + " \\\\")
    # Cold-idle reference row.
    m1 = _lookup_rq1(rq1_r1, "l3_lpm", 256, 0, cold=True)
    m2 = _lookup_rq1(rq1_r2, "l3_lpm", 256, 0, cold=True)
    body.append("\\midrule")
    cold_cell = "---" if m1 != m1 else f"{m1:.0f} \\textit{{\\small ({m2:.0f})}}"
    body.append(
        "\\multicolumn{2}{l}{\\textit{cold-idle reference}} & " + cold_cell + " & --- & --- \\\\"
    )
    body.append("\\bottomrule")
    body.append("\\end{tabular}")
    _write(out_dir / "tab_rq1_main_matrix.tex", "\n".join(body))


# ---------------------------------------------------------------------------
# §6 RQ2 N×K grid.
# ---------------------------------------------------------------------------


def tab_rq2_n_k_grid(out_dir: Path, rq2: pd.DataFrame) -> None:
    Ns = sorted(rq2["n_switches"].unique())
    Ks = sorted(rq2["n_entries_per_switch"].unique())

    def block(op: str) -> list[str]:
        rows = [
            f"\\multicolumn{{{1 + 3 * len(Ks)}}}{{l}}{{\\textbf{{{op.upper()}}}}} \\\\",
            "\\cmidrule(lr){1-" + str(1 + 3 * len(Ks)) + "}",
            "\\textbf{N} & "
            + " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{K={k}}}}}" for k in Ks)
            + " \\\\",
        ]
        sub = " & ".join(["sync", "async", "$\\times$"] * len(Ks))
        rows.append("& " + sub + " \\\\")
        for n in Ns:
            cells = [str(n)]
            for k in Ks:
                sync_r = rq2[
                    (rq2["n_switches"] == n)
                    & (rq2["n_entries_per_switch"] == k)
                    & (rq2["operation"] == op)
                    & (rq2["mode"] == "sync")
                ]
                async_r = rq2[
                    (rq2["n_switches"] == n)
                    & (rq2["n_entries_per_switch"] == k)
                    & (rq2["operation"] == op)
                    & (rq2["mode"] == "async")
                ]
                s = float(sync_r["median_s"].iloc[0]) if not sync_r.empty else float("nan")
                a = float(async_r["median_s"].iloc[0]) if not async_r.empty else float("nan")
                r = s / a if (a == a and a > 0) else float("nan")
                cells += [_fmt(s, ".4f"), _fmt(a, ".4f"), _fmt(r, ".2f")]
            rows.append(" & ".join(cells) + " \\\\")
        return rows

    body = ["\\begin{tabular}{l" + "rrr" * len(Ks) + "}", "\\toprule"]
    body += block("insert")
    body.append("\\midrule")
    body += block("read")
    body.append("\\bottomrule")
    body.append("\\end{tabular}")
    _write(out_dir / "tab_rq2_n_k_grid.tex", "\n".join(body))


# ---------------------------------------------------------------------------
# §7 RQ3 drift summary.
# ---------------------------------------------------------------------------


def tab_rq3_drift_summary(out_dir: Path, rq3_r1: pd.DataFrame, rq3_r2: pd.DataFrame) -> None:
    merged = rq3_r1.merge(
        rq3_r2, on=["n_switches", "background_load_mbps"], suffixes=("_r1", "_r2")
    )
    body = [
        "\\begin{tabular}{rrrrrr}",
        "\\toprule",
        "\\textbf{N} & \\textbf{Load (Mbps)} & "
        "\\textbf{$|mean|_{r1}$ (μs)} & \\textbf{$|mean|_{r2}$ (μs)} & "
        "\\textbf{$std_{r1}$ (μs)} & \\textbf{$std_{r2}$ (μs)} \\\\",
        "\\midrule",
    ]
    for _, row in merged.iterrows():
        body.append(
            f"{int(row['n_switches'])} & {int(row['background_load_mbps'])} & "
            f"{row['abs_mean_us_r1']:.0f} & {row['abs_mean_us_r2']:.0f} & "
            f"{row['std_us_r1']:.0f} & {row['std_us_r2']:.0f} \\\\"
        )
    body.append("\\bottomrule")
    body.append("\\end{tabular}")
    _write(out_dir / "tab_rq3_drift_summary.tex", "\n".join(body))


# ---------------------------------------------------------------------------
# §8 RQ4 resource summary.
# ---------------------------------------------------------------------------


def tab_rq4_resource_summary(out_dir: Path, rq4: pd.DataFrame) -> None:
    sub = rq4[rq4["source_workload_type"] == "resource_only"].copy()
    # Reduce to one row per (config) via metric pivot.
    cfg_keys = ["p4_program", "topology", "n_switches", "background_load_mbps"]
    cpu = sub[sub["metric"] == "cpu_percent_per_bmv2"].set_index(cfg_keys)["mean"]
    cpu_max = sub[sub["metric"] == "cpu_percent_per_bmv2"].set_index(cfg_keys)["max"]
    rss = sub[sub["metric"] == "rss_per_bmv2_bytes"].set_index(cfg_keys)["max"] / 1e6
    rx = sub[sub["metric"] == "net_io_pps_per_iface"].set_index(cfg_keys)["mean"]
    keys = sorted(set(cpu.index) | set(rss.index))

    body = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "\\textbf{Config} & \\textbf{CPU avg (\\%)} & \\textbf{CPU max (\\%)} & "
        "\\textbf{RSS max (MB)} & \\textbf{RX pps} \\\\",
        "\\midrule",
    ]
    for key in keys:
        prog, topo, n, load = key
        label = f"\\texttt{{{prog}}} {topo} $N{{=}}{n}$ {load}\\,Mbps"
        c_avg = cpu.get(key, float("nan"))
        c_max = cpu_max.get(key, float("nan"))
        r_max = rss.get(key, float("nan"))
        x_avg = rx.get(key, float("nan"))
        body.append(
            f"{label} & {_fmt(c_avg, '.2f')} & {_fmt(c_max, '.1f')} & "
            f"{_fmt(r_max, '.1f')} & {_fmt(x_avg, '.0f')} \\\\"
        )
    body.append("\\bottomrule")
    body.append("\\end{tabular}")
    _write(out_dir / "tab_rq4_resource_summary.tex", "\n".join(body))


# ---------------------------------------------------------------------------
# §4 cross-day divergence summary.
# ---------------------------------------------------------------------------


def tab_cross_day_divergence(out_dir: Path, divergence_summary: dict) -> None:
    body = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "\\textbf{RQ} & \\textbf{N configs} & \\textbf{Within 5\\%} & "
        "\\textbf{Within 10\\%} & \\textbf{Within 20\\%} & \\textbf{max $|\\Delta\\%|$} \\\\",
        "\\midrule",
    ]
    for rq in ("rq1", "rq2", "rq3", "rq4"):
        stats = divergence_summary.get("per_rq", {}).get(rq)
        if stats is None:
            continue
        body.append(
            f"\\textbf{{{rq.upper()}}} & {int(stats['n_configs'])} & "
            f"{stats['pct_within_5']:.1f}\\% & {stats['pct_within_10']:.1f}\\% & "
            f"{stats['pct_within_20']:.1f}\\% & {stats['max_abs_delta_pct']:.1f}\\% \\\\"
        )
    body.append("\\bottomrule")
    body.append("\\end{tabular}")
    _write(out_dir / "tab_cross_day_divergence.tex", "\n".join(body))


# ---------------------------------------------------------------------------
# §4 cross-phase methodology table.
# ---------------------------------------------------------------------------


def tab_cross_phase_methodology(out_dir: Path) -> None:
    rows = [
        (
            "B",
            "none (cold-cache, offload bug)",
            "242",
            "---",
            "---",
            "---",
            "Initial impl + first pilot",
        ),
        ("C", "none (offload bug found)", "238", "---", "---", "---", "veth L4 offload disabled"),
        (
            "D",
            "none (offload + monitor)",
            "216",
            "---",
            "---",
            "---",
            "Resource monitor added; 100~Mbps load infeasible",
        ),
        (
            "E",
            "warmup 30\\,s @ 1\\,Mbps then stop",
            "523",
            "---",
            "108",
            "108",
            "Asymmetry persisted; cache-warming hypothesis failed",
        ),
        (
            "F",
            "continuous 1\\,Mbps carrier",
            "550",
            "126",
            "96",
            "100",
            "Continuous-carrier methodology; cold/warm 4.4$\\times$ gap fixed",
        ),
    ]
    body = [
        "\\begin{tabular}{lllrrrl}",
        "\\toprule",
        "\\textbf{Phase} & \\textbf{Warmup policy} & "
        "\\textbf{0\\,Mbps (μs)} & \\textbf{1\\,Mbps (μs)} & "
        "\\textbf{25\\,Mbps (μs)} & \\textbf{45\\,Mbps (μs)} & "
        "\\textbf{Methodology change} \\\\",
        "\\midrule",
    ]
    for r in rows:
        body.append(" & ".join(r) + " \\\\")
    body.append("\\bottomrule")
    body.append("\\end{tabular}")
    _write(out_dir / "tab_cross_phase_methodology.tex", "\n".join(body))


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from summary CSVs.")
    parser.add_argument("--summary", type=Path, default=Path("data/summaries"))
    parser.add_argument("--output", type=Path, default=Path.home() / "projects/paper/tables")
    args = parser.parse_args(argv)

    rq1_r1 = pd.read_csv(args.summary / "rq1_summary_rep1.csv")
    rq1_r2 = pd.read_csv(args.summary / "rq1_summary_rep2.csv")
    rq2 = pd.read_csv(args.summary / "rq2_summary_rep1.csv")
    rq3_r1 = pd.read_csv(args.summary / "rq3_summary_rep1.csv")
    rq3_r2 = pd.read_csv(args.summary / "rq3_summary_rep2.csv")
    rq4 = pd.read_csv(args.summary / "rq4_summary_rep1.csv")
    div_summary = json.loads((args.summary / "divergence_summary.json").read_text())

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"writing tables to {args.output}")
    tab_hardware_software(args.output, args.summary)
    tab_experimental_matrix(args.output)
    tab_rq1_main_matrix(args.output, rq1_r1, rq1_r2)
    tab_rq2_n_k_grid(args.output, rq2)
    tab_rq3_drift_summary(args.output, rq3_r1, rq3_r2)
    tab_rq4_resource_summary(args.output, rq4)
    tab_cross_day_divergence(args.output, div_summary)
    tab_cross_phase_methodology(args.output)
    tex = sorted(args.output.glob("*.tex"))
    print(f"  produced {len(tex)} LaTeX snippets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
