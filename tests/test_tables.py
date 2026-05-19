"""Smoke test for :mod:`analysis.tables`.

Plants fixture CSVs + a minimal divergence_summary.json, runs
``main()``, asserts every .tex file produced parses as booktabs
(contains \\toprule and \\bottomrule).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.tables import main

TABLE_NAMES = [
    "tab_hardware_software",
    "tab_experimental_matrix",
    "tab_rq1_main_matrix",
    "tab_rq2_n_k_grid",
    "tab_rq3_drift_summary",
    "tab_rq4_resource_summary",
    "tab_cross_day_divergence",
    "tab_cross_phase_methodology",
]


def _build_fixture(summary_dir: Path) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    # Minimal RQ1: every program × every size × every load + cold-idle
    r1 = []
    for prog in ("l2_forward", "l3_lpm", "l3_lpm_acl", "l3_lpm_int"):
        for size in (64, 256, 1500):
            for load in (1, 25, 45):
                r1.append(
                    {
                        "p4_program": prog,
                        "packet_size_bytes": size,
                        "background_load_mbps": load,
                        "cold_idle_reference": False,
                        "n_samples": 1000,
                        "mean_us": 100.0,
                        "std_us": 10.0,
                        "median_us": 100.0,
                        "p25_us": 95.0,
                        "p75_us": 105.0,
                        "p99_us": 200.0,
                        "p999_us": 300.0,
                    }
                )
    r1.append(
        {
            "p4_program": "l3_lpm",
            "packet_size_bytes": 256,
            "background_load_mbps": 0,
            "cold_idle_reference": True,
            "n_samples": 1000,
            "mean_us": 550.0,
            "std_us": 50.0,
            "median_us": 550.0,
            "p25_us": 500.0,
            "p75_us": 600.0,
            "p99_us": 800.0,
            "p999_us": 1000.0,
        }
    )
    pd.DataFrame(r1).to_csv(summary_dir / "rq1_summary_rep1.csv", index=False)
    pd.DataFrame(r1).to_csv(summary_dir / "rq1_summary_rep2.csv", index=False)

    # RQ2 minimal
    rq2 = []
    for n in (1, 2, 4, 8):
        for k in (10, 100, 1000):
            for op in ("insert", "read"):
                modes = ("sync",) if n == 1 else ("sync", "async")
                for mode in modes:
                    rq2.append(
                        {
                            "n_switches": n,
                            "n_entries_per_switch": k,
                            "operation": op,
                            "mode": mode,
                            "n_reps": 10,
                            "mean_s": 0.1,
                            "std_s": 0.01,
                            "median_s": 0.1 if mode == "sync" else 0.06,
                            "p25_s": 0.09,
                            "p75_s": 0.11,
                            "median_entries_per_sec": 1000.0,
                        }
                    )
    pd.DataFrame(rq2).to_csv(summary_dir / "rq2_summary_rep1.csv", index=False)

    # RQ3 minimal
    rq3 = []
    for n in (2, 3):
        for load in (0, 25, 45):
            rq3.append(
                {
                    "n_switches": n,
                    "background_load_mbps": load,
                    "n_samples": 1000,
                    "mean_us": -4000.0,
                    "std_us": 200.0,
                    "median_us": -4000.0,
                    "p1_us": -4500.0,
                    "p99_us": -3500.0,
                    "abs_mean_us": 4000.0,
                    "abs_p99_us": 4500.0,
                    "per_hop_n": 1000 * (n - 1),
                    "per_hop_abs_mean_us": 4000.0,
                }
            )
    pd.DataFrame(rq3).to_csv(summary_dir / "rq3_summary_rep1.csv", index=False)
    pd.DataFrame(rq3).to_csv(summary_dir / "rq3_summary_rep2.csv", index=False)

    # RQ4 minimal
    rq4 = []
    for prog, topo, n, load in [
        ("l3_lpm", "single_switch", 1, 1),
        ("l3_lpm", "linear_n", 4, 1),
        ("l3_lpm", "linear_n", 8, 45),
    ]:
        for metric in ("cpu_percent_per_bmv2", "rss_per_bmv2_bytes", "net_io_pps_per_iface"):
            rq4.append(
                {
                    "p4_program": prog,
                    "topology": topo,
                    "n_switches": n,
                    "background_load_mbps": load,
                    "source_workload_type": "resource_only",
                    "metric": metric,
                    "n_samples": 600,
                    "mean": 10.0 * n,
                    "max": 50.0 * n,
                    "std": 5.0,
                    "p5": 0.0,
                    "p95": 40.0 * n,
                }
            )
    pd.DataFrame(rq4).to_csv(summary_dir / "rq4_summary_rep1.csv", index=False)

    # divergence_summary.json minimal
    div = {
        "per_rq": {
            "rq1": {
                "n_configs": 37,
                "pct_within_5": 8.1,
                "pct_within_10": 54.1,
                "pct_within_20": 89.2,
                "max_abs_delta_pct": 32.2,
            },
            "rq2": {
                "n_configs": 42,
                "pct_within_5": 26.2,
                "pct_within_10": 81.0,
                "pct_within_20": 92.9,
                "max_abs_delta_pct": 26.4,
            },
            "rq3": {
                "n_configs": 6,
                "pct_within_5": 0.0,
                "pct_within_10": 0.0,
                "pct_within_20": 0.0,
                "max_abs_delta_pct": 174.9,
            },
            "rq4": {
                "n_configs": 124,
                "pct_within_5": 64.5,
                "pct_within_10": 79.0,
                "pct_within_20": 89.5,
                "max_abs_delta_pct": 37.8,
            },
        },
        "flagged_configs": [],
    }
    (summary_dir / "divergence_summary.json").write_text(json.dumps(div))


def test_tables_main_produces_all_files(tmp_path: Path) -> None:
    summary = tmp_path / "summaries"
    out = tmp_path / "tables"
    _build_fixture(summary)
    rc = main(["--summary", str(summary), "--output", str(out)])
    assert rc == 0
    for name in TABLE_NAMES:
        tex = out / f"{name}.tex"
        assert tex.is_file(), f"missing {tex.name}"
        body = tex.read_text()
        # booktabs sanity
        assert "\\toprule" in body, f"{tex.name} missing \\toprule"
        assert "\\bottomrule" in body, f"{tex.name} missing \\bottomrule"
        assert "\\hline" not in body, f"{tex.name} uses \\hline (booktabs only)"
        assert "\\begin{tabular}" in body
        assert "\\end{tabular}" in body


def test_tables_output_count_matches_spec(tmp_path: Path) -> None:
    summary = tmp_path / "summaries"
    out = tmp_path / "tables"
    _build_fixture(summary)
    main(["--summary", str(summary), "--output", str(out)])
    tex_files = sorted(out.glob("*.tex"))
    assert len(tex_files) == 8, [t.name for t in tex_files]
