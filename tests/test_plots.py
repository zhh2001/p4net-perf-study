"""Smoke test for :mod:`analysis.plots`.

Plants minimal rep1+rep2 + divergence fixtures, runs ``main()``, and
asserts every figure produces both a non-empty PDF and a non-empty
PNG. Detailed visual correctness is left to manual inspection — these
tests prevent the analysis pipeline from silently breaking when CSV
schemas evolve.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.plots import main

FIGURE_NAMES = [
    "fig_cross_phase_methodology",
    "fig_cross_day_reproducibility_heatmap",
    "fig_rq1_latency_by_load",
    "fig_rq1_packet_size_independence",
    "fig_rq1_cold_warm_regime",
    "fig_rq2_async_vs_sync_speedup",
    "fig_rq2_scaling_curves",
    "fig_rq2_cross_day_reproducibility",
    "fig_rq3_drift_envelope",
    "fig_rq3_noise_decomposition",
    "fig_rq4_resource_scaling",
    "fig_rq4_pipeline_overhead",
]


def _rq1_summary(load: int, median: float, cold: bool = False) -> dict:
    return {
        "p4_program": "l3_lpm",
        "packet_size_bytes": 256,
        "background_load_mbps": load,
        "cold_idle_reference": cold,
        "n_samples": 1000,
        "mean_us": median,
        "std_us": 10.0,
        "median_us": median,
        "p25_us": median - 5,
        "p75_us": median + 5,
        "p99_us": median + 100,
        "p999_us": median + 200,
    }


def _build_fixture(summary_dir: Path) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    # RQ1 — three rows per program × the right config space the plots assume
    rows_r1 = []
    for prog in ("l2_forward", "l3_lpm", "l3_lpm_acl", "l3_lpm_int"):
        for size in (64, 256, 1500):
            for load in (1, 25, 45):
                rows_r1.append(
                    {**_rq1_summary(load, 100.0), "p4_program": prog, "packet_size_bytes": size}
                )
    rows_r1.append(_rq1_summary(0, 550.0, cold=True))
    pd.DataFrame(rows_r1).to_csv(summary_dir / "rq1_summary_rep1.csv", index=False)
    pd.DataFrame(rows_r1).to_csv(summary_dir / "rq1_summary_rep2.csv", index=False)

    # RQ2 — 2 N × 3 K × 2 op × 2 mode + N=1 sync-only
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
                            "median_s": 0.1 if mode == "sync" else 0.07,
                            "p25_s": 0.09,
                            "p75_s": 0.11,
                            "median_entries_per_sec": 1000.0,
                        }
                    )
    pd.DataFrame(rq2).to_csv(summary_dir / "rq2_summary_rep1.csv", index=False)
    pd.DataFrame(rq2).to_csv(summary_dir / "rq2_summary_rep2.csv", index=False)

    # RQ3 — 2 hops × 3 loads
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

    # RQ4 — l3_lpm at varying N, 1 Mbps + 45 Mbps + N=4 with l3_lpm_acl and _int
    rq4 = []
    for prog, topo, n, load in [
        ("l3_lpm", "single_switch", 1, 1),
        ("l3_lpm", "single_switch", 1, 45),
        ("l3_lpm", "linear_n", 4, 1),
        ("l3_lpm", "linear_n", 4, 45),
        ("l3_lpm", "linear_n", 8, 1),
        ("l3_lpm", "linear_n", 8, 45),
        ("l3_lpm_acl", "linear_n", 4, 1),
        ("l3_lpm_int", "linear_n", 4, 1),
    ]:
        for metric, mean_val, max_val in [
            ("cpu_percent_per_bmv2", 10.0 * n, 50.0 * n),
            ("cpu_percent_total", 5.0 * n, 25.0 * n),
            ("rss_per_bmv2_bytes", 42e6 * n, 45e6 * n),
            ("net_io_pps_per_iface", 1000.0 * n * load, 2000.0 * n * load),
        ]:
            rq4.append(
                {
                    "p4_program": prog,
                    "topology": topo,
                    "n_switches": n,
                    "background_load_mbps": load,
                    "source_workload_type": "resource_only",
                    "metric": metric,
                    "n_samples": 600,
                    "mean": mean_val,
                    "max": max_val,
                    "std": 5.0,
                    "p5": 0.0,
                    "p95": max_val * 0.9,
                }
            )
    pd.DataFrame(rq4).to_csv(summary_dir / "rq4_summary_rep1.csv", index=False)
    pd.DataFrame(rq4).to_csv(summary_dir / "rq4_summary_rep2.csv", index=False)

    # divergence_report — one row per RQ minimum
    div = []
    # rq1 row
    for i in range(5):
        div.append(
            {
                "rq": "rq1",
                "delta_pct_us": 3.0 + i,
                "within_5pct_us": True,
                "within_10pct_us": True,
                "within_20pct_us": True,
            }
        )
    for i in range(3):
        div.append(
            {
                "rq": "rq2",
                "delta_pct_s": 8.0 + i,
                "within_5pct_s": False,
                "within_10pct_s": True,
                "within_20pct_s": True,
            }
        )
    for i in range(2):
        div.append(
            {
                "rq": "rq3",
                "delta_pct_us": 25.0 + i,
                "within_5pct_us": False,
                "within_10pct_us": False,
                "within_20pct_us": False,
            }
        )
    for i in range(4):
        div.append(
            {
                "rq": "rq4",
                "delta_pct_value": 4.0 + i,
                "within_5pct_value": True,
                "within_10pct_value": True,
                "within_20pct_value": True,
            }
        )
    pd.DataFrame(div).to_csv(summary_dir / "divergence_report.csv", index=False)


def test_plots_main_produces_all_figures(tmp_path: Path) -> None:
    summary = tmp_path / "summaries"
    out = tmp_path / "figures"
    _build_fixture(summary)

    rc = main(["--summary", str(summary), "--output", str(out)])
    assert rc == 0
    for name in FIGURE_NAMES:
        pdf = out / f"{name}.pdf"
        png = out / f"{name}.png"
        assert pdf.is_file(), f"missing {pdf.name}"
        assert png.is_file(), f"missing {png.name}"
        assert pdf.stat().st_size > 1024, f"{pdf.name} suspiciously small"
        assert png.stat().st_size > 1024, f"{png.name} suspiciously small"


def test_plots_output_count_matches_spec(tmp_path: Path) -> None:
    """Catch the case where a figure is silently skipped — exactly 12 PDFs."""
    summary = tmp_path / "summaries"
    out = tmp_path / "figures"
    _build_fixture(summary)
    main(["--summary", str(summary), "--output", str(out)])
    pdfs = sorted(out.glob("*.pdf"))
    pngs = sorted(out.glob("*.png"))
    assert len(pdfs) == 12, [p.name for p in pdfs]
    assert len(pngs) == 12, [p.name for p in pngs]
