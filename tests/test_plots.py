"""Tests for clean-campaign paper plotting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.plots import FIGURE_NAMES, _build_rq2_ratio_grids, main

LABEL = "test_clean"


def _range(statistic: str, median: float, spread: float) -> dict[str, float]:
    return {
        f"{statistic}_median": median,
        f"{statistic}_min": median - spread,
        f"{statistic}_max": median + spread,
    }


def _build_fixture(summary_dir: Path) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)

    rq1: list[dict[str, object]] = []
    program_offset = {
        "l2_forward": 0.0,
        "l3_lpm": 10.0,
        "l3_lpm_acl": 20.0,
        "l3_lpm_int": 15.0,
    }
    for program, offset in program_offset.items():
        for packet_size in (64, 256, 1500):
            for load in (1, 25, 45):
                median = 90.0 + offset + packet_size / 1000 + load / 10
                rq1.append(
                    {
                        "n_reps": 5,
                        "p4_program": program,
                        "packet_size_bytes": packet_size,
                        "background_load_mbps": load,
                        "cold_idle_reference": False,
                        **_range("median_us", median, 5.0),
                    }
                )
    rq1.append(
        {
            "n_reps": 5,
            "p4_program": "l3_lpm",
            "packet_size_bytes": 256,
            "background_load_mbps": 0,
            "cold_idle_reference": True,
            **_range("median_us", 500.0, 25.0),
        }
    )
    pd.DataFrame(rq1).to_csv(summary_dir / f"rq1_summary_{LABEL}.csv", index=False)

    rq2: list[dict[str, object]] = []
    for n_switches in (1, 2, 4, 8):
        for n_entries in (10, 100, 1000):
            for operation in ("insert", "read"):
                modes = ("sync",) if n_switches == 1 else ("sync", "async")
                for mode in modes:
                    base = 0.01 * n_switches * (n_entries / 10)
                    if operation == "read":
                        base /= 10
                    factor = 0.75 if mode == "async" else 1.0
                    median = base * factor
                    rq2.append(
                        {
                            "n_reps": 5,
                            "n_switches": n_switches,
                            "n_entries_per_switch": n_entries,
                            "operation": operation,
                            "mode": mode,
                            **_range("wall_clock_s", median, median * 0.05),
                        }
                    )
    pd.DataFrame(rq2).to_csv(summary_dir / f"rq2_summary_{LABEL}.csv", index=False)

    rq3: list[dict[str, object]] = []
    for n_switches in (2, 3):
        for load in (0, 25, 45):
            absolute_median = 2000.0 + 250 * n_switches + 10 * load
            iqr = 40.0 + n_switches + load / 10
            rq3.append(
                {
                    "n_reps": 5,
                    "n_switches": n_switches,
                    "background_load_mbps": load,
                    **_range("abs_median_us", absolute_median, 400.0),
                    **_range("iqr_us", iqr, 5.0),
                }
            )
    pd.DataFrame(rq3).to_csv(summary_dir / f"rq3_summary_{LABEL}.csv", index=False)

    rq4: list[dict[str, object]] = []
    configurations = (
        ("l3_lpm", 1, 1),
        ("l3_lpm", 1, 45),
        ("l3_lpm", 4, 1),
        ("l3_lpm", 4, 45),
        ("l3_lpm", 8, 1),
        ("l3_lpm", 8, 45),
        ("l3_lpm_acl", 4, 1),
        ("l3_lpm_int", 4, 1),
    )
    for program, n_switches, load in configurations:
        metric_values = (
            ("cpu_percent_per_bmv2", 2.0 * n_switches + load, 4.0 * n_switches + load),
            ("cpu_percent_total", 5.0 + load / 10, 10.0 + load / 10),
            ("rss_per_bmv2_bytes", 42e6 * n_switches, 44e6 * n_switches),
            ("net_io_pps_per_iface", 1000.0 * n_switches, 2000.0 * n_switches),
        )
        for metric, mean, maximum in metric_values:
            rq4.append(
                {
                    "n_reps": 5,
                    "p4_program": program,
                    "n_switches": n_switches,
                    "background_load_mbps": load,
                    "source_workload_type": "resource_only",
                    "metric": metric,
                    **_range("mean", mean, mean * 0.05),
                    **_range("max", maximum, maximum * 0.05),
                }
            )
    pd.DataFrame(rq4).to_csv(summary_dir / f"rq4_summary_{LABEL}.csv", index=False)


def test_plots_main_produces_only_clean_campaign_figures(tmp_path: Path) -> None:
    summary = tmp_path / "summaries"
    output = tmp_path / "figures"
    _build_fixture(summary)

    assert main(
        [
            "--summary",
            str(summary),
            "--label",
            LABEL,
            "--output",
            str(output),
        ]
    ) == 0

    pdfs = sorted(path.stem for path in output.glob("*.pdf"))
    pngs = sorted(path.stem for path in output.glob("*.png"))
    assert pdfs == sorted(FIGURE_NAMES)
    assert pngs == sorted(FIGURE_NAMES)
    for name in FIGURE_NAMES:
        assert (output / f"{name}.pdf").stat().st_size > 1024
        assert (output / f"{name}.png").stat().st_size > 1024


def test_rq2_ratio_grid_uses_median_ratio_and_conservative_envelope() -> None:
    frame = pd.DataFrame(
        [
            {
                "n_switches": 2,
                "n_entries_per_switch": 100,
                "operation": "insert",
                "mode": "sync",
                "wall_clock_s_median": 10.0,
                "wall_clock_s_min": 8.0,
                "wall_clock_s_max": 12.0,
            },
            {
                "n_switches": 2,
                "n_entries_per_switch": 100,
                "operation": "insert",
                "mode": "async",
                "wall_clock_s_median": 5.0,
                "wall_clock_s_min": 4.0,
                "wall_clock_s_max": 6.0,
            },
        ]
    )

    central, lower, upper, switch_counts, entry_counts = _build_rq2_ratio_grids(
        frame, "insert"
    )
    assert switch_counts == [2]
    assert entry_counts == [100]
    assert central[0, 0] == 2.0
    assert np.isclose(lower[0, 0], 8.0 / 6.0)
    assert upper[0, 0] == 3.0


def test_active_figure_set_excludes_unverifiable_legacy_figures() -> None:
    excluded = {
        "fig_cross_phase_methodology",
        "fig_cross_day_reproducibility_heatmap",
        "fig_rq2_cross_day_reproducibility",
        "fig_rq3_noise_decomposition",
    }
    assert excluded.isdisjoint(FIGURE_NAMES)
