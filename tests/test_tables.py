"""Tests for clean five-restart LaTeX table generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis.tables import RQ4_CONFIGS, main

TABLE_NAMES = (
    "tab_rq1_main_matrix",
    "tab_rq2_n_k_grid",
    "tab_rq3_drift_summary",
    "tab_rq4_resource_summary",
)


def _range_columns(metric: str, median: float, minimum: float, maximum: float) -> dict:
    return {
        f"{metric}_median": median,
        f"{metric}_min": minimum,
        f"{metric}_max": maximum,
    }


def _build_fixture(summary_dir: Path, label: str = "c4_clean") -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)

    rq1 = []
    for program in ("l2_forward", "l3_lpm", "l3_lpm_acl", "l3_lpm_int"):
        for size in (64, 256, 1500):
            for load in (1, 25, 45):
                rq1.append(
                    {
                        "p4_program": program,
                        "packet_size_bytes": size,
                        "background_load_mbps": load,
                        "cold_idle_reference": False,
                        "n_reps": 5,
                        **_range_columns("median_us", 100.04, 90.04, 110.04),
                    }
                )
    rq1.append(
        {
            "p4_program": "l3_lpm",
            "packet_size_bytes": 256,
            "background_load_mbps": 0,
            "cold_idle_reference": True,
            "n_reps": 5,
            **_range_columns("median_us", 550.04, 500.04, 600.04),
        }
    )
    pd.DataFrame(rq1).to_csv(summary_dir / f"rq1_summary_{label}.csv", index=False)

    rq2 = []
    for n_switches in (1, 2, 4, 8):
        modes = ("sync",) if n_switches == 1 else ("sync", "async")
        for n_entries in (10, 100, 1000):
            for operation in ("insert", "read"):
                for mode in modes:
                    rq2.append(
                        {
                            "n_switches": n_switches,
                            "n_entries_per_switch": n_entries,
                            "operation": operation,
                            "mode": mode,
                            "n_reps": 5,
                            **_range_columns("wall_clock_s", 0.123456, 0.12, 0.13),
                        }
                    )
    pd.DataFrame(rq2).to_csv(summary_dir / f"rq2_summary_{label}.csv", index=False)

    rq3 = []
    for n_switches in (2, 3):
        for load in (0, 25, 45):
            rq3.append(
                {
                    "n_switches": n_switches,
                    "background_load_mbps": load,
                    "n_reps": 5,
                    **_range_columns("median_us", -4000.0, -4100.0, -3900.0),
                    **_range_columns("abs_median_us", 4000.0, 3900.0, 4100.0),
                    **_range_columns("iqr_us", 200.0, 180.0, 220.0),
                }
            )
    pd.DataFrame(rq3).to_csv(summary_dir / f"rq3_summary_{label}.csv", index=False)

    rq4 = []
    for program, topology, n_switches, load in RQ4_CONFIGS:
        base = {
            "p4_program": program,
            "topology": topology,
            "n_switches": n_switches,
            "background_load_mbps": load,
            "source_workload_type": "resource_only",
            "n_reps": 5,
        }
        for metric in (
            "cpu_percent_per_bmv2",
            "rss_per_bmv2_bytes",
            "net_io_pps_per_iface",
        ):
            rq4.append(
                {
                    **base,
                    "metric": metric,
                    **_range_columns("mean", 10.04, 9.04, 11.04),
                    **_range_columns("p95", 40.04, 38.04, 42.04),
                    **_range_columns(
                        "max", 168_800_000.0, 167_800_000.0, 169_800_000.0
                    ),
                }
            )
    pd.DataFrame(rq4).to_csv(summary_dir / f"rq4_summary_{label}.csv", index=False)


def test_main_produces_only_clean_tables_with_fixed_precision(tmp_path: Path) -> None:
    summary = tmp_path / "summaries"
    output = tmp_path / "tables"
    _build_fixture(summary)

    assert main(["--summary", str(summary), "--output", str(output)]) == 0
    assert sorted(path.stem for path in output.glob("*.tex")) == sorted(TABLE_NAMES)

    for name in TABLE_NAMES:
        body = (output / f"{name}.tex").read_text(encoding="utf-8")
        assert "\\toprule" in body
        assert "\\bottomrule" in body
        assert "\\hline" not in body
        assert "\\rev{" not in body

    rq1 = (output / "tab_rq1_main_matrix.tex").read_text(encoding="utf-8")
    assert "100.0 [90.0\\,--\\,110.0]" in rq1
    assert "\\texttt{l3\\_lpm\\_int}" in rq1

    rq2 = (output / "tab_rq2_n_k_grid.tex").read_text(encoding="utf-8")
    assert "\\textbf{INSERT}" in rq2
    assert "\\textbf{READ}" in rq2
    assert "0.1235 [0.1200\\,--\\,0.1300]" in rq2

    rq3 = (output / "tab_rq3_drift_summary.tex").read_text(encoding="utf-8")
    assert "-4000.0 [-4100.0\\,--\\,-3900.0]" in rq3
    assert "200.0 [180.0\\,--\\,220.0]" in rq3

    rq4 = (output / "tab_rq4_resource_summary.tex").read_text(encoding="utf-8")
    assert "CPU mean" in rq4
    assert "CPU p95" in rq4
    assert "168.8 [167.8\\,--\\,169.8]" in rq4
    assert "10.0 [9.0\\,--\\,11.0]" in rq4


def test_custom_label_is_used_for_all_four_inputs(tmp_path: Path) -> None:
    summary = tmp_path / "summaries"
    output = tmp_path / "tables"
    _build_fixture(summary, label="audit")

    assert (
        main(
            [
                "--summary",
                str(summary),
                "--output",
                str(output),
                "--label",
                "audit",
            ]
        )
        == 0
    )


def test_cli_lists_missing_clean_summary_files(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    summary = tmp_path / "summaries"
    summary.mkdir()
    pd.DataFrame([{"n_reps": 5}]).to_csv(
        summary / "rq1_summary_c4_clean.csv", index=False
    )

    with pytest.raises(SystemExit, match="2"):
        main(["--summary", str(summary), "--output", str(tmp_path / "tables")])

    error = capsys.readouterr().err
    assert "missing clean summary file(s)" in error
    assert "rq2_summary_c4_clean.csv" in error
    assert "rq3_summary_c4_clean.csv" in error
    assert "rq4_summary_c4_clean.csv" in error


def test_rejects_non_five_restart_summary(tmp_path: Path) -> None:
    summary = tmp_path / "summaries"
    output = tmp_path / "tables"
    _build_fixture(summary)
    rq1_path = summary / "rq1_summary_c4_clean.csv"
    rq1 = pd.read_csv(rq1_path)
    rq1.loc[0, "n_reps"] = 2
    rq1.to_csv(rq1_path, index=False)

    with pytest.raises(ValueError, match="must contain n_reps=5"):
        main(["--summary", str(summary), "--output", str(output)])
