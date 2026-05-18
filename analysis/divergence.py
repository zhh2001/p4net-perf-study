"""Cross-day replication divergence analysis.

Inner-joins the per-RQ summary CSVs across two replications and reports
per-config median deltas. The output is the paper's §4 reproducibility
evidence: a table of (config, rep1_value, rep2_value, delta, delta_pct,
within_5pct, within_10pct, within_20pct) plus an aggregate summary
counting how many configs fall in each band.

Reproducibility thresholds (paper §4 Methodology)
-------------------------------------------------

* **Within 5 %** — gold-standard reproducible; the paper says "highly
  stable across days."
* **Within 10 %** — standard; the paper says "reproducible within
  typical measurement noise."
* **Within 20 %** — acceptable; the paper acknowledges some configs
  show this level of cross-day variation.
* **Exceeds 20 %** — flagged for investigation. If unexplained the
  paper either re-runs the config a third time or discloses the
  variability as a measurement-floor limitation.

Domain-specific notes
---------------------

* **RQ3 abs_mean** comparisons can show very large absolute differences
  (often 1 000+ μs) because the boot_timestamp offset is captured once
  per run and is essentially random across runs. RQ3 reproducibility
  is judged on the **within-run std** being comparable across reps,
  not on the per-run mean being comparable. The divergence table
  reports both columns; the percent-band thresholds applied to
  abs_mean will look bad, but that's structural, not a measurement
  failure.

* **RQ4** metrics like ``cpu_max`` have high inherent variance because
  the per-100-ms-sample maximum is a single observation; even 30 %
  cross-day deltas can be within noise. The 5/10/20 % bands are
  stricter than what RQ4 needs.

CLI::

    python -m analysis.divergence --summary data/summaries/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def _add_delta_columns(
    df: pd.DataFrame,
    key_cols: list[str],
    rep1_col: str,
    rep2_col: str,
    out_prefix: str,
) -> pd.DataFrame:
    """Add ``delta_<prefix>``, ``delta_pct_<prefix>`` and three
    ``within_*pct`` boolean columns based on a rep1-vs-rep2 pair."""
    delta = df[rep2_col] - df[rep1_col]
    delta_pct = 100.0 * delta / df[rep1_col].replace(0, math.nan)
    df = df.copy()
    df[f"delta_{out_prefix}"] = delta
    df[f"delta_pct_{out_prefix}"] = delta_pct
    abs_pct = delta_pct.abs()
    df[f"within_5pct_{out_prefix}"] = abs_pct <= 5.0
    df[f"within_10pct_{out_prefix}"] = abs_pct <= 10.0
    df[f"within_20pct_{out_prefix}"] = abs_pct <= 20.0
    # Preserve the key columns at the front for readability.
    front = [
        *key_cols,
        rep1_col,
        rep2_col,
        f"delta_{out_prefix}",
        f"delta_pct_{out_prefix}",
        f"within_5pct_{out_prefix}",
        f"within_10pct_{out_prefix}",
        f"within_20pct_{out_prefix}",
    ]
    rest = [c for c in df.columns if c not in front]
    return df[[*front, *rest]]


def compare_rq1(rep1_path: Path, rep2_path: Path) -> pd.DataFrame:
    """Per-config rep1-vs-rep2 deltas for switch_transit_us medians."""
    r1 = pd.read_csv(rep1_path)
    r2 = pd.read_csv(rep2_path)
    keys = ["p4_program", "packet_size_bytes", "background_load_mbps", "cold_idle_reference"]
    merged = r1.merge(r2, on=keys, suffixes=("_rep1", "_rep2"), how="inner")
    return _add_delta_columns(merged, keys, "median_us_rep1", "median_us_rep2", "us")


def compare_rq2(rep1_path: Path, rep2_path: Path) -> pd.DataFrame:
    """Per-config rep1-vs-rep2 deltas for control_plane_wall_clock_s medians."""
    r1 = pd.read_csv(rep1_path)
    r2 = pd.read_csv(rep2_path)
    keys = ["n_switches", "n_entries_per_switch", "operation", "mode"]
    merged = r1.merge(r2, on=keys, suffixes=("_rep1", "_rep2"), how="inner")
    return _add_delta_columns(merged, keys, "median_s_rep1", "median_s_rep2", "s")


def compare_rq3(rep1_path: Path, rep2_path: Path) -> pd.DataFrame:
    """Per-config rep1-vs-rep2 deltas for INT drift |mean| (abs_mean_us).

    The signed mean is dominated by per-run boot_timestamp offset and
    is not a stable measurement target; abs_mean is the useful one.
    """
    r1 = pd.read_csv(rep1_path)
    r2 = pd.read_csv(rep2_path)
    keys = ["n_switches", "background_load_mbps"]
    merged = r1.merge(r2, on=keys, suffixes=("_rep1", "_rep2"), how="inner")
    return _add_delta_columns(merged, keys, "abs_mean_us_rep1", "abs_mean_us_rep2", "us")


def compare_rq4(rep1_path: Path, rep2_path: Path) -> pd.DataFrame:
    """Per-config rep1-vs-rep2 deltas for RQ4 resource metric means.

    RQ4 rows are keyed by (config, metric); the primary stat varies by
    metric — for cpu_percent_total we compare mean, for rss_per_bmv2_bytes
    we compare max, for net_io_pps_per_iface we compare mean. This helper
    uses ``mean`` uniformly; downstream readers who care about ``max`` can
    re-merge the raw CSVs directly.
    """
    r1 = pd.read_csv(rep1_path)
    r2 = pd.read_csv(rep2_path)
    keys = [
        "p4_program",
        "topology",
        "n_switches",
        "background_load_mbps",
        "source_workload_type",
        "metric",
    ]
    merged = r1.merge(r2, on=keys, suffixes=("_rep1", "_rep2"), how="inner")
    return _add_delta_columns(merged, keys, "mean_rep1", "mean_rep2", "value")


def summarize(comparisons: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Aggregate-band statistics + the configs flagged for investigation."""
    out: dict[str, Any] = {"per_rq": {}, "flagged_configs": []}
    for rq, df in comparisons.items():
        # Locate the within_* columns regardless of prefix.
        w5 = [c for c in df.columns if c.startswith("within_5pct_")]
        w10 = [c for c in df.columns if c.startswith("within_10pct_")]
        w20 = [c for c in df.columns if c.startswith("within_20pct_")]
        delta_pct = [c for c in df.columns if c.startswith("delta_pct_")]
        if not (w5 and w10 and w20 and delta_pct):
            continue
        n_total = len(df)
        valid = df[delta_pct[0]].notna()
        n_valid = int(valid.sum())
        per = {
            "n_configs": n_total,
            "n_valid": n_valid,
            "pct_within_5": float(df[w5[0]].mean() * 100) if n_total else 0.0,
            "pct_within_10": float(df[w10[0]].mean() * 100) if n_total else 0.0,
            "pct_within_20": float(df[w20[0]].mean() * 100) if n_total else 0.0,
            "max_abs_delta_pct": (float(df[delta_pct[0]].abs().max()) if n_valid else float("nan")),
        }
        out["per_rq"][rq] = per
        # Flag configs whose |delta_pct| > 20.
        flagged = df[df[delta_pct[0]].abs() > 20.0].copy()
        for _, row in flagged.iterrows():
            out["flagged_configs"].append(
                {
                    "rq": rq,
                    "delta_pct": float(row[delta_pct[0]]),
                    "config": {
                        k: row[k]
                        for k in df.columns
                        if k not in delta_pct
                        and not k.startswith("within_")
                        and not k.startswith("delta_")
                    },
                }
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare rep1 and rep2 summary CSVs.")
    parser.add_argument("--summary", type=Path, default=Path("data/summaries"))
    args = parser.parse_args(argv)

    comparisons: dict[str, pd.DataFrame] = {}
    pairs = (
        ("rq1", compare_rq1),
        ("rq2", compare_rq2),
        ("rq3", compare_rq3),
        ("rq4", compare_rq4),
    )
    for rq, fn in pairs:
        rep1 = args.summary / f"{rq}_summary_rep1.csv"
        rep2 = args.summary / f"{rq}_summary_rep2.csv"
        if not rep1.is_file() or not rep2.is_file():
            print(f"skipping {rq}: missing {rep1.name if not rep1.is_file() else rep2.name}")
            continue
        df = fn(rep1, rep2)
        comparisons[rq] = df
        print(f"  {rq}: {len(df)} configs compared")

    if not comparisons:
        print("nothing to compare — no rep1/rep2 pairs found")
        return 0

    # Concatenate per-RQ tables with an "rq" column tag so the merged
    # report carries enough provenance to be split later.
    tagged = []
    for rq, df in comparisons.items():
        d = df.copy()
        d.insert(0, "rq", rq)
        tagged.append(d)
    report = pd.concat(tagged, ignore_index=True, sort=False)
    report_path = args.summary / "divergence_report.csv"
    report.to_csv(report_path, index=False)
    print(f"wrote {report_path} ({len(report)} rows)")

    summary = summarize(comparisons)
    summary_path = args.summary / "divergence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"wrote {summary_path}")

    for rq, stats in summary["per_rq"].items():
        print(
            f"  {rq}: {stats['n_configs']:>3d} configs — "
            f"{stats['pct_within_5']:5.1f}% within 5%, "
            f"{stats['pct_within_10']:5.1f}% within 10%, "
            f"{stats['pct_within_20']:5.1f}% within 20%, "
            f"max |Δ%|={stats['max_abs_delta_pct']:.1f}"
        )
    if summary["flagged_configs"]:
        print(f"flagged for investigation: {len(summary['flagged_configs'])} configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
