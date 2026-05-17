"""Aggregation and plotting pipeline for the p4net-perf-study harness.

Modules:

* :mod:`analysis.aggregate` — reads ``data/raw/*.jsonl`` and writes
  per-RQ summary CSVs to ``data/summaries/``. The CSVs are the
  canonical analysis input for the paper's §5 through §8 plots.
"""
