# p4net-perf-study

## About

Measurement harness for an upcoming paper characterizing software P4
emulation in BMv2: per-pipeline latency, control-plane scaling, and
telemetry fidelity. The harness is a thin layer over
[`p4net`](https://pypi.org/project/p4net/) 1.7.0 — it does not modify p4net
and uses it as an external library only. Release notes and DOI will be
added at paper submission time.

## Status

Pre-release. The harness is under active development for a research paper
measurement campaign. Reproducibility instructions will be finalized at
paper submission.

## Hardware target

Reference rig: Intel Core i5-13500H (6P + 8E cores), 8 GB RAM available to
the Python runtime, WSL2 Ubuntu 24.04 on a Windows 11 host. The harness is
portable across Linux distributions but absolute numbers will differ on
other hardware — the paper reports per-rig results, not vendor-level
benchmarks.

## Setup (preliminary)

```bash
pip install -r requirements.txt
sudo apt install p4lang-p4c p4lang-bmv2
```

Several measurement steps (`Network.start`, namespace creation, veth
plumbing) require root. The smoke test and any integration tests must be
invoked under `sudo -E env PATH=$PATH python -m pytest ...` so the
environment and `PATH` survive privilege escalation.

## Repository layout

- `p4/` — P4_16 v1model source for the measurement matrix.
- `topologies/` — Python topology builders backed by `p4net.topo`.
- `workloads/` — packet generators, traffic profiles.
- `runner/` — measurement orchestrator, system-info capture, config matrices.
- `runner/configs/` — campaign configuration files (YAML).
- `analysis/` — post-processing scripts and notebooks for the raw data.
- `data/raw/` — per-run JSONL output (gitignored).
- `data/summaries/` — summarized statistics committed for paper figures.
- `tests/` — pytest suite (unit + integration).

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

BibTeX citation will be provided here at paper publication.
