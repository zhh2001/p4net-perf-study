# p4net-perf-study

## Scope

This repository contains the measurement harness and analysis pipeline for a configuration-specific study of BMv2-based P4 emulation. It uses [`p4net`](https://pypi.org/project/p4net/) 1.7.0 as an unmodified external dependency. The reported numerical results characterize the evaluated BMv2 1.15.0 build, workload configurations, and WSL2-based host.

## Evaluated environment

The accepted campaigns were run on the environment recorded in each campaign's `system_info_<run-id>.json` file:

- Intel Core i5-13500H, with 8 physical and 16 logical cores visible to WSL2;
- 16 GB of physical RAM installed in the Windows host;
- 11.68 GiB `MemTotal` visible inside the WSL2 guest during the campaigns;
- Windows 11 with WSL2 Ubuntu 24.04.4 LTS;
- Python 3.12.3, p4net 1.7.0, p4c 1.2.5.10, and BMv2 1.15.0.

The 16-GB and 11.68-GiB values refer to different layers. Resource experiments ran inside the 11.68-GiB guest-visible limit.

## Prerequisites and Python setup

The runner requires Linux network namespaces and the following commands on `PATH`: `p4c`, `simple_switch_grpc`, `iperf3`, `ip`, and `ethtool`. Topology creation and veth plumbing require root privileges.

Create a Python environment and install the declared dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Capture the environment visible to the runner before a campaign:

```bash
.venv/bin/python -m runner.system_info
```

## Repository layout and artifact policy

- `p4/` contains the P4_16 v1model programs.
- `topologies/` contains the single-switch and linear-N topology builders.
- `workloads/` contains the probes, traffic drivers, control-plane operations, INT collector, and resource sampler.
- `runner/` contains the campaign orchestrator and system-information capture.
- `runner/configs/` contains the YAML campaign matrices.
- `analysis/` contains validation, two-level aggregation, table, and plotting code.
- `data/summaries/` contains the Git-tracked run-level and configuration-level CSV files and provenance JSON records used for the revised manuscript.
- `paper/` and `response/` contain the LaTeX sources.

Large raw archives under `data/raw_rep*/` are excluded from Git. In this working tree, the accepted raw records are stored under `data/raw_rep_c4/` at the exact paths recorded in the provenance JSON files. Any distributed reproduction package must provide these files separately and preserve those paths before the accepted measurements can be reaggregated. The tracked summary CSV files do not replace the raw JSON-Lines records.

## JSON-Lines measurement records

Every raw measurement record contains `run_id`, `timestamp_utc`, `rq`, `config`, `metric`, `value`, and an `extras` object containing metric-specific fields. The manifest stored beside each raw file records the resolved randomized schedule and hashes of the campaign inputs. `completion.json` records scheduled, attempted, successful, and failed cells and hashes the raw file, manifest, runner log, and system-information record.

RQ4 writes four aligned records per sample:

- `cpu_percent_total` is a separate whole-WSL2-guest CPU series retained in the raw data but not used in the manuscript's RQ4 table or figures.
- `cpu_percent_per_bmv2` is the sum of `psutil.Process.cpu_percent(interval=None)` over the directly reported BMv2 PID for each emulated switch. A value of 100% denotes one fully occupied logical CPU. It includes user and system time charged to those PIDs, but not CPU attributed to `iperf3`, the Python runner and sampler, other guest processes, or kernel activity not charged to the BMv2 PIDs.
- `rss_per_bmv2_bytes` is the sum of `psutil.Process.memory_info().rss` over the same PIDs. It is not whole-guest memory, PSS, or USS, and shared resident pages are not deduplicated. Tables convert bytes to decimal MB by dividing by 1,000,000.
- `net_io_pps_per_iface` sums RX packet rates derived from successive `psutil` counters for all switch-side veth interfaces and the measured monotonic elapsed time. It is an interface aggregate, not a single-link packet rate.

The RQ4 `resource_only` workload starts the continuous `iperf3` client/server carrier when the configured rate is positive, but starts no latency probe, control-plane workload, or packet/telemetry collector. CPU counters are primed, the first post-priming sample is retained, and subsequent samples follow a nominal 100-ms monotonic cadence.

## Accepted clean campaigns

Every accepted configuration has five fresh topology and BMv2 restarts. All configuration--repetition cells within an RQ were randomized with seed 42.

| RQ | Configuration file | Configurations | Independent runs | Accepted run ID |
| ---: | --- | ---: | ---: | --- |
| 1 | `runner/configs/rq1_c4.yaml` | 37 | 185 | `f4f5fe32-df82-439c-8c61-c6f8f662cdee` |
| 2 | `runner/configs/rq2_c4.yaml` | 42 | 210 | `122b7401-9adc-40db-8d2d-b789b1968d61` |
| 3 | `runner/configs/rq3_c4.yaml` | 6 | 30 | `51661e3f-940f-499d-8e24-95d9b2b70878` |
| 4 | `runner/configs/rq4_c4.yaml` | 8 | 40 | `ae42b961-22af-40d9-99a1-1694e83275cc` |

The accepted campaign paths and hashes are authoritative in `data/summaries/rqN_provenance_c4_clean.json`. In particular, RQ4 contains eight configurations and 40 independent runs. Counts of metric-summary rows must not be interpreted as configuration counts.

## Running a campaign

The following smoke campaign checks the privileged execution path without running the full matrices:

```bash
sudo -E env PATH="$PATH" .venv/bin/python -m runner.runner \
  --config runner/configs/c4_smoke.yaml \
  --output data/raw_smoke
```

Run a full RQ campaign by selecting its configuration file. For example, RQ4 is launched with:

```bash
sudo -E env PATH="$PATH" .venv/bin/python -m runner.runner \
  --config runner/configs/rq4_c4.yaml \
  --output data/raw_rep_c4
```

Each invocation creates a new UUID and writes a JSON-Lines file, a system-info file, and an `<campaign>_<run-id>_artifacts/` directory. The full campaigns are long-running experiments and should not be started merely to check the analysis pipeline.

## Reaggregating the accepted raw records

When the separately stored `data/raw_rep_c4/` archive is present, reproduce the run-level CSV, configuration-level CSV, and provenance record for each accepted campaign:

```bash
.venv/bin/python -m analysis.aggregate_clean \
  --raw-file data/raw_rep_c4/rq1_c4_clean_restarts_f4f5fe32-df82-439c-8c61-c6f8f662cdee.jsonl \
  --rq 1 --summary data/summaries --label c4_clean

.venv/bin/python -m analysis.aggregate_clean \
  --raw-file data/raw_rep_c4/rq2_c4_clean_restarts_122b7401-9adc-40db-8d2d-b789b1968d61.jsonl \
  --rq 2 --summary data/summaries --label c4_clean

.venv/bin/python -m analysis.aggregate_clean \
  --raw-file data/raw_rep_c4/rq3_c4_clean_restarts_51661e3f-940f-499d-8e24-95d9b2b70878.jsonl \
  --rq 3 --summary data/summaries --label c4_clean

.venv/bin/python -m analysis.aggregate_clean \
  --raw-file data/raw_rep_c4/rq4_c4_clean_restarts_ae42b961-22af-40d9-99a1-1694e83275cc.jsonl \
  --rq 4 --summary data/summaries --label c4_clean
```

The clean aggregator rejects source-hash mismatches, incomplete campaigns, unexpected run IDs or metrics, schedule mismatches, missing repetitions, and misaligned RQ4 samples before writing outputs. Within-run observations are summarized first; the configuration-level CSV then reports the median, minimum, and maximum of the five run-level statistics.

To verify all files referenced by the accepted provenance records without rewriting the summaries, run this standard-library-only check while the raw archive is present:

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

for record_path in sorted(Path("data/summaries").glob("rq*_provenance_c4_clean.json")):
    record = json.loads(record_path.read_text(encoding="utf-8"))
    for label, item in record["files"].items():
        path = Path(item["path"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise SystemExit(f"HASH MISMATCH: {record_path}: {label}: {path}")
    print(f"VALID: {record_path}")
PY
```

## Tests

Run the non-privileged test suite and static checks with:

```bash
.venv/bin/python -m pytest -m "not integration"
.venv/bin/python -m ruff check .
```

Tests requiring network namespaces, p4c, or BMv2 are opt-in; see `tests/conftest.py` for the corresponding flags. Run them under `sudo -E` with the virtual-environment Python and the required binaries on `PATH`.

## License

Apache 2.0; see [LICENSE](LICENSE).

## Citation

A publication citation will be added when it becomes available.
