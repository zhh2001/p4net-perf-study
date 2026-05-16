"""Capture a system fingerprint for measurement reproducibility.

Records hardware (CPU, RAM), software (kernel, distro, Python, p4net,
BMv2, p4c), and repo state (git SHA + dirty flag) at the time the
function is called. Output is JSON-serialisable and intended to be
attached to every measurement run as ``system_info.json``.

CLI: ``python -m runner.system_info`` pretty-prints to stdout;
``python -m runner.system_info --out path.json`` writes to disk.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import p4net


def _read_file(path: str) -> str:
    """Return file contents or an empty string on any error."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def _shell(argv: list[str]) -> tuple[str, str, int]:
    """Run ``argv``; return ``(stdout, stderr, returncode)`` (empty on error)."""
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        return proc.stdout, proc.stderr, proc.returncode
    except (OSError, subprocess.TimeoutExpired):
        return "", "", -1


def _cpu_model() -> str:
    text = _read_file("/proc/cpuinfo")
    for line in text.splitlines():
        if line.startswith("model name"):
            _, _, value = line.partition(":")
            return value.strip()
    return ""


def _cpu_core_counts() -> tuple[int, int]:
    """Return ``(physical, logical)`` core counts from /proc/cpuinfo."""
    text = _read_file("/proc/cpuinfo")
    physical_ids: set[tuple[str, str]] = set()
    logical = 0
    current_phys = ""
    for line in text.splitlines():
        if line.startswith("physical id"):
            current_phys = line.partition(":")[2].strip()
        elif line.startswith("core id"):
            physical_ids.add((current_phys, line.partition(":")[2].strip()))
        elif line.startswith("processor"):
            logical += 1
    physical = len(physical_ids) or logical
    return physical, logical


def _cpu_max_freq_mhz() -> float:
    out, _, rc = _shell(["lscpu"])
    if rc == 0:
        for line in out.splitlines():
            if "CPU max MHz" in line:
                try:
                    return float(line.partition(":")[2].strip())
                except ValueError:
                    pass
    text = _read_file("/proc/cpuinfo")
    best = 0.0
    for line in text.splitlines():
        if line.startswith("cpu MHz"):
            with contextlib.suppress(ValueError):
                best = max(best, float(line.partition(":")[2].strip()))
    return best


def _ram_total_gb() -> float:
    text = _read_file("/proc/meminfo")
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return round(int(parts[1]) / (1024 * 1024), 2)
                except ValueError:
                    return 0.0
    return 0.0


def _distro() -> str:
    text = _read_file("/etc/os-release")
    for line in text.splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.partition("=")[2].strip().strip('"')
    return ""


def _wsl_indicator() -> str:
    release = os.uname().release.lower()
    if "microsoft" in release or "wsl" in release:
        return release
    return "bare-metal"


def _bmv2_version() -> str:
    out, err, _ = _shell(["simple_switch_grpc", "--version"])
    combined = (out + err).strip()
    # BMv2 prints just the version on a single line, e.g. "1.15.0-2bdd0b7b".
    return combined.splitlines()[0] if combined else ""


def _p4c_version() -> str:
    out, _, _ = _shell(["p4c", "--version"])
    # e.g. "p4c 1.2.5.10 (SHA: 8b6de3c57 BUILD: Release)"
    return out.strip().splitlines()[0] if out.strip() else ""


def _git_state() -> tuple[str, bool]:
    """Return ``(commit_sha, dirty)`` for the repo containing this file.

    Operates relative to the file's enclosing repo, so the result is
    deterministic regardless of where the harness was invoked from.
    """
    cwd = str(Path(__file__).resolve().parent.parent)
    sha_out, _, sha_rc = _shell(["git", "-C", cwd, "rev-parse", "HEAD"])
    status_out, _, status_rc = _shell(["git", "-C", cwd, "status", "--porcelain"])
    sha = sha_out.strip() if sha_rc == 0 else ""
    dirty = bool(status_out.strip()) if status_rc == 0 else False
    return sha, dirty


def capture() -> dict[str, Any]:
    """Return a dict with every fingerprint field documented in the module
    docstring. All fields are present; values may be empty strings when a
    subsystem is unavailable (e.g. BMv2 not on PATH), so callers can rely
    on the schema without ``.get()``."""
    physical, logical = _cpu_core_counts()
    sha, dirty = _git_state()
    return {
        "cpu_model": _cpu_model(),
        "cpu_cores_physical": physical,
        "cpu_cores_logical": logical,
        "cpu_max_freq_mhz": _cpu_max_freq_mhz(),
        "ram_total_gb": _ram_total_gb(),
        "kernel_version": os.uname().release,
        "distro": _distro(),
        "wsl_version": _wsl_indicator(),
        "python_version": sys.version.split()[0],
        "p4net_version": p4net.__version__,
        "bmv2_version": _bmv2_version(),
        "p4c_version": _p4c_version(),
        "git_sha": sha,
        "git_dirty": dirty,
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }


# Sanity: surface obviously-missing keys at module load time if someone
# edits ``capture()`` and forgets one. Cheap and fail-fast.
_EXPECTED_KEYS = frozenset(
    {
        "cpu_model",
        "cpu_cores_physical",
        "cpu_cores_logical",
        "cpu_max_freq_mhz",
        "ram_total_gb",
        "kernel_version",
        "distro",
        "wsl_version",
        "python_version",
        "p4net_version",
        "bmv2_version",
        "p4c_version",
        "git_sha",
        "git_dirty",
        "timestamp_utc",
    }
)


def _verify_schema(info: Mapping[str, Any]) -> None:
    missing = _EXPECTED_KEYS - info.keys()
    extra = info.keys() - _EXPECTED_KEYS
    if missing or extra:
        raise RuntimeError(
            f"system_info schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture system fingerprint as JSON for measurement reproducibility.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write JSON to this path instead of stdout.",
    )
    args = parser.parse_args(argv)

    info = capture()
    _verify_schema(info)
    blob = json.dumps(info, indent=2, sort_keys=True)
    if args.out is None:
        sys.stdout.write(blob + "\n")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(blob + "\n", encoding="utf-8")
    return 0


# Keep the re imports used by future helpers alive; avoid an unused-import lint.
_ = re

if __name__ == "__main__":
    raise SystemExit(main())
