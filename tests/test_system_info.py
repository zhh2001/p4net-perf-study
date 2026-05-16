"""Unit tests for ``runner.system_info``. No sudo / no BMv2 required."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runner.system_info import _EXPECTED_KEYS, capture, main


def test_capture_has_all_expected_keys() -> None:
    info = capture()
    assert set(info.keys()) == _EXPECTED_KEYS


def test_capture_values_have_plausible_types() -> None:
    info = capture()
    assert isinstance(info["cpu_model"], str)
    assert isinstance(info["cpu_cores_physical"], int)
    assert isinstance(info["cpu_cores_logical"], int)
    assert info["cpu_cores_logical"] >= info["cpu_cores_physical"] >= 1
    assert isinstance(info["cpu_max_freq_mhz"], float)
    assert info["cpu_max_freq_mhz"] >= 0
    assert isinstance(info["ram_total_gb"], float)
    assert info["ram_total_gb"] > 0
    assert isinstance(info["kernel_version"], str) and info["kernel_version"]
    assert isinstance(info["distro"], str)
    assert isinstance(info["wsl_version"], str) and info["wsl_version"]
    assert isinstance(info["python_version"], str) and info["python_version"]
    assert isinstance(info["p4net_version"], str) and info["p4net_version"]
    assert isinstance(info["bmv2_version"], str)
    assert isinstance(info["p4c_version"], str)
    assert isinstance(info["git_sha"], str)
    assert isinstance(info["git_dirty"], bool)
    assert isinstance(info["timestamp_utc"], str) and "T" in info["timestamp_utc"]


def test_main_writes_to_file(tmp_path: Path) -> None:
    out_path = tmp_path / "info.json"
    rc = main(["--out", str(out_path)])
    assert rc == 0
    blob = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(blob.keys()) == _EXPECTED_KEYS


def test_main_prints_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([])
    assert rc == 0
    captured = capsys.readouterr()
    blob = json.loads(captured.out)
    assert set(blob.keys()) == _EXPECTED_KEYS
