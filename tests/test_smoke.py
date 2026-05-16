"""End-to-end smoke test for Week 2 Phase A.

Three tests:

1. ``test_p4_programs_compile`` — runs ``p4c`` over every program in
   ``p4/``, asserts each produces a BMv2 JSON and a P4Info artifact.

2. ``test_single_switch_bring_up`` — uses ``topologies.single_switch``
   to build h1↔s1↔h2 against ``p4/l3_lpm.p4``, starts the network,
   programs forwarding entries via the sync ``P4RuntimeClient``,
   seeds static ARP from each host, and asserts a 3-ping h1→h2 round
   trip succeeds with 0% loss.

3. ``test_system_info_runs_in_smoke_context`` — quick re-check that
   ``runner.system_info.capture()`` returns valid data under the
   integration test harness (so the data this test would record from
   a real run is sane).

Test 2 is the validation gate for Phase A.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
P4_DIR = REPO_ROOT / "p4"
P4_PROGRAMS = sorted(p for p in P4_DIR.glob("*.p4") if p.is_file())


# ---------------------------------------------------------------------------
# Test 1: every P4 program compiles cleanly.
# ---------------------------------------------------------------------------


@pytest.mark.requires_p4c
@pytest.mark.parametrize("p4_path", P4_PROGRAMS, ids=[p.name for p in P4_PROGRAMS])
def test_p4_programs_compile(p4_path: Path, tmp_path: Path) -> None:
    base = p4_path.stem
    json_out = tmp_path / f"{base}.json"
    p4info_out = tmp_path / f"{base}.p4info.txtpb"
    cmd = [
        "p4c",
        "--target",
        "bmv2",
        "--arch",
        "v1model",
        "--p4runtime-files",
        str(p4info_out),
        str(p4_path),
        "-o",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60.0, check=False)
    assert result.returncode == 0, f"p4c failed for {p4_path.name}: stderr={result.stderr}"
    assert json_out.is_file(), f"missing {json_out.name}: stderr={result.stderr}"
    assert p4info_out.is_file(), f"missing {p4info_out.name}: stderr={result.stderr}"


# ---------------------------------------------------------------------------
# Test 2: single-switch bring-up + ping (THE Phase A gate).
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.requires_p4c
@pytest.mark.requires_bmv2
def test_single_switch_bring_up(tmp_path: Path) -> None:
    """h1 → h2 ping over l3_lpm.p4 with runtime-programmed forwarding."""
    from p4net import Network

    from topologies.single_switch import build

    topo = build(REPO_ROOT / "p4" / "l3_lpm.p4")
    net = Network(topo, log_dir=tmp_path / "logs")
    net.start()
    try:
        h1 = net.host("h1")
        h2 = net.host("h2")
        sw = net.switch("s1")

        # Static ARP on both hosts so ping doesn't have to resolve.
        h1.exec(
            [
                "ip",
                "neigh",
                "replace",
                "10.0.0.2",
                "lladdr",
                "00:00:00:00:00:02",
                "dev",
                "h1-eth0",
                "nud",
                "permanent",
            ]
        )
        h2.exec(
            [
                "ip",
                "neigh",
                "replace",
                "10.0.0.1",
                "lladdr",
                "00:00:00:00:00:01",
                "dev",
                "h2-eth0",
                "nud",
                "permanent",
            ]
        )

        # LPM entries: route /32 host routes so each direction reaches its peer.
        # The sync P4RuntimeClient programs set_nhop(mac, port).
        sw.client.insert_table_entry(
            "MyIngress.ipv4_lpm",
            {"hdr.ipv4.dst_addr": "10.0.0.1/32"},
            "MyIngress.set_nhop",
            {"nhop_mac": "00:00:00:00:00:01", "port": 1},
        )
        sw.client.insert_table_entry(
            "MyIngress.ipv4_lpm",
            {"hdr.ipv4.dst_addr": "10.0.0.2/32"},
            "MyIngress.set_nhop",
            {"nhop_mac": "00:00:00:00:00:02", "port": 2},
        )

        result = h1.exec(
            ["ping", "-4", "-c", "3", "-W", "2", "-w", "10", "10.0.0.2"],
            capture_output=True,
            check=False,
        )
        stdout = result.stdout.decode("utf-8", errors="replace")
        assert result.returncode == 0, (
            f"ping failed (rc={result.returncode}): "
            f"stderr={result.stderr.decode(errors='replace')!r}\nstdout={stdout!r}"
        )
        # 3 transmitted, 3 received, 0% loss expected.
        assert "3 received" in stdout, f"unexpected ping summary:\n{stdout}"
        assert "0% packet loss" in stdout, f"unexpected ping summary:\n{stdout}"

        # Stash the ping output for the phase report.
        (tmp_path / "ping_output.txt").write_text(stdout, encoding="utf-8")
    finally:
        net.stop()


# ---------------------------------------------------------------------------
# Test 3: system_info under the integration test harness.
# ---------------------------------------------------------------------------


def test_system_info_runs_in_smoke_context() -> None:
    from runner.system_info import capture

    info = capture()
    assert info["p4net_version"]
    # p4c / bmv2 may or may not be on PATH in a pure-unit run; the
    # integration tests above gate those binaries, so here we only assert
    # the call completed.
    assert isinstance(info["git_sha"], str)


# Silence the unused-import warning if uuid/shutil aren't used yet.
_ = (uuid, shutil)
