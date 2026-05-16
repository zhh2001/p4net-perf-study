"""Single-switch topology for RQ1 (per-pipeline latency).

    h1 (10.0.0.1/24, 00:00:00:00:00:01) ─── s1 ─── h2 (10.0.0.2/24, 00:00:00:00:00:02)

Both hosts hang off the same BMv2 instance on ports 1 and 2 respectively.
The probe direction is always h1 → h2; reverse direction is uninstrumented.

The caller passes the path (relative to repo root or absolute) of a P4
program. The harness compiles and loads it via the orchestrator at
``Network.start()``.

Static ARP between the two hosts is *not* installed by the builder —
that's a control-plane responsibility done by the measurement runner
after ``Network.start()`` (and before any probe), so the ARP step is
visible in the experiment log rather than hidden in topology setup.
"""

from __future__ import annotations

from pathlib import Path

from p4net.topo import Topology

H1_IP = "10.0.0.1/24"
H1_MAC = "00:00:00:00:00:01"
H2_IP = "10.0.0.2/24"
H2_MAC = "00:00:00:00:00:02"


def build(p4_program: str | Path) -> Topology:
    """Return a single-switch h1↔s1↔h2 topology loaded with ``p4_program``.

    ``p4_program`` is resolved relative to the current working directory
    if it's a relative path, so ``build("p4/l3_lpm.p4")`` works from the
    repo root.
    """
    p4_path = Path(p4_program).resolve()
    if not p4_path.is_file():
        raise FileNotFoundError(f"P4 program not found: {p4_path}")

    topo = Topology()
    h1 = topo.add_host("h1", ip=H1_IP, mac=H1_MAC)
    h2 = topo.add_host("h2", ip=H2_IP, mac=H2_MAC)
    s1 = topo.add_switch("s1", p4_src=p4_path)
    topo.add_link(h1, s1, port_b=1)
    topo.add_link(h2, s1, port_b=2)
    return topo
