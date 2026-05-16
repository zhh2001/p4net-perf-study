"""Parameterizable linear-N-switch topology for RQ2 and RQ3.

Layout::

    h1 --- s1 --- s2 --- ... --- sN --- h2

Each switch runs the same P4 program and exposes its own gRPC endpoint
via p4net's ``RunningSwitch.client`` / ``RunningSwitch.async_client``.
Endpoint hosts are h1 (left) and h2 (right); intermediate switches have
no hosts attached.

Port assignment convention — read it as "ports point toward h2"::

    h1  (port 1) <-> s1.port1
    s1.port2     <-> s2.port1
    s2.port2     <-> s3.port1
    ...
    s(N-1).port2 <-> sN.port1
    sN.port2     <-> h2 (port 1)

So switch s_i (for 1 < i < N) has port 1 facing s_{i-1} (the h1 side)
and port 2 facing s_{i+1} (the h2 side). The endpoint switches s_1 and
s_N degenerate accordingly. For N=1, s_1 has port 1 toward h1 and port
2 toward h2 — the same convention as ``topologies.single_switch``.

For RQ2 (control-plane scaling), the topology is used as a deployment
target only: actual forwarding tables are populated by the test
workload, not by ``build()``. So this function produces a topology that
comes up cleanly even without runtime entries — packets are dropped at
each switch's default action, which is fine because the data plane is
not exercised.

``subnet_per_switch`` controls IP layout:

* ``True`` (default) — h1 sits on ``10.0.0.0/30`` and h2 sits on
  ``10.0.{n_switches}.0/30``. Useful when later phases need each link
  to be its own L3 hop (RQ3 multi-hop INT).
* ``False`` — h1=10.0.0.1/24, h2=10.0.0.2/24 — flat /24, the same
  scheme as :mod:`topologies.single_switch`. Easier when probes and
  background traffic must traverse multiple switches end-to-end with a
  single LPM /32 entry per switch.

``hosts_per_endpoint`` is reserved for future fan-out experiments and
currently must be 1.
"""

from __future__ import annotations

from pathlib import Path

from p4net.topo import Topology

# Endpoint host MAC addresses; identical regardless of subnet layout so
# call sites (and the runner) can install static ARP entries without
# threading the subnet parameter through.
H1_MAC = "00:00:00:00:00:01"
H2_MAC = "00:00:00:00:00:02"

# Flat-/24 layout endpoints (subnet_per_switch=False).
H1_IP_FLAT = "10.0.0.1/24"
H2_IP_FLAT = "10.0.0.2/24"


def _endpoint_ips_per_link(n_switches: int) -> tuple[str, str]:
    """Return ``(h1_ip, h2_ip)`` for the subnet-per-switch layout."""
    return ("10.0.0.1/30", f"10.0.{n_switches}.1/30")


def build(
    n_switches: int,
    p4_program: str | Path,
    hosts_per_endpoint: int = 1,
    subnet_per_switch: bool = True,
) -> Topology:
    """Build a linear N-switch topology. See module docstring for layout."""
    if n_switches < 1:
        raise ValueError(f"n_switches must be >= 1, got {n_switches}")
    if hosts_per_endpoint != 1:
        raise NotImplementedError(
            f"only hosts_per_endpoint=1 is supported (got {hosts_per_endpoint})"
        )

    p4_path = Path(p4_program).resolve()
    if not p4_path.is_file():
        raise FileNotFoundError(f"P4 program not found: {p4_path}")

    if subnet_per_switch:
        h1_ip, h2_ip = _endpoint_ips_per_link(n_switches)
    else:
        h1_ip, h2_ip = H1_IP_FLAT, H2_IP_FLAT

    topo = Topology()
    h1 = topo.add_host("h1", ip=h1_ip, mac=H1_MAC)
    h2 = topo.add_host("h2", ip=h2_ip, mac=H2_MAC)

    switches = [topo.add_switch(f"s{i}", p4_src=p4_path) for i in range(1, n_switches + 1)]

    # h1 -- s1: h1 has only one port, s1 uses port 1.
    topo.add_link(h1, switches[0], port_b=1)
    # s_i -- s_{i+1}: port 2 on s_i (toward h2), port 1 on s_{i+1} (toward h1).
    for i in range(n_switches - 1):
        topo.add_link(switches[i], switches[i + 1], port_a=2, port_b=1)
    # s_N -- h2: s_N uses port 2 (toward h2); h2 has only one port.
    topo.add_link(switches[-1], h2, port_a=2)

    return topo
