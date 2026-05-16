"""Workload modules used by the measurement runner.

Each workload exposes a callable or context manager that the runner
invokes during a single measurement config. Workloads are responsible
for spawning their own per-host child processes inside the
:class:`p4net.Network` namespaces; the runner only owns the network
lifecycle and the JSONL output.

Modules:

* ``latency_probe`` — RQ1 switch-transit probe (L3 only for now).
* ``background_traffic`` — iperf3-driven steady-state load.
"""
