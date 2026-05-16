"""Topology builders backed by ``p4net.topo``.

Each module in this package exposes a ``build(p4_program)`` factory that
returns a configured :class:`p4net.topo.Topology`. Builders are pure —
they do not start a network, do not run any subprocesses, and have no
side effects at import time.
"""
