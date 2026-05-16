"""Steady-state UDP background traffic for the RQ1 matrix.

Wraps a pair of ``iperf3`` instances — server on the receiver host,
client on the sender host — so the runner can place the data plane
under a controlled load while latency probes measure overhead. The
client runs with ``-t 0`` (no time bound) and is terminated explicitly
by :meth:`BackgroundTraffic.stop`; this keeps the load on for the
exact duration of the probe campaign, not a wall-clock guess.

Rate semantics
--------------

* ``rate_mbps == 0`` — no-op. Both :meth:`start` and :meth:`stop` are
  still callable so the runner has a single code path; nothing spawns,
  nothing is killed, and no iperf3 binary is invoked.

* ``rate_mbps > 0`` — server binds first; once it has had a moment to
  open its UDP socket the client is launched. Cleanup is symmetric:
  client first, then server, both via ``SIGTERM`` with a hard ``SIGKILL``
  fallback.

The class is also a context manager so callers can write::

    with BackgroundTraffic(net, ..., rate_mbps=100):
        samples = run_probe(net, ..., n_probes=100)
"""

from __future__ import annotations

import contextlib
import subprocess
import time
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import p4net
    from p4net.runtime import NSProcess


DEFAULT_UDP_PORT = 5201
SERVER_BIND_GRACE_SECONDS = 0.3
TERMINATE_TIMEOUT_SECONDS = 5.0


class BackgroundTraffic:
    """Manage a single sender/receiver iperf3 UDP traffic pair."""

    def __init__(
        self,
        net: p4net.Network,
        sender_host: str,
        receiver_host: str,
        sender_ip: str,
        receiver_ip: str,
        rate_mbps: int,
        udp_port: int = DEFAULT_UDP_PORT,
        log_dir: Path | None = None,
    ) -> None:
        if rate_mbps < 0:
            raise ValueError("rate_mbps must be >= 0")
        self.net = net
        self.sender_host = sender_host
        self.receiver_host = receiver_host
        self.sender_ip = sender_ip
        self.receiver_ip = receiver_ip
        self.rate_mbps = rate_mbps
        self.udp_port = udp_port
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self._server_proc: NSProcess | None = None
        self._client_proc: NSProcess | None = None
        self._server_log_fh = None
        self._client_log_fh = None

    def start(self) -> None:
        if self.rate_mbps == 0:
            return
        if self._server_proc is not None or self._client_proc is not None:
            raise RuntimeError("BackgroundTraffic.start() already called")

        server_log, client_log = self._open_log_files()

        server_argv = [
            "iperf3",
            "-s",
            "-p",
            str(self.udp_port),
        ]
        self._server_proc = self.net.host(self.receiver_host).popen(
            server_argv,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        time.sleep(SERVER_BIND_GRACE_SECONDS)
        if self._server_proc.poll() is not None:
            rc = self._server_proc.poll()
            self._close_log_files()
            self._server_proc = None
            raise RuntimeError(f"iperf3 server exited rc={rc} before client launch")

        client_argv = [
            "iperf3",
            "-c",
            self.receiver_ip,
            "-p",
            str(self.udp_port),
            "-u",
            "-b",
            f"{self.rate_mbps}M",
            "-t",
            "0",
        ]
        self._client_proc = self.net.host(self.sender_host).popen(
            client_argv,
            stdout=client_log,
            stderr=subprocess.STDOUT,
        )

    def stop(self) -> None:
        # Terminate client first so it stops generating traffic, then the server.
        for proc_attr in ("_client_proc", "_server_proc"):
            proc = getattr(self, proc_attr)
            if proc is None:
                continue
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=TERMINATE_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=TERMINATE_TIMEOUT_SECONDS)
            setattr(self, proc_attr, None)
        self._close_log_files()

    def __enter__(self) -> BackgroundTraffic:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    # ------------------------------------------------------------------

    def _open_log_files(self):
        if self.log_dir is None:
            return subprocess.DEVNULL, subprocess.DEVNULL
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # File handles intentionally outlive this call: the iperf3
        # subprocesses write to them between start() and stop().
        self._server_log_fh = open(self.log_dir / "iperf3_server.log", "wb")  # noqa: SIM115
        self._client_log_fh = open(self.log_dir / "iperf3_client.log", "wb")  # noqa: SIM115
        return self._server_log_fh, self._client_log_fh

    def _close_log_files(self) -> None:
        for fh_attr in ("_server_log_fh", "_client_log_fh"):
            fh = getattr(self, fh_attr)
            if fh is not None:
                with contextlib.suppress(OSError):
                    fh.close()
                setattr(self, fh_attr, None)
