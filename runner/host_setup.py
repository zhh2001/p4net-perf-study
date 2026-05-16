"""Host-veth runtime setup shared between the runner and tests.

BMv2 forwards packets without recomputing L4 (TCP/UDP) checksums.
Linux's default veth offload ships outgoing TCP/UDP frames with the
L4 checksum field zeroed (offloaded to a NIC that doesn't exist for a
software veth), so when BMv2 hands them back to the receiver kernel,
the IP-layer stack rejects them as corrupt and the iperf3 / nc / etc.
listener never sees any payload. Disabling tx/rx checksum offload and
the related segmentation offload knobs forces the kernel to compute
checksums in software before the packet leaves the netns, after which
BMv2 can be a transparent middlebox for stateful L4 flows.

Probes (raw IPv4 proto=0xFD, raw Ethernet 0x88B5) carry no L4 checksum
and are unaffected, which is why Phase B's latency results were valid
even though that pilot's background traffic was almost certainly being
dropped at the receiver kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import p4net


def disable_l4_offload(net: p4net.Network, host_names: list[str]) -> None:
    """Disable veth TX/RX checksum + segmentation offload for each host.

    Called by the runner after ``Network.start()`` but before any traffic
    is generated. Failures on individual hosts are swallowed because the
    offload knobs are best-effort — some kernels expose only a subset.
    """
    for host_name in host_names:
        iface = f"{host_name}-eth0"
        net.host(host_name).exec(
            [
                "ethtool",
                "-K",
                iface,
                "tx",
                "off",
                "rx",
                "off",
                "tso",
                "off",
                "gso",
                "off",
                "gro",
                "off",
            ],
            capture_output=True,
            check=False,
        )
