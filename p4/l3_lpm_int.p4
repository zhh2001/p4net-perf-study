/* IPv4 LPM forwarder + single-hop INT shim for RQ1 (INT-cost cell)
 * and RQ4 (single-hop INT process cost).
 *
 * Wire format (probe path only, identified by IPv4 protocol 0xFD):
 *
 *     Ethernet (14)       — ether_type 0x0800 (unchanged through pipeline)
 *     IPv4    (20)        — proto = 0xFD
 *     instrument (12)     — ingress_ts + egress_ts written by data plane
 *     int_shim   (13)     — switch_id + per-hop ingress_ts + egress_ts
 *                           (emitted only when the probe enters the pipeline)
 *     <payload>           — sequence + padding from sender
 *
 * Phase G restructure: the previous design rewrote the outer Ethernet
 * etherType to 0x88B6 (INT) and inserted the shim between Ethernet and
 * IPv4, which prevented the standard L3 latency-probe receiver from
 * decoding the IPv4 layer — every l3_lpm_int RQ1 matrix cell came back
 * with zero captured samples. The new format preserves the
 * Ethernet+IPv4 outer envelope so generic IPv4 capture tools work
 * unchanged; the int_shim is appended after the instrument header and
 * the receiver simply ignores the extra bytes when it only needs
 * switch_transit_us. Background (non-probe) traffic still bypasses
 * shim insertion entirely.
 *
 * INT shim layout (13 bytes, big-endian on the wire):
 *
 *     bit<8>  switch_id   — controller-configured (register at index 0)
 *     bit<48> ingress_ts  — std.ingress_global_timestamp
 *     bit<48> egress_ts   — std.egress_global_timestamp
 */
#include <core.p4>
#include <v1model.p4>
#include "include/instrument.p4h"

const bit<16> ETHERTYPE_IPV4 = 0x0800;
const bit<8>  IP_PROTO_PROBE = 0xFD;

header ethernet_t {
    bit<48> dst_addr;
    bit<48> src_addr;
    bit<16> ether_type;
}

header ipv4_t {
    bit<4>  version;
    bit<4>  ihl;
    bit<8>  diffserv;
    bit<16> total_len;
    bit<16> identification;
    bit<3>  flags;
    bit<13> frag_offset;
    bit<8>  ttl;
    bit<8>  protocol;
    bit<16> hdr_checksum;
    bit<32> src_addr;
    bit<32> dst_addr;
}

header int_shim_t {
    bit<8>  switch_id;
    bit<48> ingress_ts;
    bit<48> egress_ts;
}

struct headers {
    ethernet_t   ethernet;
    ipv4_t       ipv4;
    instrument_t instrument;
    int_shim_t   int_shim;
}

struct metadata {}

parser MyParser(packet_in pkt, out headers hdr, inout metadata meta,
                inout standard_metadata_t std) {
    state start {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }
    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            IP_PROTO_PROBE: parse_instrument;
            default: accept;
        }
    }
    state parse_instrument {
        pkt.extract(hdr.instrument);
        /* The sender never emits an int_shim on its own; only switches
         * append it on egress. Leave int_shim invalid here so the
         * remaining bytes (sequence + padding) stay in the packet's
         * unparsed tail and pass through unmodified. */
        transition accept;
    }
}

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {
    apply {
        verify_checksum(
            hdr.ipv4.isValid(),
            {
                hdr.ipv4.version, hdr.ipv4.ihl, hdr.ipv4.diffserv,
                hdr.ipv4.total_len, hdr.ipv4.identification,
                hdr.ipv4.flags, hdr.ipv4.frag_offset,
                hdr.ipv4.ttl, hdr.ipv4.protocol,
                hdr.ipv4.src_addr, hdr.ipv4.dst_addr
            },
            hdr.ipv4.hdr_checksum, HashAlgorithm.csum16);
    }
}

control MyIngress(inout headers hdr, inout metadata meta,
                  inout standard_metadata_t std) {
    action drop() {
        mark_to_drop(std);
    }
    action set_nhop(bit<48> nhop_mac, bit<9> port) {
        hdr.ethernet.src_addr = hdr.ethernet.dst_addr;
        hdr.ethernet.dst_addr = nhop_mac;
        std.egress_spec = port;
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
    }
    table ipv4_lpm {
        key = {
            hdr.ipv4.dst_addr: lpm;
        }
        actions = {
            set_nhop;
            drop;
        }
        default_action = drop();
        size = 1024;
    }
    apply {
        if (hdr.ipv4.isValid()) {
            ipv4_lpm.apply();
        }
        if (hdr.instrument.isValid()) {
            hdr.instrument.ingress_ts = (bit<48>) std.ingress_global_timestamp;
        }
    }
}

control MyEgress(inout headers hdr, inout metadata meta,
                 inout standard_metadata_t std) {
    register<bit<8>>(1) switch_id_reg;

    apply {
        /* Probe-only: only frames whose IPv4 protocol was 0xFD reached
         * parse_instrument and have hdr.instrument valid. Background
         * IPv4 traffic bypasses both the timestamp write and the shim
         * insertion entirely, keeping ether_type 0x0800 intact for the
         * receiver kernel. */
        if (hdr.instrument.isValid()) {
            bit<8> sid;
            switch_id_reg.read(sid, 0);
            hdr.instrument.egress_ts = (bit<48>) std.egress_global_timestamp;
            hdr.int_shim.setValid();
            hdr.int_shim.switch_id  = sid;
            hdr.int_shim.ingress_ts = (bit<48>) std.ingress_global_timestamp;
            hdr.int_shim.egress_ts  = (bit<48>) std.egress_global_timestamp;
        }
    }
}

control MyComputeChecksum(inout headers hdr, inout metadata meta) {
    apply {
        update_checksum(
            hdr.ipv4.isValid(),
            {
                hdr.ipv4.version, hdr.ipv4.ihl, hdr.ipv4.diffserv,
                hdr.ipv4.total_len, hdr.ipv4.identification,
                hdr.ipv4.flags, hdr.ipv4.frag_offset,
                hdr.ipv4.ttl, hdr.ipv4.protocol,
                hdr.ipv4.src_addr, hdr.ipv4.dst_addr
            },
            hdr.ipv4.hdr_checksum, HashAlgorithm.csum16);
    }
}

control MyDeparser(packet_out pkt, in headers hdr) {
    apply {
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.instrument);
        pkt.emit(hdr.int_shim);
    }
}

V1Switch(MyParser(), MyVerifyChecksum(), MyIngress(), MyEgress(),
         MyComputeChecksum(), MyDeparser()) main;
