/* IPv4 LPM forwarder + single-hop INT shim insertion for RQ1 and RQ3.
 *
 * Pipeline: parse Ethernet → IPv4 (with optional probe instrument),
 * apply IPv4 LPM forwarding, then insert a 14-byte INT shim header
 * between Ethernet and IPv4 at egress. Shim format matches the
 * reference design used by p4net 1.7's ``int_multi_hop`` example so
 * RQ1 (single hop) and RQ3 (multi-hop) measurements share a comparable
 * wire format.
 *
 * INT shim layout (14 bytes total, big-endian on the wire):
 *
 *     bit<8>  switch_id            — controller-configured identifier
 *     bit<48> ingress_timestamp_us — BMv2 ingress_global_timestamp
 *     bit<16> egress_port          — std.egress_spec
 *     bit<16> queue_depth          — std.deq_qdepth
 *     bit<16> next_proto           — original Ethernet etherType
 *     bit<8>  reserved             — zero
 *
 * Outer Ethernet ``ether_type`` is rewritten to 0x88B6 (INT shim
 * identifier) when the shim is inserted. Probe-instrument frames
 * (etherType 0x88B5) bypass INT insertion entirely so the latency
 * measurement separates "INT cost" from "instrument cost".
 *
 * Switch identity comes from a register written by the control plane
 * at startup (same pattern as p4net 1.2+).
 */
#include <core.p4>
#include <v1model.p4>
#include "include/instrument.p4h"

const bit<16> ETHERTYPE_IPV4 = 0x0800;
const bit<16> ETHERTYPE_INT  = 0x88B6;
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
    bit<48> ingress_timestamp_us;
    bit<16> egress_port;
    bit<16> queue_depth;
    bit<16> next_proto;
    bit<8>  reserved;
}

struct headers {
    ethernet_t   ethernet;
    int_shim_t   int_shim;
    ipv4_t       ipv4;
    instrument_t instrument;
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
        /* INT processing is probe-only: only frames whose IPv4 protocol
         * was 0xFD reached parse_instrument and have hdr.instrument valid.
         * Background IPv4 traffic (TCP/UDP/ICMP/etc.) bypasses both the
         * shim insertion and the ETHERTYPE_INT rewrite so the receiver
         * kernel still sees ether_type 0x0800 and delivers to its IP
         * stack — necessary for iperf3 and other userspace consumers. */
        if (hdr.instrument.isValid()) {
            bit<8> sid;
            switch_id_reg.read(sid, 0);
            hdr.instrument.egress_ts = (bit<48>) std.egress_global_timestamp;
            hdr.int_shim.setValid();
            hdr.int_shim.switch_id            = sid;
            hdr.int_shim.ingress_timestamp_us = (bit<48>) std.ingress_global_timestamp;
            hdr.int_shim.egress_port          = (bit<16>) std.egress_spec;
            hdr.int_shim.queue_depth          = (bit<16>) std.deq_qdepth;
            hdr.int_shim.next_proto           = hdr.ethernet.ether_type;
            hdr.int_shim.reserved             = 0;
            hdr.ethernet.ether_type           = ETHERTYPE_INT;
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
        pkt.emit(hdr.int_shim);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.instrument);
    }
}

V1Switch(MyParser(), MyVerifyChecksum(), MyIngress(), MyEgress(),
         MyComputeChecksum(), MyDeparser()) main;
