/* IPv4 LPM forwarder for RQ1.
 *
 * Pipeline depth one step above ``l2_forward.p4``: parser walks
 * Ethernet → IPv4, LPM table sets next-hop MAC and egress port, and
 * the egress control optionally recomputes the IPv4 checksum after
 * TTL decrement. Probe frames (etherType 0x88B5) carry the shared
 * ``instrument_t`` header between Ethernet and IPv4 so the harness
 * recovers BMv2 ingress and egress timestamps without external clock
 * alignment.
 */
#include <core.p4>
#include <v1model.p4>
#include "include/instrument.p4h"

const bit<16> ETHERTYPE_IPV4 = 0x0800;

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

struct headers {
    ethernet_t   ethernet;
    instrument_t instrument;
    ipv4_t       ipv4;
}

struct metadata {}

parser MyParser(packet_in pkt, out headers hdr, inout metadata meta,
                inout standard_metadata_t std) {
    state start {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_PROBE: parse_instrument;
            ETHERTYPE_IPV4:  parse_ipv4;
            default: accept;
        }
    }
    state parse_instrument {
        pkt.extract(hdr.instrument);
        transition parse_ipv4;
    }
    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
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
    apply {
        if (hdr.instrument.isValid()) {
            hdr.instrument.egress_ts = (bit<48>) std.egress_global_timestamp;
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
        pkt.emit(hdr.instrument);
        pkt.emit(hdr.ipv4);
    }
}

V1Switch(MyParser(), MyVerifyChecksum(), MyIngress(), MyEgress(),
         MyComputeChecksum(), MyDeparser()) main;
