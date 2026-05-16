/* Minimal L2 forwarder — baseline pipeline for RQ1 latency.
 *
 * Single match-action stage: exact match on destination MAC, action
 * sets the egress port. Probe frames (etherType 0x88B5) carry the
 * shared `instrument_t` header; the data plane writes ingress and
 * egress timestamps from BMv2's global clocks. Background traffic
 * uses normal etherTypes and is forwarded by the same table without
 * instrumentation overhead.
 */
#include <core.p4>
#include <v1model.p4>
#include "include/instrument.p4h"

header ethernet_t {
    bit<48> dst_addr;
    bit<48> src_addr;
    bit<16> ether_type;
}

struct headers {
    ethernet_t  ethernet;
    instrument_t instrument;
}

struct metadata {}

parser MyParser(packet_in pkt, out headers hdr, inout metadata meta,
                inout standard_metadata_t std) {
    state start {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_PROBE: parse_instrument;
            default: accept;
        }
    }
    state parse_instrument {
        pkt.extract(hdr.instrument);
        transition accept;
    }
}

control MyVerifyChecksum(inout headers hdr, inout metadata meta) { apply {} }

control MyIngress(inout headers hdr, inout metadata meta,
                  inout standard_metadata_t std) {
    action set_egress(bit<9> port) {
        std.egress_spec = port;
    }
    action drop() {
        mark_to_drop(std);
    }
    table mac_forward {
        key = {
            hdr.ethernet.dst_addr: exact;
        }
        actions = {
            set_egress;
            drop;
        }
        default_action = drop();
        size = 1024;
    }
    apply {
        mac_forward.apply();
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

control MyComputeChecksum(inout headers hdr, inout metadata meta) { apply {} }

control MyDeparser(packet_out pkt, in headers hdr) {
    apply {
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.instrument);
    }
}

V1Switch(MyParser(), MyVerifyChecksum(), MyIngress(), MyEgress(),
         MyComputeChecksum(), MyDeparser()) main;
