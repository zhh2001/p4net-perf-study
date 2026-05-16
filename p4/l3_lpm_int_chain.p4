/* L3 LPM forwarder + stacked multi-hop INT for RQ3.
 *
 * Pipeline: parse Ethernet → IPv4 (with probe instrument header) →
 * int_meta(hop_count) → ``hop_count`` previously-pushed INT shims →
 * apply IPv4 LPM forwarding → in egress, push this switch's shim
 * onto the front of the stack and increment hop_count. The receiver
 * decodes the stack to recover per-hop timing.
 *
 * Wire format (probe path only, identified by IPv4 protocol 0xFD):
 *
 *     Ethernet (14)
 *     IPv4(proto=0xFD) (20)
 *     instrument (12): ingress_ts + egress_ts  (per-flight; last hop wins)
 *     int_meta   (1):  hop_count
 *     int_stack  (N x 13): switch_id + ingress_ts + egress_ts per hop
 *                          ordered front-of-stack = most recent hop
 *     <payload: sequence + padding>
 *
 * Background IPv4 traffic (TCP/UDP/etc., protocol != 0xFD) bypasses
 * all INT processing and is forwarded with ether_type 0x0800 intact —
 * the same conditional pattern as p4/l3_lpm_int.p4 after the Phase C
 * fix. iperf3 background traffic therefore traverses this program
 * cleanly.
 *
 * Switch identity comes from a register written by the controller
 * before the measurement window starts (one register entry per switch,
 * holding that switch's small-integer ID).
 *
 * Up to 8 stacked shims; the parser hard-stops at index 7 to bound
 * processing cost. Longer chains would overflow and lose the oldest
 * data; for the Phase E pilot N <= 3.
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

header int_meta_t {
    bit<8>  hop_count;
}

header int_shim_t {
    bit<8>  switch_id;
    bit<48> ingress_ts;
    bit<48> egress_ts;
}

struct headers {
    ethernet_t    ethernet;
    ipv4_t        ipv4;
    instrument_t  instrument;
    int_meta_t    int_meta;
    int_shim_t[8] int_stack;
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
        transition parse_int_meta;
    }
    state parse_int_meta {
        pkt.extract(hdr.int_meta);
        transition select(hdr.int_meta.hop_count) {
            0: accept;
            default: parse_shim_0;
        }
    }
    /* Unrolled chain — one state per stack slot. Falls through to the
     * next slot when hop_count > slot index; accepts when equal. */
    state parse_shim_0 {
        pkt.extract(hdr.int_stack[0]);
        transition select(hdr.int_meta.hop_count) {
            1: accept;
            default: parse_shim_1;
        }
    }
    state parse_shim_1 {
        pkt.extract(hdr.int_stack[1]);
        transition select(hdr.int_meta.hop_count) {
            2: accept;
            default: parse_shim_2;
        }
    }
    state parse_shim_2 {
        pkt.extract(hdr.int_stack[2]);
        transition select(hdr.int_meta.hop_count) {
            3: accept;
            default: parse_shim_3;
        }
    }
    state parse_shim_3 {
        pkt.extract(hdr.int_stack[3]);
        transition select(hdr.int_meta.hop_count) {
            4: accept;
            default: parse_shim_4;
        }
    }
    state parse_shim_4 {
        pkt.extract(hdr.int_stack[4]);
        transition select(hdr.int_meta.hop_count) {
            5: accept;
            default: parse_shim_5;
        }
    }
    state parse_shim_5 {
        pkt.extract(hdr.int_stack[5]);
        transition select(hdr.int_meta.hop_count) {
            6: accept;
            default: parse_shim_6;
        }
    }
    state parse_shim_6 {
        pkt.extract(hdr.int_stack[6]);
        transition select(hdr.int_meta.hop_count) {
            7: accept;
            default: parse_shim_7;
        }
    }
    state parse_shim_7 {
        pkt.extract(hdr.int_stack[7]);
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
         * Background traffic (any other proto) bypasses entirely. */
        if (hdr.instrument.isValid()) {
            bit<8> sid;
            switch_id_reg.read(sid, 0);
            hdr.instrument.egress_ts = (bit<48>) std.egress_global_timestamp;

            /* push_front(1) shifts all valid shims back by one, leaving
             * index 0 invalid; we then setValid() and populate it with
             * this switch's measurements. The deparser emits only valid
             * stack entries in index order, so the most-recent hop is
             * the first shim on the wire. */
            hdr.int_stack.push_front(1);
            hdr.int_stack[0].setValid();
            hdr.int_stack[0].switch_id  = sid;
            hdr.int_stack[0].ingress_ts = (bit<48>) std.ingress_global_timestamp;
            hdr.int_stack[0].egress_ts  = (bit<48>) std.egress_global_timestamp;

            hdr.int_meta.hop_count = hdr.int_meta.hop_count + 1;
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
        pkt.emit(hdr.int_meta);
        pkt.emit(hdr.int_stack);
    }
}

V1Switch(MyParser(), MyVerifyChecksum(), MyIngress(), MyEgress(),
         MyComputeChecksum(), MyDeparser()) main;
