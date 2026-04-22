# ============================================================
# parser.py - 封包欄位解析工具（v2.0.0 加強版）
#
# 新增功能：
#   - IPv6 Layer 解析（src/dst IPv6 位址、Next Header）
#   - TLS/SSL 版本識別（ClientHello / ServerHello / AppData）
#   - 加強 ICMP 類型（Destination Unreachable Code 對應）
#   - UDP Payload 長度與可能的協議猜測
#   - 封包方向判斷輔助欄位
# ============================================================

from datetime import datetime
from scapy.all import Packet
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.inet6 import IPv6, ICMPv6EchoRequest, ICMPv6EchoReply
from scapy.layers.l2 import Ether, ARP
from scapy.layers.dns import DNS, DNSQR, DNSRR
from config import (
    PROTOCOL_MAP, PORT_SERVICE_MAP, MAX_PAYLOAD_DISPLAY,
    ICMP_TYPE_MAP, ICMP_UNREACH_CODE_MAP,
    TLS_VERSION_MAP, TLS_CONTENT_TYPE_MAP, TLS_HANDSHAKE_TYPE_MAP,
)


class PacketParser:
    """
    封包解析器（加強版）
    輸入：Scapy Packet 物件
    輸出：標準化 dict，包含所有關鍵欄位
    """

    def parse(self, pkt: Packet) -> dict:
        """
        解析單一封包，回傳標準化 dict

        完整欄位說明：
          timestamp      - 封包時間戳記
          length         - 封包總長度（bytes）
          src_mac        - 來源 MAC 地址
          dst_mac        - 目標 MAC 地址
          ethertype      - 乙太幀類型（十六進位）
          src_ip         - 來源 IP（IPv4 或 IPv6）
          dst_ip         - 目標 IP（IPv4 或 IPv6）
          ip_version     - IP 版本（4 或 6）
          protocol       - 協議名稱（TCP/UDP/ICMP/DNS/ARP/TLS...）
          protocol_num   - 協議號碼（IP header proto 欄位）
          ttl            - Time To Live（IPv4）/ Hop Limit（IPv6）
          ip_flags       - IP 標誌（DF / MF）
          ip_fragment    - IP 分片偏移
          src_port       - 來源 Port
          dst_port       - 目標 Port
          service        - 服務名稱（依 dst_port 對應）
          flags          - TCP Flags 字串 / ICMP 類型描述
          seq            - TCP 序號
          ack            - TCP 確認號
          window         - TCP 滑動視窗大小
          checksum       - 封包校驗和
          icmp_type      - ICMP 類型號碼
          icmp_code      - ICMP 代碼號碼
          icmp_desc      - ICMP 類型 + Code 詳細描述
          arp_op         - ARP 操作（Request / Reply）
          arp_src_ip     - ARP 來源 IP
          arp_dst_ip     - ARP 目標 IP
          arp_src_mac    - ARP 來源 MAC（用於 Spoof 偵測）
          dns_query      - DNS 查詢域名（若為 DNS 封包）
          dns_response   - DNS 回應（域名 -> IP）
          tls_version    - TLS/SSL 版本字串
          tls_type       - TLS Content Type 字串
          tls_handshake  - TLS Handshake Type 字串（ClientHello 等）
          payload_hex    - Payload 十六進位字串（前 N bytes）
          payload_text   - Payload ASCII 表示（不可見字符 -> "."）
          payload_len    - Payload 實際長度（bytes）
        """
        record = {
            "timestamp":     self._get_timestamp(pkt),
            "length":        len(pkt),
            "src_mac":       None, "dst_mac":      None, "ethertype":    None,
            "src_ip":        None, "dst_ip":       None, "ip_version":   None,
            "protocol":      "UNKNOWN", "protocol_num": None,
            "ttl":           None, "ip_flags":     None, "ip_fragment":  None,
            "src_port":      None, "dst_port":     None, "service":      None,
            "flags":         None, "seq":          None, "ack":          None,
            "window":        None, "checksum":     None,
            "icmp_type":     None, "icmp_code":    None, "icmp_desc":    None,
            "arp_op":        None, "arp_src_ip":   None, "arp_dst_ip":   None,
            "arp_src_mac":   None,
            "dns_query":     None, "dns_response": None,
            "tls_version":   None, "tls_type":     None, "tls_handshake": None,
            "payload_hex":   None, "payload_text": None, "payload_len":  None,
        }

        self._parse_ethernet(pkt, record)

        if pkt.haslayer(IP):
            self._parse_ipv4(pkt, record)
        elif pkt.haslayer(IPv6):
            self._parse_ipv6(pkt, record)

        self._parse_transport(pkt, record)
        self._parse_application(pkt, record)
        self._parse_payload(pkt, record)

        if record["dst_port"]:
            record["service"] = PORT_SERVICE_MAP.get(record["dst_port"])

        return record

    # ── 第二層：Ethernet / ARP ───────────────────────────
    def _parse_ethernet(self, pkt, record):
        if pkt.haslayer(Ether):
            eth = pkt[Ether]
            record["src_mac"]   = eth.src
            record["dst_mac"]   = eth.dst
            record["ethertype"] = hex(eth.type)

        if pkt.haslayer(ARP):
            arp = pkt[ARP]
            record["protocol"]   = "ARP"
            record["arp_op"]     = {1: "Request", 2: "Reply"}.get(arp.op, str(arp.op))
            record["arp_src_ip"] = arp.psrc
            record["arp_dst_ip"] = arp.pdst
            record["arp_src_mac"]= arp.hwsrc
            record["src_ip"]     = arp.psrc
            record["dst_ip"]     = arp.pdst

    # ── 第三層：IPv4 ─────────────────────────────────────
    def _parse_ipv4(self, pkt, record):
        ip = pkt[IP]
        record["ip_version"]   = 4
        record["src_ip"]       = ip.src
        record["dst_ip"]       = ip.dst
        record["ttl"]          = ip.ttl
        record["protocol_num"] = ip.proto
        record["protocol"]     = PROTOCOL_MAP.get(ip.proto, str(ip.proto))
        record["ip_flags"]     = str(ip.flags)
        record["ip_fragment"]  = ip.frag
        record["checksum"]     = hex(ip.chksum) if ip.chksum else None

    # ── 第三層：IPv6 ─────────────────────────────────────
    def _parse_ipv6(self, pkt, record):
        ip6 = pkt[IPv6]
        record["ip_version"]   = 6
        record["src_ip"]       = ip6.src
        record["dst_ip"]       = ip6.dst
        record["ttl"]          = ip6.hlim          # Hop Limit
        record["protocol_num"] = ip6.nh            # Next Header
        record["protocol"]     = PROTOCOL_MAP.get(ip6.nh, str(ip6.nh))

        # ICMPv6
        if pkt.haslayer(ICMPv6EchoRequest):
            record["protocol"]  = "ICMPv6"
            record["icmp_type"] = 128
            record["icmp_desc"] = "Echo Request"
            record["flags"]     = "Echo Request"
        elif pkt.haslayer(ICMPv6EchoReply):
            record["protocol"]  = "ICMPv6"
            record["icmp_type"] = 129
            record["icmp_desc"] = "Echo Reply"
            record["flags"]     = "Echo Reply"

    # ── 第四層：TCP / UDP / ICMP ─────────────────────────
    def _parse_transport(self, pkt, record):
        if pkt.haslayer(TCP):
            self._parse_tcp(pkt, record)
        elif pkt.haslayer(UDP):
            self._parse_udp(pkt, record)
        elif pkt.haslayer(ICMP):
            self._parse_icmp(pkt, record)

    def _parse_tcp(self, pkt, record):
        tcp = pkt[TCP]
        record["src_port"] = tcp.sport
        record["dst_port"] = tcp.dport
        record["seq"]      = tcp.seq
        record["ack"]      = tcp.ack
        record["window"]   = tcp.window
        record["checksum"] = hex(tcp.chksum) if tcp.chksum else None
        record["flags"]    = self._parse_tcp_flags(tcp.flags)
        # DNS over TCP: dport=53 為查詢；sport=53 需有 DNS layer 才算回應
        if tcp.dport in (53, 5353):
            record["protocol"] = "DNS"
        elif tcp.sport in (53, 5353) and pkt.haslayer(DNS):
            record["protocol"] = "DNS"

    def _parse_udp(self, pkt, record):
        udp = pkt[UDP]
        record["src_port"] = udp.sport
        record["dst_port"] = udp.dport
        record["checksum"] = hex(udp.chksum) if udp.chksum else None
        # Scapy 的 UDP 預設 sport=53，若單純用 sport in (53,5353) 判斷
        # 會讓 NTP/SSDP 等未指定 sport 的封包被誤判為 DNS。
        # 修正：dport=53 才直接判斷 DNS；sport=53 需同時有 DNS layer。
        if udp.dport in (53, 5353):
            record["protocol"] = "DNS"
        elif udp.sport in (53, 5353) and pkt.haslayer(DNS):
            record["protocol"] = "DNS"
        elif udp.dport in (67, 68):
            record["protocol"] = "DHCP"
        elif udp.dport == 5355:
            record["protocol"] = "LLMNR"
        elif udp.dport == 1900:
            record["protocol"] = "SSDP"
        elif udp.dport == 123:
            record["protocol"] = "NTP"

    def _parse_icmp(self, pkt, record):
        icmp = pkt[ICMP]
        itype = icmp.type
        icode = icmp.code
        record["icmp_type"] = itype
        record["icmp_code"] = icode

        type_str = ICMP_TYPE_MAP.get(itype, f"Type {itype}")

        # Destination Unreachable: 顯示細分 Code
        if itype == 3:
            code_str = ICMP_UNREACH_CODE_MAP.get(icode, f"Code {icode}")
            record["icmp_desc"] = f"{type_str} ({code_str})"
        # Time Exceeded: Code 0=TTL, Code 1=Fragment
        elif itype == 11:
            code_str = "TTL Exceeded" if icode == 0 else "Fragment Reassembly"
            record["icmp_desc"] = f"{type_str} ({code_str})"
        else:
            record["icmp_desc"] = type_str

        record["flags"] = record["icmp_desc"]

    # ── 應用層：DNS / TLS ─────────────────────────────────
    def _parse_application(self, pkt, record):
        # DNS
        if pkt.haslayer(DNS):
            self._parse_dns(pkt, record)
        # TLS（偵測 TCP 443 / 8443 的 Raw Payload）
        if (pkt.haslayer(TCP) and pkt.haslayer("Raw")
                and record.get("dst_port") in (443, 8443, 993, 995, 465, 587)
                or record.get("src_port") in (443, 8443, 993, 995, 465, 587)):
            self._parse_tls(pkt, record)

    def _parse_dns(self, pkt, record):
        """提取 DNS 查詢域名與回應記錄"""
        dns = pkt[DNS]
        if pkt.haslayer(DNSQR):
            try:
                record["dns_query"] = pkt[DNSQR].qname.decode(
                    errors="replace").rstrip(".")
            except Exception:
                pass
        # DNS 回應
        if dns.qr == 1 and pkt.haslayer(DNSRR):
            try:
                rr = pkt[DNSRR]
                record["dns_response"] = (
                    f"{rr.rrname.decode(errors='replace').rstrip('.')} -> {rr.rdata}"
                )
            except Exception:
                pass

    def _parse_tls(self, pkt, record):
        """
        偵測 TLS Record Layer：
          - Content Type（Handshake/AppData）
          - Protocol Version（TLS 1.2 / 1.3）
          - Handshake Type（ClientHello / ServerHello）
        """
        try:
            raw = bytes(pkt["Raw"].load)
            if len(raw) < 5:
                return

            content_type = raw[0]
            version_bytes = raw[1:3]
            # length = int.from_bytes(raw[3:5], 'big')

            ct_str = TLS_CONTENT_TYPE_MAP.get(content_type)
            ver_str = TLS_VERSION_MAP.get(version_bytes)

            if ct_str is None or ver_str is None:
                return

            record["protocol"]    = "TLS"
            record["tls_type"]    = ct_str
            record["tls_version"] = ver_str

            # 進一步解析 Handshake
            if content_type == 22 and len(raw) >= 6:
                hs_type = raw[5]
                record["tls_handshake"] = TLS_HANDSHAKE_TYPE_MAP.get(
                    hs_type, f"Type {hs_type}"
                )
        except Exception:
            pass

    # ── Payload ───────────────────────────────────────────
    def _parse_payload(self, pkt, record):
        """提取最底層 Raw Payload（十六進位 + 可讀 ASCII）"""
        if not pkt.haslayer("Raw"):
            return
        raw = bytes(pkt["Raw"].load)
        if not raw:
            return
        record["payload_len"]  = len(raw)
        display = raw[:MAX_PAYLOAD_DISPLAY]
        record["payload_hex"]  = display.hex()
        record["payload_text"] = "".join(
            chr(b) if 32 <= b < 127 else "." for b in display
        )

    # ── 靜態工具 ──────────────────────────────────────────
    @staticmethod
    def _parse_tcp_flags(flags) -> str:
        """
        TCP Flags 整數轉可讀字串
        0x002 -> "SYN"   0x012 -> "SYN+ACK"   0x018 -> "PSH+ACK"
        """
        flag_map = {
            0x001: "FIN", 0x002: "SYN", 0x004: "RST", 0x008: "PSH",
            0x010: "ACK", 0x020: "URG", 0x040: "ECE", 0x080: "CWR",
        }
        active = [name for bit, name in flag_map.items() if int(flags) & bit]
        return "+".join(active) if active else "NONE"

    @staticmethod
    def _get_timestamp(pkt) -> str:
        """取得封包時間戳記字串"""
        try:
            return datetime.fromtimestamp(float(pkt.time)).strftime(
                "%Y-%m-%d %H:%M:%S.%f")[:-3]
        except Exception:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S.000")
