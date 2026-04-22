# ============================================================
# pcap_analyzer.py - PCAP 離線分析模組（v2.0.0 加強版）
#
# 加強功能：
#   - TLS/SSL 記錄分析（Handshake 類型、版本分布）
#   - ARP 封包分析（偵測 ARP Spoofing）
#   - 時間軸分析（每秒 / 每分鐘封包數量趨勢）
#   - 離線攻擊特徵偵測（整合 AnomalyDetector）
#   - 封包大小分布
#   - IPv6 封包統計
# ============================================================

import os
from collections import Counter, defaultdict
from datetime import datetime

from scapy.all import rdpcap, PcapReader
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.dns import DNS, DNSQR
from tabulate import tabulate
from colorama import Fore, Style

from config import PORT_SERVICE_MAP
from parser import PacketParser
from storage import PacketStorage
from anomaly_detector import AnomalyDetector


class PcapAnalyzer:
    """
    PCAP 離線分析類別（加強版）

    功能：
      - 讀取 PCAP / PCAPNG 檔案（支援串流讀取大型檔案）
      - 統計 Top IP、Port、Protocol 分布
      - 提取 DNS 查詢記錄
      - 提取 HTTP 請求
      - TLS/SSL 分析（版本分布 / Handshake 類型）
      - ARP 分析（偵測 IP-MAC 異常）
      - 時間軸分析（每秒 / 每分鐘流量趨勢）
      - 離線異常偵測（整合 AnomalyDetector）
      - 重建 TCP 連線流（四元組分組）
      - 儲存分析結果為 CSV / JSON / SQLite
    """

    def __init__(self, pcap_path: str, save_db=False, session_dir=None):
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"找不到 PCAP 檔案: {pcap_path}")
        self.pcap_path   = pcap_path
        self.save_db     = save_db
        self.session_dir = session_dir
        self.parser      = PacketParser()
        self.storage     = PacketStorage(session_dir=session_dir)
        self.detector    = AnomalyDetector()
        self.records     = []
        self.packets     = []

    # ── 讀取 PCAP ────────────────────────────────────────
    def load(self, use_streaming=False):
        """
        讀取 PCAP 檔案

        Args:
            use_streaming: True -> 串流讀取（適合 >500MB 大型 PCAP）
        """
        print(f"\n{Fore.CYAN}載入 PCAP: {self.pcap_path}{Style.RESET_ALL}")

        if use_streaming:
            with PcapReader(self.pcap_path) as reader:
                for pkt in reader:
                    self.packets.append(pkt)
                    self.records.append(self.parser.parse(pkt))
        else:
            self.packets = rdpcap(self.pcap_path)
            self.records = [self.parser.parse(p) for p in self.packets]

        total_bytes = sum(r.get("length", 0) for r in self.records)
        print(f"{Fore.GREEN}  共載入 {len(self.packets):,} 個封包  "
              f"({total_bytes/1024:.1f} KB){Style.RESET_ALL}")
        return self

    # ── 基本統計摘要 ──────────────────────────────────────
    def summary(self):
        """統計協議分布、Top 來源 IP、Top 目標 Port、封包大小分布"""
        self._require_loaded()

        protocols   = Counter(r["protocol"] for r in self.records)
        src_ips     = Counter(r["src_ip"] for r in self.records if r["src_ip"])
        dst_ports   = Counter(
            r["dst_port"] for r in self.records
            if isinstance(r.get("dst_port"), int)
        )
        total_bytes  = sum(r.get("length", 0) for r in self.records)
        timestamps   = [r["timestamp"] for r in self.records if r.get("timestamp")]
        ip_versions  = Counter(r.get("ip_version") for r in self.records
                               if r.get("ip_version"))

        # 封包大小分布
        size_buckets = Counter()
        for r in self.records:
            size = r.get("length", 0)
            if size < 64:    size_buckets["<64B"] += 1
            elif size < 256: size_buckets["64-255B"] += 1
            elif size < 512: size_buckets["256-511B"] += 1
            elif size < 1024:size_buckets["512-1023B"] += 1
            elif size < 1500:size_buckets["1024-1499B"] += 1
            else:             size_buckets[">=1500B"] += 1

        print(f"\n{Fore.CYAN}{'='*65}\n  PCAP 分析摘要\n{'='*65}{Style.RESET_ALL}")
        print(f"  封包總數  : {len(self.records):,}")
        print(f"  總流量    : {total_bytes:,} Bytes ({total_bytes/1024:.1f} KB)")
        if timestamps:
            print(f"  開始時間  : {timestamps[0]}")
            print(f"  結束時間  : {timestamps[-1]}")
        print(f"  IP 版本   : " +
              " / ".join(f"IPv{v}: {c}" for v, c in ip_versions.items()))

        print(f"\n{Fore.YELLOW}  [協議分布 Top 10]{Style.RESET_ALL}")
        print(tabulate(
            [[p, c, f"{c/len(self.records)*100:.1f}%"]
             for p, c in protocols.most_common(10)],
            headers=["協議", "封包數", "佔比"],
            tablefmt="rounded_outline"
        ))

        print(f"\n{Fore.YELLOW}  [Top 10 來源 IP]{Style.RESET_ALL}")
        print(tabulate(
            list(src_ips.most_common(10)),
            headers=["來源 IP", "封包數"],
            tablefmt="rounded_outline"
        ))

        print(f"\n{Fore.YELLOW}  [Top 10 目標 Port]{Style.RESET_ALL}")
        print(tabulate(
            [[p, PORT_SERVICE_MAP.get(p, "Unknown"), c]
             for p, c in dst_ports.most_common(10)],
            headers=["Port", "服務", "封包數"],
            tablefmt="rounded_outline"
        ))

        print(f"\n{Fore.YELLOW}  [封包大小分布]{Style.RESET_ALL}")
        for bucket in ["<64B","64-255B","256-511B","512-1023B","1024-1499B",">=1500B"]:
            cnt = size_buckets.get(bucket, 0)
            if cnt:
                bar = "█" * int(cnt / max(size_buckets.values()) * 30)
                print(f"  {bucket:<12}  {bar:<30}  {cnt}")

        return self

    # ── 提取 DNS 查詢 ─────────────────────────────────────
    def extract_dns(self):
        """提取所有 DNS 查詢與回應記錄"""
        dns_records = []
        for pkt in self.packets:
            if not pkt.haslayer(DNS):
                continue
            dns = pkt[DNS]
            src_ip = pkt[IP].src if pkt.haslayer(IP) else "N/A"
            ts     = self.parser._get_timestamp(pkt)

            if pkt.haslayer(DNSQR) and dns.qr == 0:
                dns_records.append({
                    "type": "Query",
                    "timestamp": ts,
                    "src_ip":    src_ip,
                    "name":      pkt[DNSQR].qname.decode(errors="replace").rstrip("."),
                    "qtype":     pkt[DNSQR].qtype,
                })
            elif dns.qr == 1:  # 回應
                from scapy.layers.dns import DNSRR
                if pkt.haslayer(DNSRR):
                    try:
                        rr = pkt[DNSRR]
                        name = rr.rrname.decode(errors="replace").rstrip(".")
                        dns_records.append({
                            "type": "Response",
                            "timestamp": ts,
                            "src_ip":    src_ip,
                            "name":      name,
                            "rdata":     str(rr.rdata),
                        })
                    except Exception:
                        pass

        if dns_records:
            queries   = [r for r in dns_records if r["type"] == "Query"]
            responses = [r for r in dns_records if r["type"] == "Response"]
            print(f"\n{Fore.YELLOW}  [DNS 記錄] 查詢 {len(queries)} 筆 / "
                  f"回應 {len(responses)} 筆{Style.RESET_ALL}")
            if queries:
                print(tabulate(
                    [[q["timestamp"], q["src_ip"], q["name"]]
                     for q in queries[:20]],
                    headers=["時間", "來源 IP", "查詢域名"],
                    tablefmt="rounded_outline"
                ))
        return dns_records

    # ── 提取 HTTP 請求 ────────────────────────────────────
    def extract_http(self):
        """提取 HTTP 請求（TCP Port 80 的 Raw Payload）"""
        http_requests = []
        for pkt in self.packets:
            if not (pkt.haslayer(TCP) and pkt[TCP].dport == 80
                    and pkt.haslayer("Raw")):
                continue
            payload = bytes(pkt["Raw"].load)
            if not payload.startswith((b"GET", b"POST", b"HEAD", b"PUT",
                                       b"DELETE", b"OPTIONS", b"PATCH")):
                continue
            lines = payload.decode(errors="replace").split("\r\n")
            host  = next(
                (l.split(": ", 1)[1] for l in lines if l.startswith("Host:")),
                "Unknown"
            )
            src_ip = pkt[IP].src if pkt.haslayer(IP) else "N/A"
            http_requests.append({
                "src_ip": src_ip,
                "host":   host,
                "method": lines[0][:80],
                "timestamp": self.parser._get_timestamp(pkt),
            })

        if http_requests:
            print(f"\n{Fore.YELLOW}  [HTTP 請求] 共 {len(http_requests)} 筆{Style.RESET_ALL}")
            print(tabulate(
                [[r["timestamp"], r["src_ip"], r["host"], r["method"]]
                 for r in http_requests[:20]],
                headers=["時間", "來源 IP", "Host", "請求"],
                tablefmt="rounded_outline"
            ))
        return http_requests

    # ── TLS/SSL 分析（新增）───────────────────────────────
    def analyze_tls(self):
        """
        分析 TLS/SSL 封包：
          - Handshake 類型分布（ClientHello / ServerHello）
          - TLS 版本分布（TLS 1.2 / TLS 1.3）
          - HTTPS 連線的來源 IP
        """
        tls_records = [r for r in self.records if r.get("protocol") == "TLS"]

        if not tls_records:
            print(f"\n{Fore.YELLOW}  [TLS] 未找到 TLS 封包{Style.RESET_ALL}")
            return []

        version_counter = Counter(r.get("tls_version") for r in tls_records
                                  if r.get("tls_version"))
        hs_counter      = Counter(r.get("tls_handshake") for r in tls_records
                                  if r.get("tls_handshake"))
        type_counter    = Counter(r.get("tls_type") for r in tls_records
                                  if r.get("tls_type"))
        src_ips         = Counter(r.get("src_ip") for r in tls_records
                                  if r.get("src_ip"))

        print(f"\n{Fore.YELLOW}  [TLS/SSL 分析] 共 {len(tls_records)} 個 TLS 封包{Style.RESET_ALL}")

        if version_counter:
            print(f"\n  TLS 版本分布:")
            print(tabulate(list(version_counter.most_common()),
                           headers=["版本", "封包數"],
                           tablefmt="rounded_outline"))

        if hs_counter:
            print(f"\n  Handshake 類型:")
            print(tabulate(list(hs_counter.most_common()),
                           headers=["類型", "封包數"],
                           tablefmt="rounded_outline"))

        if type_counter:
            print(f"\n  Content Type:")
            print(tabulate(list(type_counter.most_common()),
                           headers=["Type", "封包數"],
                           tablefmt="rounded_outline"))

        if src_ips:
            print(f"\n  Top 10 TLS 來源 IP:")
            print(tabulate(list(src_ips.most_common(10)),
                           headers=["來源 IP", "封包數"],
                           tablefmt="rounded_outline"))

        return tls_records

    # ── ARP 分析（新增）──────────────────────────────────
    def analyze_arp(self):
        """
        分析 ARP 封包：
          - ARP 請求 / 回應 比例
          - IP -> MAC 對應表
          - 偵測 ARP Spoofing（一個 IP 對應多個 MAC）
        """
        arp_records = [r for r in self.records if r.get("protocol") == "ARP"]

        if not arp_records:
            print(f"\n{Fore.YELLOW}  [ARP] 未找到 ARP 封包{Style.RESET_ALL}")
            return {}

        op_counter = Counter(r.get("arp_op") for r in arp_records)
        ip_mac_map = defaultdict(set)  # {ip: {mac1, mac2, ...}}

        for r in arp_records:
            ip  = r.get("arp_src_ip", "")
            mac = r.get("arp_src_mac", "")
            if ip and mac:
                ip_mac_map[ip].add(mac)

        spoof_candidates = {ip: macs for ip, macs in ip_mac_map.items()
                            if len(macs) >= 2}

        print(f"\n{Fore.YELLOW}  [ARP 分析] 共 {len(arp_records)} 個 ARP 封包{Style.RESET_ALL}")

        print(tabulate(
            [[op, cnt] for op, cnt in op_counter.most_common()],
            headers=["操作", "封包數"],
            tablefmt="rounded_outline"
        ))

        print(f"\n  IP -> MAC 對應表 (Top 15):")
        ip_mac_rows = []
        for ip, macs in list(ip_mac_map.items())[:15]:
            warning = f"  {Fore.RED}⚠ 疑似 Spoof{Style.RESET_ALL}" if len(macs) >= 2 else ""
            ip_mac_rows.append([ip, ", ".join(macs), warning])
        print(tabulate(ip_mac_rows,
                       headers=["IP", "MAC", "狀態"],
                       tablefmt="rounded_outline"))

        if spoof_candidates:
            print(f"\n{Fore.RED}  ⚠  偵測到 ARP Spoofing 疑似目標:{Style.RESET_ALL}")
            for ip, macs in spoof_candidates.items():
                print(f"    IP: {ip}  ->  MAC: {', '.join(macs)}")

        return ip_mac_map

    # ── 時間軸分析（新增）────────────────────────────────
    def analyze_timeline(self, granularity="second"):
        """
        分析封包數量隨時間的變化趨勢

        Args:
            granularity: "second" 或 "minute"
        """
        time_counter = Counter()
        for r in self.records:
            ts = r.get("timestamp", "")
            if not ts:
                continue
            try:
                if granularity == "second":
                    key = ts[:19]   # YYYY-MM-DD HH:MM:SS
                else:
                    key = ts[:16]   # YYYY-MM-DD HH:MM
                time_counter[key] += 1
            except Exception:
                pass

        if not time_counter:
            return {}

        sorted_times = sorted(time_counter.items())
        max_count    = max(v for _, v in sorted_times)

        print(f"\n{Fore.YELLOW}  [時間軸分析] ({granularity}){Style.RESET_ALL}")
        # 顯示最多 30 個時間點
        display = sorted_times[-30:] if len(sorted_times) > 30 else sorted_times
        for ts, cnt in display:
            bar = "█" * int(cnt / max_count * 40)
            print(f"  {ts}  {bar:<40}  {cnt:>5}")

        return dict(sorted_times)

    # ── 離線異常偵測（新增）──────────────────────────────
    def detect_attacks(self):
        """
        對已載入的 PCAP 進行完整攻擊特徵偵測
        整合 AnomalyDetector 逐一掃描所有封包
        """
        print(f"\n{Fore.CYAN}  開始離線攻擊偵測...{Style.RESET_ALL}")
        self.detector.reset()

        for pkt, record in zip(self.packets, self.records):
            self.detector.inspect(pkt, record)

        print(f"\n  偵測完成")
        self.detector.print_history()

        summary = self.detector.get_summary()
        if summary["total_alerts"] == 0:
            print(f"{Fore.GREEN}  ✓ 未發現攻擊特徵{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}  ⚠  發現 {summary['total_alerts']} 個告警{Style.RESET_ALL}")
        return summary

    # ── 重建 TCP 連線流 ───────────────────────────────────
    def rebuild_tcp_streams(self):
        """以四元組重建 TCP 連線流，統計每條流量"""
        streams = defaultdict(list)
        for pkt in self.packets:
            if pkt.haslayer(TCP) and pkt.haslayer(IP):
                key = (pkt[IP].src, pkt[TCP].sport,
                       pkt[IP].dst, pkt[TCP].dport)
                streams[key].append(pkt)

        print(f"\n{Fore.YELLOW}  [TCP 連線流] 共 {len(streams)} 條{Style.RESET_ALL}")
        table = []
        for (s_ip, s_port, d_ip, d_port), pkts in list(streams.items())[:15]:
            total_bytes = sum(len(p) for p in pkts)
            svc = PORT_SERVICE_MAP.get(d_port, "")
            table.append([
                f"{s_ip}:{s_port}",
                f"{d_ip}:{d_port}",
                svc, len(pkts),
                f"{total_bytes:,} B"
            ])
        print(tabulate(table,
                       headers=["來源", "目標", "服務", "封包數", "流量"],
                       tablefmt="rounded_outline"))
        return streams

    # ── 儲存結果 ──────────────────────────────────────────
    def save_results(self):
        """儲存所有解析記錄為 CSV、JSON（可選 SQLite）"""
        if self.records:
            self.storage.save_csv(self.records, filename="pcap_analysis.csv")
            self.storage.save_json(self.records, filename="pcap_analysis.json")
            if self.save_db:
                self.storage.save_sqlite(self.records)
        return self

    # ── 一鍵完整分析 ──────────────────────────────────────
    def full_analysis(self):
        """
        執行完整分析流程：
        載入 -> 摘要 -> DNS -> HTTP -> TLS -> ARP ->
        時間軸 -> 攻擊偵測 -> TCP 流 -> 儲存
        """
        self.load()
        self.summary()
        self.extract_dns()
        self.extract_http()
        self.analyze_tls()
        self.analyze_arp()
        self.analyze_timeline()
        self.detect_attacks()
        self.rebuild_tcp_streams()
        self.save_results()
        return self

    # ── 輔助 ──────────────────────────────────────────────
    def _require_loaded(self):
        if not self.records:
            raise RuntimeError("請先執行 load() 載入 PCAP 檔案")
