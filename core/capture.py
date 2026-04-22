# ============================================================
# capture.py - 即時監聽網路介面（NIC）v2.0.0 加強版
#
# 加強功能：
#   - 整合 AnomalyDetector（SYN/ICMP/UDP Flood、Port Scan、ARP Spoof、DNS Amp）
#   - 即時流量統計儀表板（每 N 秒自動列印）
#   - 封包速率（pps）監控
#   - 封包大小分布統計
#   - 每次停止後顯示完整統計摘要
# ============================================================

import time
import threading
from collections import defaultdict, Counter
from datetime import datetime

from scapy.all import sniff, get_if_list, conf, wrpcap
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import Ether, ARP

from colorama import Fore, Style, init
from tabulate import tabulate

from config import OUTPUT_PCAP, PORT_SERVICE_MAP
from parser import PacketParser
from storage import PacketStorage
from anomaly_detector import AnomalyDetector

init(autoreset=True)


class LiveCapture:
    """
    即時網路封包擷取類別（加強版）

    功能：
      - 監聽指定網路介面（NIC）
      - 即時解析並顯示封包資訊（含協議顏色）
      - 整合 AnomalyDetector：SYN Flood / Port Scan / ICMP Flood /
        UDP Flood / ARP Spoofing / DNS 放大
      - 每 stats_interval 秒自動列印即時流量統計
      - 自動儲存為 PCAP、CSV、JSON（可選 SQLite）
    """

    def __init__(self,
                 interface=None,
                 bpf_filter="",
                 count=0,
                 save_pcap=True,
                 stats_interval=10,
                 save_db=False,
                 custom_alert_callback=None,
                 session_dir=None):
        """
        Args:
            interface            : 網路介面名稱，None 則自動選擇預設介面
            bpf_filter           : BPF 封包過濾語法
            count                : 擷取封包數量，0 = 無限制
            save_pcap            : 是否儲存 PCAP 檔案
            stats_interval       : 多少秒列印一次即時統計（0 = 停用）
            save_db              : 是否同時儲存到 SQLite 資料庫
            custom_alert_callback: 自訂告警回調 fn(alert_dict)
        """
        self.interface       = interface or conf.iface
        self.bpf_filter      = bpf_filter
        self.count           = count
        self.save_pcap       = save_pcap
        self.stats_interval  = stats_interval
        self.save_db         = save_db

        self.packets         = []
        self.parsed_records  = []
        self.packet_count    = 0
        self.total_bytes     = 0
        self.start_time      = None

        self.session_dir = session_dir
        self.parser   = PacketParser()
        self.storage  = PacketStorage(session_dir=session_dir)
        self.detector = AnomalyDetector(on_alert=custom_alert_callback)

        # 統計計數器
        self.proto_counter  = Counter()   # 協議分布
        self.src_ip_counter = Counter()   # 來源 IP
        self.dst_port_counter = Counter() # 目標 Port
        self.size_buckets   = Counter()   # 封包大小分布

        self._stop_event     = threading.Event()
        self._stats_thread   = None
        self._lock           = threading.Lock()

    # ── 列出所有網路介面 ─────────────────────────────────
    @staticmethod
    def list_interfaces():
        """列出系統上所有可用網路介面（Windows 顯示人類可讀名稱）"""
        print(f"\n{Fore.CYAN}{'='*65}")
        print("  可用的網路介面")
        print(f"{'='*65}{Style.RESET_ALL}")

        try:
            # Windows：透過 get_windows_if_list() 取得描述名稱
            from scapy.arch.windows import get_windows_if_list
            ifaces = get_windows_if_list()
            for idx, iface in enumerate(ifaces):
                name = iface["name"]
                desc = iface.get("description", "")[:40]
                mark = (f"{Fore.GREEN}*{Style.RESET_ALL}"
                        if name == conf.iface else " ")
                short = name[-45:] if len(name) > 45 else name
                print(f"  {mark} [{idx}] {desc:<42} {Fore.LIGHTBLACK_EX}{short}{Style.RESET_ALL}")
        except Exception:
            # Linux / macOS fallback
            for idx, iface in enumerate(get_if_list()):
                mark = (f"{Fore.GREEN}*{Style.RESET_ALL}"
                        if iface == conf.iface else " ")
                print(f"  {mark} [{idx}] {iface}")

        print(f"\n{Fore.YELLOW}  * = 目前預設介面{Style.RESET_ALL}")
        print(f"  使用方式: python main.py live -i \"介面名稱\" -c 50")
        print(f"{Fore.CYAN}{'='*65}{Style.RESET_ALL}\n")


    # ── 封包回調 ─────────────────────────────────────────
    def _packet_callback(self, pkt):
        """每次擷取到封包時呼叫"""
        with self._lock:
            self.packet_count += 1
            self.total_bytes  += len(pkt)
            self.packets.append(pkt)

        record = self.parser.parse(pkt)

        with self._lock:
            self.parsed_records.append(record)
            # 更新統計計數器
            proto = record.get("protocol", "UNKNOWN")
            self.proto_counter[proto] += 1
            if record.get("src_ip"):
                self.src_ip_counter[record["src_ip"]] += 1
            if isinstance(record.get("dst_port"), int):
                self.dst_port_counter[record["dst_port"]] += 1
            # 封包大小分桶
            size = len(pkt)
            bucket = self._size_bucket(size)
            self.size_buckets[bucket] += 1

        self._display_packet(record)
        self.detector.inspect(pkt, record)

    @staticmethod
    def _size_bucket(size: int) -> str:
        """將封包大小分類到區間"""
        if size < 64:    return "<64B"
        if size < 256:   return "64-255B"
        if size < 512:   return "256-511B"
        if size < 1024:  return "512-1023B"
        if size < 1500:  return "1024-1499B"
        return ">=1500B"

    # ── 即時顯示 ─────────────────────────────────────────
    def _display_packet(self, record):
        """格式化顯示封包（帶顏色）"""
        proto    = record.get("protocol", "UNK")
        src_ip   = record.get("src_ip",   "N/A")
        dst_ip   = record.get("dst_ip",   "N/A")
        src_port = record.get("src_port", "-")
        dst_port = record.get("dst_port", "-")
        length   = record.get("length",   0)
        flags    = record.get("flags",    "")
        ts       = record.get("timestamp","")

        # TLS 顯示額外資訊
        tls_info = ""
        if proto == "TLS":
            tls_info = (f" [{record.get('tls_version','')} "
                        f"{record.get('tls_handshake') or record.get('tls_type','')}]")

        # DNS 顯示查詢域名
        dns_info = ""
        if proto == "DNS" and record.get("dns_query"):
            dns_info = f" [{record['dns_query']}]"

        color_map = {
            "TCP":   Fore.GREEN,
            "UDP":   Fore.CYAN,
            "ICMP":  Fore.YELLOW,
            "ICMPv6": Fore.LIGHTYELLOW_EX,
            "ARP":   Fore.MAGENTA,
            "DNS":   Fore.BLUE,
            "TLS":   Fore.LIGHTCYAN_EX,
            "DHCP":  Fore.LIGHTBLUE_EX,
        }
        color   = color_map.get(proto, Fore.WHITE)
        svc     = PORT_SERVICE_MAP.get(dst_port, "")
        svc_str = f"({svc})" if svc else ""

        print(
            f"{Fore.WHITE}[{self.packet_count:05d}] "
            f"{Fore.LIGHTBLACK_EX}{ts}  "
            f"{color}{proto:<8}{Style.RESET_ALL}  "
            f"{src_ip}:{src_port}  ->  "
            f"{dst_ip}:{dst_port} {svc_str}  "
            f"{Fore.LIGHTBLACK_EX}{length}B  "
            f"{Fore.YELLOW}{flags}"
            f"{Fore.LIGHTCYAN_EX}{tls_info}{dns_info}"
        )

    # ── 即時統計儀表板 ────────────────────────────────────
    def _stats_loop(self):
        """背景執行緒：每 stats_interval 秒印出統計儀表板"""
        while not self._stop_event.wait(timeout=self.stats_interval):
            self._print_stats_dashboard()

    def _print_stats_dashboard(self):
        """列印即時統計儀表板"""
        with self._lock:
            elapsed    = time.time() - (self.start_time or time.time())
            pkt_count  = self.packet_count
            tot_bytes  = self.total_bytes
            proto_snap = dict(self.proto_counter.most_common(8))
            src_snap   = dict(self.src_ip_counter.most_common(5))
            port_snap  = dict(self.dst_port_counter.most_common(5))
            size_snap  = dict(self.size_buckets)

        pps = pkt_count / max(elapsed, 1)
        bps = tot_bytes / max(elapsed, 1)

        print(f"\n{Fore.CYAN}{'─'*65}")
        print(f"  📊 即時統計  |  時間: {elapsed:.0f}s  "
              f"|  封包: {pkt_count:,}  "
              f"|  速率: {pps:.1f} pps  "
              f"|  流量: {bps/1024:.1f} KB/s")
        print(f"{'─'*65}{Style.RESET_ALL}")

        # 協議分布
        if proto_snap:
            proto_rows = [[p, c, f"{c/max(pkt_count,1)*100:.1f}%"]
                          for p, c in proto_snap.items()]
            print(tabulate(proto_rows, headers=["協議", "封包數", "佔比"],
                           tablefmt="simple"))

        # Top 5 來源 IP
        if src_snap:
            print(f"\n  Top 來源 IP:")
            for ip, cnt in src_snap.items():
                print(f"    {ip:<20} {cnt:>6} 封包")

        # 告警數量
        alert_count = len(self.detector.alert_history)
        if alert_count:
            print(f"\n  {Fore.RED}⚠  累計告警: {alert_count} 筆{Style.RESET_ALL}")

        print(f"{Fore.CYAN}{'─'*65}{Style.RESET_ALL}\n")

    # ── 開始 / 停止 ──────────────────────────────────────
    def start(self):
        """開始擷取（阻塞式，Ctrl+C 停止）"""
        print(f"\n{Fore.GREEN}{'='*65}")
        print(f"  開始即時封包擷取")
        print(f"  介面    : {self.interface}")
        print(f"  過濾器  : {self.bpf_filter or '(無，擷取全部)'}")
        print(f"  數量限制: {self.count or '無限制'}")
        print(f"  統計間隔: {self.stats_interval}s")
        print(f"  異常偵測: ✓ SYN Flood / Port Scan / ICMP Flood / "
              f"UDP Flood / ARP Spoof / DNS Amp")
        print(f"  按 Ctrl+C 停止")
        print(f"{'='*65}{Style.RESET_ALL}\n")

        self.start_time = time.time()

        # 啟動統計執行緒
        if self.stats_interval > 0:
            self._stats_thread = threading.Thread(
                target=self._stats_loop, daemon=True)
            self._stats_thread.start()

        try:
            sniff(
                iface=self.interface,
                filter=self.bpf_filter,
                prn=self._packet_callback,
                count=self.count,
                store=False,
                stop_filter=lambda x: self._stop_event.is_set()
            )
        except KeyboardInterrupt:
            pass
        finally:
            self._on_stop()

    def stop(self):
        """外部呼叫停止擷取"""
        self._stop_event.set()

    def _on_stop(self):
        """停止後的收尾處理：列印完整報告並儲存檔案"""
        self._stop_event.set()
        elapsed = time.time() - self.start_time

        print(f"\n{Fore.CYAN}{'='*65}")
        print(f"  擷取結束")
        print(f"  總封包: {self.packet_count:,}")
        print(f"  總流量: {self.total_bytes:,} Bytes ({self.total_bytes/1024:.1f} KB)")
        print(f"  時間:   {elapsed:.1f} 秒")
        print(f"  速率:   {self.packet_count / max(elapsed, 1):.1f} pps  |  "
              f"{self.total_bytes / max(elapsed, 1) / 1024:.1f} KB/s")
        print(f"{'='*65}{Style.RESET_ALL}\n")

        # 最終統計儀表板
        self._print_stats_dashboard()

        # 告警歷史
        self.detector.print_history()

        # 儲存檔案
        if self.parsed_records:
            self.storage.save_csv(self.parsed_records)
            self.storage.save_json(self.parsed_records)
            if self.save_db:
                self.storage.save_sqlite(self.parsed_records)

        if self.save_pcap and self.packets:
            wrpcap(OUTPUT_PCAP, self.packets)
            print(f"{Fore.GREEN}  PCAP 已儲存: {OUTPUT_PCAP}{Style.RESET_ALL}")
