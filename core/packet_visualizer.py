# ============================================================
# packet_visualizer.py - 封包特徵影像化模組
#
# 功能：
#   將原始網路封包位元組序列轉換為 2D 灰階影像，
#   作為 CNN Autoencoder 的輸入資料格式。
#
# 實作策略（依計畫書規範）：
#   1. 欄位遮罩（Field Masking）
#      - 遮蔽 IP 位址、Port 等識別欄位，消除 IP/Port 偏差
#      - 保留封包語意特徵，讓模型學習行為而非身份
#
#   2. 截斷與填充（Truncation & Padding）
#      - 封包長度 > MAX_BYTES：截斷至 MAX_BYTES
#      - 封包長度 < MAX_BYTES：以 0x00 補齊至 MAX_BYTES
#      - 確保所有影像尺寸一致，符合 CNN 輸入要求
#
#   3. 影像尺寸最佳化（Image Size Optimization）
#      - 支援多種尺寸：28×28(784B) / 32×32(1024B) / 40×40(1600B)
#      - 預設 32×32：在資訊量與計算成本間取得最佳平衡
#
# 參考文獻：
#   - Wang et al. (2017) ISCX VPN-nonVPN 封包影像化方法
#   - Chen et al. (2021) CIC-IDS2017 封包影像分類
# ============================================================

import os
import struct
import hashlib
from typing import Optional, Tuple, List, Union

import warnings
warnings.filterwarnings("ignore", message="Glyph.*missing from font", category=UserWarning)

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as _fm

# 自動偵測系統可用的中文字型，依優先順序選用
# Windows：Microsoft YaHei（微軟正黑體）最常見
# Linux：WenQuanYi Zen Hei（文泉驛正黑）
_CJK_CANDIDATES = [
    "Microsoft YaHei",   # Windows
    "SimHei",             # Windows（黑體）
    "PingFang TC",        # macOS 繁體
    "Heiti TC",           # macOS
    "WenQuanYi Zen Hei",  # Linux
    "Noto Sans CJK TC",   # Noto 系列（跨平台）
]

def _find_cjk_font():
    """回傳第一個系統內找得到的中文字型名稱，找不到則回傳 None。"""
    available = {f.name for f in _fm.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            return name
    return None

_cjk_font = _find_cjk_font()
if _cjk_font:
    matplotlib.rcParams["font.family"] = [_cjk_font, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False  # 負號正常顯示


# ── 影像尺寸預設值 ────────────────────────────────────────
IMAGE_SIZES = {
    "small":  (28, 28, 784),
    "medium": (32, 32, 1024),   # 預設
    "large":  (40, 40, 1600),
}

# ── IPv4 Header 欄位偏移（相對於 IP 層起始） ──────────────
# Ethernet header = 14 bytes
# IPv4: src=12~15, dst=16~19  (相對 IP 層)
ETH_HEADER_LEN  = 14
IPV4_SRC_OFFSET = ETH_HEADER_LEN + 12   # bytes 26~29（絕對偏移）
IPV4_DST_OFFSET = ETH_HEADER_LEN + 16   # bytes 30~33
IPV4_IHL_OFFSET = ETH_HEADER_LEN + 0    # IP header length 欄位
TCP_SPORT_OFFSET_BASE = ETH_HEADER_LEN + 20   # IP 層後，TCP/UDP sport
TCP_DPORT_OFFSET_BASE = ETH_HEADER_LEN + 22


class PacketVisualizer:
    """
    封包影像化類別

    核心方法：
        bytes_to_image(raw_bytes)  -> np.ndarray (H, W) 灰階矩陣
        packet_to_image(pkt)       -> np.ndarray (需要 Scapy)
        save_image(arr, path)      -> 儲存 PNG
        visualize_comparison(...)  -> 生成對比圖（原始 vs 遮罩後）
        create_heatmap_overlay(...)-> Grad-CAM 熱力圖疊加（為後續 XAI 準備）
    """

    def __init__(self,
                 image_size: str = "medium",
                 apply_mask: bool = True,
                 normalize: bool = True,
                 skip_ethernet: bool = True):
        """
        Args:
            image_size   : "small"(28×28) / "medium"(32×32) / "large"(40×40)
            apply_mask   : 是否遮蔽 IP/Port 欄位
            normalize    : 是否將像素值正規化至 [0.0, 1.0]
            skip_ethernet: 是否跳過 Ethernet Header（14 bytes）直接從 IP 層開始
        """
        if image_size not in IMAGE_SIZES:
            raise ValueError(f"image_size 必須為 {list(IMAGE_SIZES.keys())} 之一")

        self.H, self.W, self.MAX_BYTES = IMAGE_SIZES[image_size]
        self.image_size    = image_size
        self.apply_mask    = apply_mask
        self.normalize     = normalize
        self.skip_ethernet = skip_ethernet

    # ──────────────────────────────────────────────────────
    # 核心轉換：原始 bytes → 2D 灰階矩陣
    # ──────────────────────────────────────────────────────
    def bytes_to_image(self, raw_bytes: bytes,
                       packet_type: str = "unknown") -> np.ndarray:
        """
        將封包位元組序列轉換為 2D 灰階影像矩陣

        處理流程：
            raw_bytes
              → (選) 跳過 Ethernet Header
              → 欄位遮罩（IP src/dst / Port）
              → 截斷或補零至 MAX_BYTES
              → reshape 為 (H, W) 矩陣
              → (選) 正規化至 [0.0, 1.0]

        Args:
            raw_bytes   : 原始封包位元組（bytes 物件）
            packet_type : 封包類型標籤（用於記錄，不影響轉換）

        Returns:
            np.ndarray shape=(H, W), dtype=float32 或 uint8
        """
        data = bytearray(raw_bytes)

        # ① 跳過 Ethernet Header
        if self.skip_ethernet and len(data) > ETH_HEADER_LEN:
            data = data[ETH_HEADER_LEN:]

        # ② 欄位遮罩
        if self.apply_mask:
            data = self._apply_field_mask(data)

        # ③ 截斷
        data = data[:self.MAX_BYTES]

        # ④ 補零（Padding）
        if len(data) < self.MAX_BYTES:
            data = data + bytearray(self.MAX_BYTES - len(data))

        # ⑤ Reshape 為 2D
        arr = np.frombuffer(bytes(data), dtype=np.uint8).reshape(self.H, self.W)

        # ⑥ 正規化
        if self.normalize:
            arr = arr.astype(np.float32) / 255.0

        return arr

    # ──────────────────────────────────────────────────────
    # 欄位遮罩策略
    # ──────────────────────────────────────────────────────
    def _apply_field_mask(self, data: bytearray) -> bytearray:
        """
        遮蔽封包中的識別型欄位，避免模型學到 IP/Port 偏差

        遮蔽欄位（相對於 IP 層起始，已跳過 Ethernet）：
            IPv4 Source IP       : bytes  12~15 → 0x00
            IPv4 Destination IP  : bytes  16~19 → 0x00
            Source Port          : bytes  20~21 → 0x00 (TCP/UDP)
            Destination Port     : bytes  22~23 → 0x00 (TCP/UDP)
            TCP Sequence Number  : bytes  24~27 → 0x00
            TCP Ack Number       : bytes  28~31 → 0x00

        保留欄位（對行為特徵重要）：
            Protocol, TTL, IP Flags, Payload 內容
        """
        if len(data) < 24:
            return data

        masked = bytearray(data)

        # IPv4 src/dst IP（bytes 12~19，相對 IP 層）
        if len(masked) > 19:
            masked[12:20] = b"\x00" * 8

        # TCP/UDP src/dst Port（bytes 20~23）
        if len(masked) > 23:
            masked[20:24] = b"\x00" * 4

        # TCP seq/ack（bytes 24~31）- 高度隨機，消除雜訊
        if len(masked) > 31:
            masked[24:32] = b"\x00" * 8

        return masked

    # ──────────────────────────────────────────────────────
    # 從 Scapy Packet 物件轉換（整合用）
    # ──────────────────────────────────────────────────────
    def packet_to_image(self, pkt) -> np.ndarray:
        """
        從 Scapy Packet 物件直接轉換為影像矩陣

        Args:
            pkt: Scapy Packet 物件

        Returns:
            np.ndarray shape=(H, W)
        """
        raw = bytes(pkt)
        return self.bytes_to_image(raw)

    # ──────────────────────────────────────────────────────
    # 儲存與讀取
    # ──────────────────────────────────────────────────────
    def save_image(self, arr: np.ndarray, path: str,
                   colormap: str = "gray") -> str:
        """
        儲存影像矩陣為 PNG 檔案

        Args:
            arr     : shape=(H, W) 的 numpy 矩陣
            path    : 輸出路徑（含副檔名 .png）
            colormap: 色彩映射（預設 gray，可改 viridis/hot 等）

        Returns:
            str: 實際儲存路徑
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # 轉換為 uint8
        if arr.dtype == np.float32 or arr.max() <= 1.0:
            pixel = (arr * 255).astype(np.uint8)
        else:
            pixel = arr.astype(np.uint8)

        if colormap == "gray":
            img = Image.fromarray(pixel, mode="L")
        else:
            cmap = plt.get_cmap(colormap)
            colored = (cmap(pixel / 255.0) * 255).astype(np.uint8)[:, :, :3]
            img = Image.fromarray(colored, mode="RGB")

        img.save(path)
        return path

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """讀取 PNG 並回傳正規化 float32 矩陣"""
        img = Image.open(path).convert("L")
        arr = np.array(img).astype(np.float32) / 255.0
        return arr

    # ──────────────────────────────────────────────────────
    # 視覺化：原始 vs 遮罩後 對比圖
    # ──────────────────────────────────────────────────────
    def visualize_comparison(self,
                              raw_bytes: bytes,
                              label: str = "",
                              save_path: Optional[str] = None,
                              show: bool = False) -> plt.Figure:
        """
        生成「原始封包影像 vs 欄位遮罩後影像」對比圖

        Args:
            raw_bytes : 原始封包位元組
            label     : 圖標題（例如封包類型）
            save_path : 儲存路徑（None 則不儲存）
            show      : 是否顯示（headless 環境設 False）

        Returns:
            matplotlib.figure.Figure
        """
        # 原始影像（不遮罩）
        orig_vis = PacketVisualizer(self.image_size, apply_mask=False, normalize=True)
        arr_orig = orig_vis.bytes_to_image(raw_bytes)

        # 遮罩後影像
        arr_mask = self.bytes_to_image(raw_bytes)

        # 差異圖（高亮遮罩位置）
        diff = np.abs(arr_orig.astype(float) - arr_mask.astype(float))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        fig.patch.set_facecolor("#0d1117")

        titles   = ["原始封包影像", "欄位遮罩後影像", "差異（遮罩位置）"]
        arrays   = [arr_orig, arr_mask, diff]
        cmaps    = ["gray", "gray", "hot"]

        for ax, title, arr, cmap in zip(axes, titles, arrays, cmaps):
            im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1 if cmap != "hot" else None,
                           aspect="equal", interpolation="nearest")
            ax.set_title(title, color="white", fontsize=12, pad=8)
            ax.set_xlabel(f"{self.W} pixels", color="#888", fontsize=9)
            ax.set_ylabel(f"{self.H} pixels", color="#888", fontsize=9)
            ax.tick_params(colors="#555")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white", labelcolor="white")

        # 封包統計資訊
        n_nonzero_orig = np.count_nonzero(arr_orig)
        n_nonzero_mask = np.count_nonzero(arr_mask)
        pkt_len = len(raw_bytes)
        info = (f"封包長度: {pkt_len} bytes  |  "
                f"有效像素(原始): {n_nonzero_orig}  |  "
                f"有效像素(遮罩後): {n_nonzero_mask}  |  "
                f"尺寸: {self.H}×{self.W} = {self.MAX_BYTES} bytes")

        fig.suptitle(
            f"封包影像化視覺化  {'— ' + label if label else ''}\n{info}",
            color="white", fontsize=11, y=1.01
        )
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=120, bbox_inches="tight",
                        facecolor="#0d1117")
            print(f"  [影像化] 對比圖已儲存: {save_path}")

        if show:
            plt.show()

        return fig

    # ──────────────────────────────────────────────────────
    # 視覺化：Hex 位元組熱力圖（封包內容詳細視圖）
    # ──────────────────────────────────────────────────────
    def visualize_byte_heatmap(self,
                                raw_bytes: bytes,
                                label: str = "",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        將封包位元組以熱力圖形式呈現，並標示各層 Header 邊界

        每行 16 bytes，X 軸為列偏移（0-15），Y 軸為行號（位元組 / 16）
        顏色深淺代表位元組值大小（0x00=深, 0xFF=亮）
        """
        data = np.frombuffer(raw_bytes[:256], dtype=np.uint8)
        rows = (len(data) + 15) // 16
        padded = np.zeros(rows * 16, dtype=np.uint8)
        padded[:len(data)] = data
        grid = padded.reshape(rows, 16)

        fig, (ax_main, ax_hex) = plt.subplots(
            1, 2, figsize=(14, max(rows * 0.45 + 1.5, 6)),
            gridspec_kw={"width_ratios": [2, 1]}
        )
        fig.patch.set_facecolor("#0d1117")

        # 熱力圖
        custom_cmap = LinearSegmentedColormap.from_list(
            "pkt", ["#0d1117", "#1a3a6b", "#2979ff", "#00e5ff", "#ffffff"]
        )
        im = ax_main.imshow(grid, cmap=custom_cmap, aspect="auto",
                            vmin=0, vmax=255, interpolation="nearest")
        ax_main.set_title(f"封包位元組熱力圖  {'— ' + label if label else ''}",
                          color="white", fontsize=12, pad=10)
        ax_main.set_xlabel("位元組偏移（列）", color="#aaa", fontsize=9)
        ax_main.set_ylabel("行（×16 bytes）", color="#aaa", fontsize=9)
        ax_main.set_xticks(range(16))
        ax_main.set_xticklabels([f"+{i:X}" for i in range(16)], color="#aaa", fontsize=8)
        ax_main.tick_params(colors="#555")

        # 標示各層邊界（Ethernet=14, IP=20, TCP/UDP=20/8）
        boundaries = {
            14: ("Ethernet", "#ff6b6b"),
            34: ("IP Header", "#ffd93d"),
            54: ("TCP Header", "#6bcb77"),
        }
        for byte_offset, (name, color) in boundaries.items():
            row = byte_offset // 16
            col = byte_offset % 16
            if row < rows:
                ax_main.axhline(y=row - 0.5, color=color,
                                linewidth=1.5, linestyle="--", alpha=0.7)
                ax_main.text(15.5, row - 0.5, name, color=color,
                             fontsize=7, va="center", ha="right")

        plt.colorbar(im, ax=ax_main, fraction=0.025, pad=0.02,
                     label="位元組值 (0x00~0xFF)").ax.yaxis.set_tick_params(
            color="white", labelcolor="white")

        # Hex Dump 面板
        ax_hex.set_facecolor("#0d1117")
        ax_hex.axis("off")
        hex_lines = []
        for r in range(min(rows, 16)):
            row_bytes = grid[r]
            hex_str  = " ".join(f"{b:02X}" for b in row_bytes[:8])
            hex_str2 = " ".join(f"{b:02X}" for b in row_bytes[8:])
            ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in row_bytes)
            hex_lines.append(f"{r*16:04X}  {hex_str}  {hex_str2}  {ascii_str}")

        ax_hex.text(0.02, 0.98, "\n".join(hex_lines),
                    transform=ax_hex.transAxes,
                    fontfamily="monospace", fontsize=7.5,
                    color="#00e5ff", va="top", ha="left",
                    linespacing=1.6)
        ax_hex.set_title("Hex Dump", color="white", fontsize=11, pad=10)

        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=120, bbox_inches="tight",
                        facecolor="#0d1117")
            print(f"  [影像化] 熱力圖已儲存: {save_path}")

        plt.close(fig)
        return fig

    # ──────────────────────────────────────────────────────
    # Grad-CAM 熱力圖疊加（為後續 XAI 整合準備介面）
    # ──────────────────────────────────────────────────────
    def create_heatmap_overlay(self,
                                packet_image: np.ndarray,
                                cam_map: np.ndarray,
                                alpha: float = 0.6,
                                save_path: Optional[str] = None) -> np.ndarray:
        """
        將 Grad-CAM 或重建誤差圖疊加到封包影像上

        此方法作為 CNN Autoencoder + Grad-CAM 整合的介面，
        當模型訓練完成後，可直接傳入 CAM 圖進行視覺化。

        Args:
            packet_image : 封包影像矩陣 shape=(H, W)，值域 [0,1]
            cam_map      : Grad-CAM / 重建誤差圖 shape=(H, W)，值域 [0,1]
            alpha        : 熱力圖透明度（0=完全透明, 1=完全不透明）
            save_path    : 儲存路徑

        Returns:
            np.ndarray: RGB 疊加影像 shape=(H, W, 3)，uint8
        """
        H, W = packet_image.shape

        # 將 cam_map resize 到封包影像尺寸
        if cam_map.shape != (H, W):
            cam_pil = Image.fromarray((cam_map * 255).astype(np.uint8), mode="L")
            cam_pil = cam_pil.resize((W, H), Image.BILINEAR)
            cam_map = np.array(cam_pil).astype(np.float32) / 255.0

        # 熱力圖顏色映射
        cmap = plt.get_cmap("jet")
        heat_rgb = (cmap(cam_map)[:, :, :3] * 255).astype(np.uint8)

        # 原始影像轉 RGB
        orig_rgb = np.stack([
            (packet_image * 255).astype(np.uint8)] * 3, axis=-1)

        # 疊加
        overlay = (alpha * heat_rgb + (1 - alpha) * orig_rgb).astype(np.uint8)

        if save_path:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.patch.set_facecolor("#0d1117")
            panels = [
                (orig_rgb, "原始封包影像", "gray"),
                ((cam_map * 255).astype(np.uint8), "Grad-CAM / 重建誤差圖", "jet"),
                (overlay, "疊加視覺化", None),
            ]
            for ax, (img, title, cmap_str) in zip(axes, panels):
                ax.imshow(img, cmap=cmap_str, interpolation="nearest")
                ax.set_title(title, color="white", fontsize=11)
                ax.axis("off")
            fig.suptitle("封包異常特徵 Grad-CAM 熱力圖",
                         color="white", fontsize=13)
            fig.tight_layout()
            fig.savefig(save_path, dpi=120, bbox_inches="tight",
                        facecolor="#0d1117")
            print(f"  [影像化] Grad-CAM 疊加圖已儲存: {save_path}")
            plt.close(fig)

        return overlay

    # ──────────────────────────────────────────────────────
    # 批次處理
    # ──────────────────────────────────────────────────────
    def batch_convert(self,
                      packets_bytes: List[bytes],
                      labels: Optional[List[str]] = None) -> np.ndarray:
        """
        批次轉換多個封包為影像矩陣（用於 CNN 訓練資料準備）

        Args:
            packets_bytes : 原始封包 bytes 列表
            labels        : 封包標籤列表（用於顯示進度）

        Returns:
            np.ndarray shape=(N, H, W)，float32
        """
        n = len(packets_bytes)
        result = np.zeros((n, self.H, self.W), dtype=np.float32)

        for i, raw in enumerate(packets_bytes):
            result[i] = self.bytes_to_image(raw)
            if (i + 1) % 500 == 0 or (i + 1) == n:
                label_str = labels[i] if labels else ""
                print(f"  [影像化] 進度: {i+1}/{n}  {label_str}")

        return result

    # ──────────────────────────────────────────────────────
    # 統計資訊
    # ──────────────────────────────────────────────────────
    def get_stats(self, arr: np.ndarray) -> dict:
        """取得影像統計資訊（用於資料集品質檢查）"""
        return {
            "shape":    arr.shape,
            "min":      float(arr.min()),
            "max":      float(arr.max()),
            "mean":     float(arr.mean()),
            "std":      float(arr.std()),
            "nonzero_ratio": float(np.count_nonzero(arr) / arr.size),
            "entropy":  float(self._entropy(arr)),
        }

    @staticmethod
    def _entropy(arr: np.ndarray) -> float:
        """計算影像的資訊熵（衡量封包複雜度）"""
        pixel_int = (arr * 255).astype(np.uint8).flatten()
        hist, _ = np.histogram(pixel_int, bins=256, range=(0, 255))
        hist = hist[hist > 0].astype(float)
        prob = hist / hist.sum()
        return float(-np.sum(prob * np.log2(prob)))
