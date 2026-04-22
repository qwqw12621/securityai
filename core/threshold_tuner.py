# ============================================================
# threshold_tuner.py - 重建誤差閾值調校模組
#
# 解決原有 trainer.py 的問題：
#   原版只支援單一固定百分位數（預設 95th），
#   無法針對不同攻擊類型調整，也看不出各閾值的 Precision/Recall 變化。
#
# 新增功能：
#   1. 多百分位數掃描：同時計算 80~99th 百分位對應的 P/R/F1
#   2. 攻擊類型分組評估：DoS / DDoS / PortScan 各自最佳閾值
#   3. F1 Score 最大化自動選閾值
#   4. 消融實驗支援：比較欄位遮罩、影像尺寸等設計決策的影響
#   5. 閾值對比報告圖（同一張圖呈現 P/R/F1 vs 閾值）
#
# 使用方式：
#   from threshold_tuner import ThresholdTuner
#   tuner = ThresholdTuner(model, device)
#   report = tuner.scan_percentiles(X_normal, X_attack, y_attack_type)
#   tuner.plot_threshold_curve(report)
#   best = tuner.find_best_threshold(report, metric="f1")
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# 自動偵測中文字型
_CJK = ["Microsoft YaHei", "SimHei", "PingFang TC",
        "Heiti TC", "WenQuanYi Zen Hei", "Noto Sans CJK TC"]
_avail = {f.name for f in _fm.fontManager.ttflist}
for _f in _CJK:
    if _f in _avail:
        matplotlib.rcParams["font.family"] = [_f, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ThresholdTuner:
    """
    閾值調校器

    原版 trainer.py 的 compute_threshold() 只用單一百分位數，
    本模組提供系統性的閾值掃描與攻擊類型分組評估。

    Args:
        model : 已訓練完成的 CNNAutoencoder
        device: torch.device
    """

    def __init__(self, model, device=None):
        self.model  = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    # ── 計算重建誤差 ──────────────────────────────────────
    def _compute_errors(self, X: np.ndarray) -> np.ndarray:
        """
        對影像矩陣計算每筆樣本的重建誤差（MSE）

        Args:
            X: shape=(N, 32, 32) float32
        Returns:
            errors: shape=(N,) float32
        """
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]            # (N,1,32,32)
        tensor  = torch.from_numpy(X.astype(np.float32))
        dataset = TensorDataset(tensor)
        loader  = DataLoader(dataset, batch_size=128, shuffle=False)

        all_errors = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                err   = self.model.reconstruction_error(batch)
                all_errors.append(err.cpu().numpy())

        return np.concatenate(all_errors)

    # ── 單一閾值評估 ──────────────────────────────────────
    @staticmethod
    def _evaluate_at_threshold(errors_normal: np.ndarray,
                               errors_attack: np.ndarray,
                               threshold: float) -> dict:
        """
        給定閾值，計算 TP/TN/FP/FN 與各評估指標

        判斷邏輯：
            誤差 > 閾值 → 判定為攻擊（Positive）
            誤差 ≤ 閾值 → 判定為正常（Negative）

        Args:
            errors_normal: 正常封包的重建誤差陣列
            errors_attack: 攻擊封包的重建誤差陣列
            threshold    : 判斷閾值

        Returns:
            dict 包含 tp/tn/fp/fn/precision/recall/f1/fpr/accuracy
        """
        # 正常流量被誤判為攻擊 → False Positive
        fp = int((errors_normal > threshold).sum())
        # 正常流量正確判定 → True Negative
        tn = len(errors_normal) - fp
        # 攻擊流量正確偵測 → True Positive
        tp = int((errors_attack > threshold).sum())
        # 攻擊流量漏掉 → False Negative
        fn = len(errors_attack) - tp

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        fpr       = fp / (fp + tn + 1e-9)
        accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-9)

        return {
            "threshold": threshold,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "fpr":       fpr,
            "accuracy":  accuracy,
        }

    # ── 多百分位數掃描 ────────────────────────────────────
    def scan_percentiles(self,
                         X_normal: np.ndarray,
                         X_attack: np.ndarray,
                         percentiles: list = None) -> list:
        """
        掃描多個百分位數閾值，找出 Precision / Recall / F1 的變化趨勢

        原版 trainer.py 只計算單一百分位（預設 95th），
        本方法讓使用者看到「調高閾值 → 誤報減少但漏報增加」的完整曲線，
        從而選出最符合需求的閾值。

        Args:
            X_normal    : 正常流量影像矩陣 (N1, 32, 32)
            X_attack    : 攻擊流量影像矩陣 (N2, 32, 32)
            percentiles : 要掃描的百分位數列表
                         （預設 [80, 85, 90, 92, 95, 97, 99]）

        Returns:
            list of dict，每個 dict 包含一個百分位數對應的評估指標
        """
        if percentiles is None:
            percentiles = [80, 85, 90, 92, 95, 97, 99]

        print("\n  [ThresholdTuner] 計算重建誤差...")
        errors_normal = self._compute_errors(X_normal)
        errors_attack = self._compute_errors(X_attack)

        print(f"  正常流量誤差範圍: [{errors_normal.min():.6f}, {errors_normal.max():.6f}]")
        print(f"  攻擊流量誤差範圍: [{errors_attack.min():.6f}, {errors_attack.max():.6f}]")
        print(f"  攻擊 / 正常誤差比值: "
              f"{errors_attack.mean() / (errors_normal.mean() + 1e-9):.2f}x")

        print(f"\n  {'百分位':>6}  {'閾值':>10}  {'Precision':>10}  "
              f"{'Recall':>8}  {'F1':>8}  {'FPR':>8}")
        print("  " + "-" * 58)

        results = []
        for pct in percentiles:
            # 以正常流量誤差的第 pct 百分位數作為閾值
            # 意義：pct% 的正常封包誤差低於閾值，(100-pct)% 的正常封包被誤判
            threshold = float(np.percentile(errors_normal, pct))
            metrics   = self._evaluate_at_threshold(
                errors_normal, errors_attack, threshold
            )
            metrics["percentile"] = pct
            results.append(metrics)

            print(f"  {pct:>6}th  {threshold:>10.6f}  "
                  f"{metrics['precision']:>10.4f}  "
                  f"{metrics['recall']:>8.4f}  "
                  f"{metrics['f1']:>8.4f}  "
                  f"{metrics['fpr']:>8.4f}")

        self._errors_normal = errors_normal
        self._errors_attack = errors_attack
        return results

    # ── 自動選最佳閾值 ────────────────────────────────────
    def find_best_threshold(self, results: list,
                            metric: str = "f1") -> dict:
        """
        從掃描結果中自動選出最佳閾值

        不同場景的選擇策略：
            metric="f1"       → 最大化 F1 Score（Precision 與 Recall 平衡）
            metric="recall"   → 最大化偵測率（零日攻擊、高風險環境）
            metric="precision"→ 最大化精確率（降低誤報、一般監控環境）

        Args:
            results: scan_percentiles() 的回傳值
            metric : 最佳化目標

        Returns:
            最佳閾值設定的 dict
        """
        best = max(results, key=lambda r: r[metric])
        print(f"\n  [最佳閾值] 依 {metric.upper()} 最大化：")
        print(f"    百分位數 : {best['percentile']}th")
        print(f"    閾值     : {best['threshold']:.6f}")
        print(f"    Precision: {best['precision']:.4f}")
        print(f"    Recall   : {best['recall']:.4f}")
        print(f"    F1 Score : {best['f1']:.4f}")
        print(f"    FPR      : {best['fpr']:.4f}")
        print(f"    TP={best['tp']}  TN={best['tn']}  "
              f"FP={best['fp']}  FN={best['fn']}")
        return best

    # ── 攻擊類型分組評估 ──────────────────────────────────
    def evaluate_by_attack_type(self,
                                X_normal: np.ndarray,
                                X_attack: np.ndarray,
                                attack_labels: np.ndarray,
                                threshold: float) -> dict:
        """
        針對不同攻擊類型分別評估同一閾值的偵測效果

        CIC-IDS2017 包含多種攻擊類型，同一個閾值對 DoS 與 PortScan
        的偵測效果可能差異很大。本方法讓使用者了解哪種攻擊類型偵測率較低，
        進而決定是否要為不同類型設定不同閾值。

        Args:
            X_normal     : 正常流量影像 (N1, 32, 32)
            X_attack     : 攻擊流量影像 (N2, 32, 32)
            attack_labels: 攻擊類型標籤陣列 shape=(N2,)，例如 ["DoS", "PortScan", ...]
            threshold    : 要評估的閾值

        Returns:
            dict 鍵為攻擊類型名稱，值為該類型的評估指標
        """
        errors_normal = self._compute_errors(X_normal)
        errors_attack = self._compute_errors(X_attack)

        attack_types = np.unique(attack_labels)
        report = {}

        print(f"\n  [攻擊類型分組評估] 閾值 = {threshold:.6f}")
        print(f"  {'攻擊類型':<20}  {'樣本數':>6}  "
              f"{'TP':>5}  {'FN':>5}  {'Recall':>8}")
        print("  " + "-" * 55)

        for atype in attack_types:
            mask    = attack_labels == atype
            errors_this = errors_attack[mask]
            tp  = int((errors_this > threshold).sum())
            fn  = len(errors_this) - tp
            rec = tp / (tp + fn + 1e-9)

            report[atype] = {
                "count": int(mask.sum()),
                "tp": tp, "fn": fn,
                "recall": rec,
            }
            print(f"  {atype:<20}  {int(mask.sum()):>6}  "
                  f"{tp:>5}  {fn:>5}  {rec:>8.4f}")

        return report

    # ── 閾值曲線圖 ────────────────────────────────────────
    def plot_threshold_curve(self,
                             results: list,
                             output_dir: str = "output/model"):
        """
        繪製「百分位數 vs Precision / Recall / F1」曲線圖

        讓使用者視覺化地看到閾值調整的 trade-off：
          - 百分位數越高（閾值越嚴）→ Precision 升高，Recall 下降
          - 最佳 F1 Score 對應的百分位數是最均衡的選擇

        Args:
            results   : scan_percentiles() 的回傳值
            output_dir: 圖片輸出目錄
        """
        pcts  = [r["percentile"] for r in results]
        precs = [r["precision"]  for r in results]
        recs  = [r["recall"]     for r in results]
        f1s   = [r["f1"]         for r in results]
        fprs  = [r["fpr"]        for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0d1117")

        # 左圖：P / R / F1 vs 百分位數
        ax = axes[0]
        ax.set_facecolor("#161b22")
        ax.plot(pcts, precs, "o-", color="#3fb950", linewidth=2, label="Precision")
        ax.plot(pcts, recs,  "s-", color="#58a6ff", linewidth=2, label="Recall")
        ax.plot(pcts, f1s,   "^-", color="#e3b341", linewidth=2, label="F1 Score")
        ax.set_xlabel("閾值百分位數 (Percentile)", color="#8b949e")
        ax.set_ylabel("指標值",                   color="#8b949e")
        ax.set_title("閾值調整對 Precision / Recall / F1 的影響",
                     color="white", fontsize=11)
        ax.legend(facecolor="#161b22", labelcolor="white")
        ax.tick_params(colors="#8b949e")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(pcts)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        # 右圖：誤差分布（正常 vs 攻擊）
        ax2 = axes[1]
        ax2.set_facecolor("#161b22")
        if hasattr(self, "_errors_normal") and hasattr(self, "_errors_attack"):
            ax2.hist(self._errors_normal, bins=50, alpha=0.6,
                     color="#3fb950", label="正常流量", density=True)
            ax2.hist(self._errors_attack, bins=50, alpha=0.6,
                     color="#f85149", label="攻擊流量", density=True)
            # 標示幾個代表性閾值
            for r in results[::2]:
                ax2.axvline(r["threshold"], color="#e3b341",
                            linewidth=0.8, alpha=0.6,
                            linestyle="--",
                            label=f"{r['percentile']}th pct")
        ax2.set_xlabel("重建誤差 (MSE)", color="#8b949e")
        ax2.set_ylabel("密度",           color="#8b949e")
        ax2.set_title("正常 vs 攻擊流量重建誤差分布",
                      color="white", fontsize=11)
        ax2.legend(facecolor="#161b22", labelcolor="white", fontsize=8)
        ax2.tick_params(colors="#8b949e")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#30363d")

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "threshold_curve.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"\n  [圖表] 閾值調校曲線已儲存: {path}")
        return path

    # ── 消融實驗輔助 ──────────────────────────────────────
    def ablation_compare(self,
                         experiments: list,
                         output_dir: str = "output/model") -> dict:
        """
        消融實驗：比較不同設計決策對偵測效能的影響

        每個實驗是一個 dict：
            {
                "name":     實驗名稱（例如「有遮罩」）,
                "model":    已訓練的 CNNAutoencoder,
                "X_normal": 正常流量影像矩陣,
                "X_attack": 攻擊流量影像矩陣,
                "pct":      使用的閾值百分位數（預設 95）
            }

        設計用來比較：
            - 有無欄位遮罩（apply_mask=True vs False）
            - 不同影像尺寸（28x28 vs 32x32 vs 40x40）
            - 不同 latent_dim（16 vs 32 vs 64）

        Args:
            experiments: 實驗設定列表
            output_dir : 圖表輸出目錄

        Returns:
            dict 鍵為實驗名稱，值為評估指標
        """
        results = {}
        print(f"\n  [消融實驗]")
        print(f"  {'實驗名稱':<20}  {'Precision':>10}  "
              f"{'Recall':>8}  {'F1':>8}  {'閾值':>10}")
        print("  " + "-" * 62)

        for exp in experiments:
            name      = exp["name"]
            model_exp = exp.get("model", self.model)
            X_n       = exp["X_normal"]
            X_a       = exp["X_attack"]
            pct       = exp.get("pct", 95)

            # 暫時切換模型
            orig_model  = self.model
            self.model  = model_exp
            self.model.eval()

            errors_n = self._compute_errors(X_n)
            errors_a = self._compute_errors(X_a)
            threshold = float(np.percentile(errors_n, pct))
            metrics   = self._evaluate_at_threshold(errors_n, errors_a, threshold)
            metrics["percentile"] = pct

            self.model = orig_model   # 還原

            results[name] = metrics
            print(f"  {name:<20}  {metrics['precision']:>10.4f}  "
                  f"{metrics['recall']:>8.4f}  "
                  f"{metrics['f1']:>8.4f}  "
                  f"{threshold:>10.6f}")

        # 繪製消融實驗對比圖
        self._plot_ablation(results, output_dir)
        return results

    def _plot_ablation(self, results: dict, output_dir: str):
        """繪製消融實驗結果長條圖"""
        names  = list(results.keys())
        precs  = [results[n]["precision"] for n in names]
        recs   = [results[n]["recall"]    for n in names]
        f1s    = [results[n]["f1"]        for n in names]

        x   = np.arange(len(names))
        w   = 0.25
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        ax.bar(x - w, precs, w, label="Precision", color="#3fb950", alpha=0.85)
        ax.bar(x,     recs,  w, label="Recall",    color="#58a6ff", alpha=0.85)
        ax.bar(x + w, f1s,   w, label="F1 Score",  color="#e3b341", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(names, color="#8b949e", rotation=15)
        ax.set_ylabel("指標值",    color="#8b949e")
        ax.set_title("消融實驗：各設計決策對偵測效能的影響",
                     color="white", fontsize=11)
        ax.legend(facecolor="#161b22", labelcolor="white")
        ax.tick_params(colors="#8b949e")
        ax.set_ylim(0, 1.1)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "ablation_results.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        print(f"  [圖表] 消融實驗圖已儲存: {path}")
