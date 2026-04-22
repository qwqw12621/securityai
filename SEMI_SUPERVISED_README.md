
# 半監督式學習模組 — 使用說明與驗證步驟

## 一、模組概覽

本次新增 / 修改的檔案：

| 檔案 | 類型 | 說明 |
|------|------|------|
| `semi_supervised_trainer.py`  半監督式訓練器（核心模組） |
| `run_semi_supervised.py`      完整訓練流程主程式 |
| `threshold_tuner_semi.py`     半監督增強版閾值調校器 |
| `test_semi_supervised.py`     單元測試（驗證用） |

---

## 二、半監督學習原理

### 為何改用半監督式學習？

| 比較項目 | 純非監督（原版）      | 半監督（本方案） |
|---------|----------------------|---------------|
| 訓練資料 | 只用正常流量          | 正常流量 + 少量標記攻擊 |
| 損失函數 | MSE（重建損失）       | MSE + Margin Loss（邊界損失） |
| 閾值選擇 | 固定百分位(如 95th)   | 利用攻擊樣本掃描最佳 F1 |
| 偵測率   | 中等                 | 通常提升 5~15%   |
| 誤報率   | 中等                 | 通常下降 |

### 損失函數設計

```
L_total = α × L_recon_normal + β × L_margin_attack

L_recon_normal  = MSE(x_normal, reconstruct(x_normal))
L_margin_attack = mean(max(0, margin - MSE(x_attack, reconstruct(x_attack))))

參數：
  α（alpha）= 正常重建損失權重，預設 1.0
  β（beta） = 攻擊邊界損失權重，預設 0.5
  margin    = 希望攻擊誤差超過的目標值，預設 0.05
```

### 兩階段訓練流程

```
Phase 1：無監督預訓練（Unsupervised Pretraining）
    ├── 資料：只使用正常流量（X_normal）
    ├── 損失：純重建損失 MSE
    └── 目標：讓模型學會重建正常封包的典型結構

Phase 2：半監督微調（Semi-supervised Fine-tuning）
    ├── 資料：正常流量 + 少量標記攻擊流量（attack_ratio=20%）
    ├── 損失：α×L_recon + β×L_margin
    └── 目標：進一步拉大正常/攻擊流量的誤差差距
```

---

## 三、快速開始

### 步驟 1：環境確認

確認以下套件已安裝：
```bash
pip install torch numpy matplotlib scikit-learn
```

### 步驟 2：將新檔案放入 core/ 目錄

```
network_capture/
└── core/
    ├── semi_supervised_trainer.py   ← 新增
    ├── run_semi_supervised.py       ← 新增
    ├── threshold_tuner_semi.py      ← 新增
    └── test_semi_supervised.py      ← 新增
```

### 步驟 3：執行訓練

```bash
# 進入 core/ 目錄
cd core/

# 快速測試（模擬資料，無需下載，約 2~3 分鐘）
python run_semi_supervised.py

# 使用真實資料集
python run_semi_supervised.py \
    --dataset cicids2017 \
    --data-dir data/cicids2017

# 開啟對比實驗（同時訓練非監督基線）
python run_semi_supervised.py --compare-unsupervised

# 完整選項範例
python run_semi_supervised.py \
    --dataset simulate \
    --pretrain-epochs 80 \
    --finetune-epochs 60 \
    --alpha 1.0 \
    --beta 0.5 \
    --margin 0.05 \
    --attack-ratio 0.2 \
    --threshold-method optimal \
    --compare-unsupervised
```

---

## 四、程式碼功能驗證方法

### 方法 A：單元測試（推薦）

```bash
# 在 core/ 目錄下執行
cd core/

# 執行全部測試
pytest test_semi_supervised.py -v

# 只執行特定測試類別
pytest test_semi_supervised.py -v -k "TestMarginLoss"
pytest test_semi_supervised.py -v -k "TestSemiSupervisedDataLoader"
pytest test_semi_supervised.py -v -k "TestSemiSupervisedTrainer"
pytest test_semi_supervised.py -v -k "TestSemiSupervisedThresholdTuner"
```

預期輸出範例：
```
test_semi_supervised.py::TestMarginLoss::test_loss_zero_when_errors_exceed_margin PASSED
test_semi_supervised.py::TestMarginLoss::test_loss_positive_when_errors_below_margin PASSED
test_semi_supervised.py::TestMarginLoss::test_gradient_pushes_errors_up PASSED
test_semi_supervised.py::TestMarginLoss::test_partial_loss_calculation PASSED
test_semi_supervised.py::TestMarginLoss::test_loss_with_different_margins PASSED
...
PASSED 23 項，FAILED 0 項
```

### 方法 B：快速功能驗證腳本

```python
# quick_validate.py - 快速驗證（在 core/ 目錄下執行）

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

print("=" * 55)
print("  半監督模組快速驗證")
print("=" * 55)

# 1. 生成合成測試資料
np.random.seed(42)
X_normal = np.random.beta(2, 5, (300, 32, 32)).astype(np.float32)
X_attack = np.zeros((100, 32, 32), dtype=np.float32)
X_attack[:, :8, :8] = 0.9   # 攻擊特徵

# 2. 驗證 MarginLoss
from semi_supervised_trainer import MarginLoss
print("\n[1] 驗證 MarginLoss...")
loss_fn = MarginLoss(margin=0.05)
errors_low  = torch.tensor([0.01, 0.02])  # 低誤差（應有損失）
errors_high = torch.tensor([0.10, 0.20])  # 高誤差（應無損失）
assert loss_fn(errors_low).item() > 0,  "✓ 低誤差應有損失"
assert loss_fn(errors_high).item() == 0, "✓ 高誤差損失為 0"
print("  MarginLoss 邏輯驗證通過 ✓")

# 3. 驗證 SemiSupervisedDataLoader
from semi_supervised_trainer import SemiSupervisedDataLoader
print("\n[2] 驗證 SemiSupervisedDataLoader...")
dm = SemiSupervisedDataLoader(X_normal, X_attack,
                               attack_ratio=0.2, batch_size=16)
assert dm.n_attack_labeled == 20, "攻擊樣本數正確 ✓"
print(f"  資料分配正確（標記攻擊樣本：{dm.n_attack_labeled}）✓")

# 4. 驗證訓練流程（2 epochs）
from semi_supervised_trainer import SemiSupervisedTrainer
print("\n[3] 驗證訓練流程（2 epochs）...")
config = {
    "latent_dim": 8, "batch_size": 16,
    "pretrain_epochs": 2, "finetune_epochs": 2,
    "learning_rate": 1e-3, "patience": 10,
    "alpha": 1.0, "beta": 0.5, "margin": 0.05,
    "attack_ratio": 0.2, "val_split": 0.2,
    "weight_decay": 1e-5, "min_delta": 1e-8,
    "lr_patience": 5, "lr_factor": 0.5,
}
trainer = SemiSupervisedTrainer(config=config, output_dir="/tmp/test_semi")
trainer.load_data(X_normal, X_attack)
trainer.pretrain()
print(f"  Phase 1 完成，損失: {trainer.pretrain_losses[-1]:.6f} ✓")
trainer.finetune()
print(f"  Phase 2 完成，總損失: {trainer.finetune_total[-1]:.6f} ✓")

# 5. 驗證閾值計算
threshold = trainer.compute_threshold(X_normal, X_attack, method="optimal")
assert threshold > 0, "閾值應為正數"
print(f"  最佳閾值: {threshold:.6f} ✓")

# 6. 驗證閾值調校器
from threshold_tuner_semi import SemiSupervisedThresholdTuner
print("\n[4] 驗證閾值調校器...")
device = torch.device("cpu")
tuner = SemiSupervisedThresholdTuner(trainer.model, device)
results = tuner.calibrate_with_labels(X_normal, X_attack, n_thresholds=50)
assert len(results) == 50, "應有 50 個候選閾值"
best = max(results, key=lambda r: r["f1"])
print(f"  最佳 F1={best['f1']:.4f}，閾值={best['threshold']:.6f} ✓")

recs = tuner.multi_strategy_report(results)
assert len(recs) == 3, "應有 3 種策略"
print(f"  三種策略推薦已生成 ✓")

print("\n" + "=" * 55)
print("  全部驗證通過！")
print("=" * 55)
```

執行方式：
```bash
cd core/
python quick_validate.py
```

### 方法 C：端對端效能驗證

```bash
# 使用模擬資料做完整驗證（約 5~10 分鐘）
cd core/
python run_semi_supervised.py \
    --compare-unsupervised \
    --pretrain-epochs 50 \
    --finetune-epochs 50
```

驗證輸出示例（應看到此類結果）：
```
======================================================
  改善幅度（半監督 vs 非監督）：
    precision      : +0.0823
    recall         : +0.1256
    f1_score       : +0.1034
    accuracy       : +0.0712
    分離比改善      : +2.14x
======================================================
```

---

## 五、詳細參數說明

### run_semi_supervised.py 主要參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--pretrain-epochs` | 50 | Phase 1 無監督預訓練輪數 |
| `--finetune-epochs` | 50 | Phase 2 半監督微調輪數 |
| `--alpha` | 1.0 | 正常流量重建損失權重 |
| `--beta` | 0.5 | 攻擊邊界損失權重 |
| `--margin` | 0.05 | 邊界損失目標值 |
| `--attack-ratio` | 0.2 | 用於微調的標記攻擊比例 |
| `--threshold-method` | optimal | 閾值計算方法（optimal/percentile） |
| `--compare-unsupervised` | False | 是否同時訓練非監督基線做對比 |

### 參數調校建議

**正確的 `margin` 設定：**
```
margin ≈ 2~5 × 純非監督模型的正常流量平均誤差

例如：若非監督模型正常流量誤差均值 = 0.01
      則 margin 建議設為 0.02~0.05
```

**正確的 `beta` 設定：**
```
beta 太小（<0.1）：邊界損失效果不明顯
beta 適中（0.3~0.8）：最佳平衡點
beta 太大（>1.5）：可能破壞重建能力，正常流量誤差也上升
```

**正確的 `attack_ratio` 設定：**
```
標記樣本越多，效果越好，但標記成本也越高
通常 10~30% 即可看到明顯效果
```

---

## 六、輸出檔案說明

訓練完成後，`output/model_semi/` 目錄下會包含：

```
output/model_semi/
├── best_model.pt                   # 最佳半監督模型權重
├── pretrained_model.pt             # Phase 1 預訓練模型（中間結果）
├── semi_training_curve.png         # Phase 1 + Phase 2 訓練曲線
├── semi_error_distribution.png     # 正常/攻擊誤差分布圖
├── semi_threshold_curve.png        # 閾值調校曲線（P/R/F1 vs 閾值）
├── semi_training_result.json       # 訓練設定與閾值
├── semi_evaluation_report.json     # 效能評估報告（P/R/F1/AUC）
├── comparison_report.png           # 半監督 vs 非監督對比圖（加 --compare）
└── comparison_data.json            # 對比數據
```

---

## 七、與原系統的整合

### 在 Django 的 `api/views.py` 中使用半監督模型

```python
# api/views.py 中的 CNNAnalyzeAPI.post() 方法
# 修改模型載入路徑指向半監督模型

model_path = str(settings.CNN_MODEL_PATH)  # 指向 output/model_semi/best_model.pt
```

### 更新 `settings.py`

```python
CNN_MODEL_PATH = BASE_DIR / "media" / "model" / "best_model.pt"
CNN_THRESHOLD  = 0.023456  # 從 semi_evaluation_report.json 的 threshold 欄位取得
```

### 一鍵部署

```bash
# 1. 訓練半監督模型
python core/run_semi_supervised.py \
    --dataset cicids2017 \
    --data-dir data/cicids2017 \
    --threshold-method optimal

# 2. 複製模型到 media 目錄
cp output/model_semi/best_model.pt media/model/best_model.pt

# 3. 從評估報告取得閾值並更新 settings.py
python -c "
import json
with open('output/model_semi/semi_evaluation_report.json') as f:
    d = json.load(f)
print(f'CNN_THRESHOLD = {d[\"threshold\"]:.6f}')
"

# 4. 重啟 Django
python manage.py runserver
```
