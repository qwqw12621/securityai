# ============================================================
# run_threshold_tuning.py - 模型訓練與閾值調校完整腳本
#
# 使用資料集：
#   CIC-IDS2017（主要）：真實 DoS / DDoS / PortScan 攻擊流量
#     下載：https://www.unb.ca/cic/datasets/ids-2017.html
#     放置：data/cicids2017/*.csv
#
#   NSL-KDD（輔助對照）：較小、格式簡單、適合快速驗證
#     下載：https://www.unb.ca/cic/datasets/nsl.html
#     放置：data/nslkdd/KDDTrain+.txt
#
#   模擬資料（開發測試，無需下載）：
#     python core/run_threshold_tuning.py --dataset simulate
#
# 執行方式：
#   python core/run_threshold_tuning.py --dataset simulate
#   python core/run_threshold_tuning.py --dataset cicids2017 --data-dir data/cicids2017
#   python core/run_threshold_tuning.py --dataset nslkdd --data-dir data/nslkdd
#   python core/run_threshold_tuning.py --ablation       # 執行消融實驗
#
# 輸出：
#   output/model/best_model.pt              最佳模型權重
#   output/model/training_curve.png         訓練曲線
#   output/model/threshold_curve.png        閾值調校曲線
#   output/model/ablation_results.png       消融實驗圖（--ablation 時）
#   output/model/threshold_report.json      各閾值評估結果
# ============================================================

import os
import sys
import json
import argparse
import numpy as np

# 確保 core 目錄在 sys.path 中
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


def check_torch():
    try:
        import torch
        print(f"  PyTorch {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        return True
    except ImportError:
        print("  PyTorch 未安裝")
        print("  請執行: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="模型訓練與閾值調校",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  python core/run_threshold_tuning.py
  python core/run_threshold_tuning.py --dataset cicids2017 --data-dir data/cicids2017
  python core/run_threshold_tuning.py --dataset nslkdd --data-dir data/nslkdd
  python core/run_threshold_tuning.py --ablation
  python core/run_threshold_tuning.py --eval-only --model output/model/best_model.pt
        """
    )
    parser.add_argument("--dataset",   default="simulate",
                        choices=["simulate", "nslkdd", "cicids2017", "cicddos2019"])
    parser.add_argument("--data-dir",  default=None)
    parser.add_argument("--epochs",    type=int,   default=100)
    parser.add_argument("--latent",    type=int,   default=32)
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--output",    default="output/model")
    parser.add_argument("--ablation",  action="store_true",
                        help="執行消融實驗（欄位遮罩 / 影像尺寸 / latent_dim 對比）")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model",     default=None)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(f"  模型訓練與閾值調校（資料集：{args.dataset}）")
    print("=" * 60)

    if not check_torch():
        sys.exit(1)

    import torch
    from trainer import Trainer
    from dataset_loader import DatasetFactory
    from threshold_tuner import ThresholdTuner
    from cnn_autoencoder import CNNAutoencoder

    # ══════════════════════════════════════════════════════
    # Step 1：載入資料集
    # ══════════════════════════════════════════════════════
    print(f"\n[Step 1] 載入資料集：{args.dataset}")
    print()

    # 使用 DatasetFactory 統一介面，
    # 三種真實資料集都輸出相同格式的 (N, 32, 32) float32 矩陣
    load_kwargs = {}
    if args.dataset in ("cicids2017", "cicddos2019"):
        load_kwargs["max_normal"] = 50000
        load_kwargs["max_attack"] = 30000

    try:
        X_normal, X_attack, y = DatasetFactory.load(
            args.dataset,
            data_dir=args.data_dir,
            **load_kwargs
        )
    except FileNotFoundError as e:
        print(f"\n  錯誤：{e}")
        sys.exit(1)

    print(f"  正常流量: {len(X_normal):,} 筆  攻擊流量: {len(X_attack):,} 筆")

    # 儲存為 .npy 供後續分析使用
    dataset_dir = f"output/dataset_{args.dataset}"
    paths = DatasetFactory.save_as_npy(X_normal, X_attack, y, dataset_dir)
    normal_npy = paths["normal"]
    attack_npy = paths["attack"]

    # ══════════════════════════════════════════════════════
    # Step 2：訓練模型
    # ══════════════════════════════════════════════════════
    model_path = args.model or os.path.join(args.output, "best_model.pt")
    config_path = os.path.join(args.output, "training_result.json")

    if args.eval_only:
        print(f"\n[Step 2] 載入現有模型: {model_path}")
        model, threshold = Trainer.load_model(model_path, config_path, args.latent)
    else:
        print("\n[Step 2] 訓練 CNN Autoencoder")
        config = {
            "latent_dim": args.latent,
            "batch_size": args.batch,
            "epochs":     args.epochs,
            "learning_rate": 1e-3,
        }
        trainer = Trainer(config=config, output_dir=args.output)
        trainer.load_data(normal_npy)
        trainer.train()
        trainer.plot_training_curve()
        model = trainer.model

    # ══════════════════════════════════════════════════════
    # Step 3：閾值調校與分析
    # ══════════════════════════════════════════════════════
    print("\n[Step 3] 執行閾值調校分析 (ThresholdTuner)")
    tuner = ThresholdTuner(model)
    
    # 掃描不同百分位數的成效
    report = tuner.scan_percentiles(X_normal, X_attack)
    
    # 儲存報告
    with open(os.path.join(args.output, "threshold_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    
    # 繪製閾值曲線 (Precision-Recall vs Percentile)
    tuner.plot_threshold_curve(report)

    # ══════════════════════════════════════════════════════
    # Step 4：消融實驗（Ablation Study）
    # ══════════════════════════════════════════════════════
    if args.ablation and not args.eval_only:
        print("\n[Step 4] 執行消融實驗 (Ablation Study)")
        # 測試不同的 latent_dim 對效能的影響
        latent_dims = [8, 16, 32, 64]
        ablation_results = []

        for ld in latent_dims:
            print(f"  測試 latent_dim = {ld}...")
            config = {
                "latent_dim": ld,
                "batch_size": args.batch,
                "epochs":     20,  # 消融實驗用較少 epoch 快速掃描
            }
            tmp_output = f"output/ablation_ld_{ld}"
            trainer = Trainer(config=config, output_dir=tmp_output)
            trainer.load_data(normal_npy)
            trainer.train()
            
            # 評估 F1 Score (使用 95th percentile)
            threshold = trainer.compute_threshold(normal_npy, 95)
            from anomaly_scorer import AnomalyScorer
            scorer = AnomalyScorer(trainer.model, threshold=threshold)
            
            X_test = np.concatenate([X_normal, X_attack])
            y_test = np.concatenate([np.zeros(len(X_normal)), np.ones(len(X_attack))])
            res = scorer.evaluate(X_test, y_test)
            
            ablation_results.append({
                "latent_dim": ld,
                "f1": res["f1"],
                "auc": res["auc"]
            })

        # 繪製消融實驗結果圖
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        ld_vals = [r["latent_dim"] for r in ablation_results]
        f1_vals = [r["f1"] for r in ablation_results]
        auc_vals = [r["auc"] for r in ablation_results]
        
        plt.plot(ld_vals, f1_vals, "o-", label="F1 Score")
        plt.plot(ld_vals, auc_vals, "s-", label="AUC-ROC")
        plt.xlabel("Latent Dimension")
        plt.ylabel("Score")
        plt.title("Ablation Study: Effect of Latent Dimension")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output, "ablation_results.png"))
        print(f"  消融實驗完成，結果儲存至 {args.output}/ablation_results.png")

    print("\n" + "=" * 60)
    print("  閾值調校流程完成！")
    print(f"  報告輸出：{os.path.abspath(args.output)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
