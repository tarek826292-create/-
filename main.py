"""
金融风控数据建模 - 主程序入口
完整流程：数据加载 → EDA → 特征工程 → 不平衡处理 → 模型训练 → 评估 → SHAP → 报告
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# 让模块导入可见
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_data, basic_info
from eda import run_eda, eda_summary
from preprocess import feature_engineer, split_and_scale, apply_smote
from train import train_all, get_best
from evaluate import compute_metrics
from explain import shap_analysis
from visualize import (
    plot_roc, plot_pr, plot_ks, plot_model_bar,
    plot_confusion, plot_proba_dist
)

OUT = Path(__file__).parent / "output"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def banner(s: str):
    print()
    print("=" * 70)
    print(s)
    print("=" * 70)


def main():
    t_total = time.time()

    banner("STEP 1 / 6  数据加载与基础信息")
    df = load_data()
    info = basic_info(df)
    for k, v in info.items():
        print(f"  {k}: {v}")

    banner("STEP 2 / 6  探索性分析 (EDA)")
    t0 = time.time()
    eda_res = run_eda()
    print(f"  EDA done in {time.time() - t0:.1f}s")

    banner("STEP 3 / 6  特征工程 & 数据划分")
    df_fe = feature_engineer(df)
    X_train, X_test, y_train, y_test, scaler, feat_cols = split_and_scale(df_fe)
    print(f"  train={X_train.shape}, test={X_test.shape}, features={len(feat_cols)}")
    print(f"  train fraud={int(y_train.sum())}, test fraud={int(y_test.sum())}")

    # 准备两种策略的训练集
    X_train_cw, y_train_cw = X_train, y_train
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    print(f"  SMOTE train shape: {X_train_sm.shape}, fraud after SMOTE: {int(y_train_sm.sum())}")

    banner("STEP 4 / 6  模型训练 (4 模型 × 2 策略)")
    t0 = time.time()
    print("  >>> Strategy A: class_weight / scale_pos_weight")
    res_a = train_all(X_train_cw, y_train_cw, X_test, use_smote=False)
    print(f"  Strategy A done in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("  >>> Strategy B: SMOTE")
    res_b = train_all(X_train_sm, y_train_sm, X_test, use_smote=True)
    print(f"  Strategy B done in {time.time() - t0:.1f}s")

    # 汇总
    all_results = []
    for r in res_a:
        m = compute_metrics(y_test, r["proba"])
        m["model"] = r["name"]
        m["strategy"] = "class_weight"
        m["fit_time_s"] = r["fit_time"]
        all_results.append(m)
    for r in res_b:
        m = compute_metrics(y_test, r["proba"])
        m["model"] = r["name"]
        m["strategy"] = "SMOTE"
        m["fit_time_s"] = r["fit_time"]
        all_results.append(m)

    metrics_df = pd.DataFrame(all_results).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    metrics_csv = OUT / "metrics_summary.csv"
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    print("\n  Metrics summary:")
    print(metrics_df[["model", "strategy", "auc", "pr_auc", "ks", "f1@0.5", "recall@1%", "recall@5%"]].to_string(index=False))

    # 最佳模型：以 PR-AUC 为准
    best = max(all_results, key=lambda x: x["pr_auc"])
    print(f"\n  >>> Best model: {best['model']} ({best['strategy']}), PR-AUC={best['pr_auc']:.4f}")

    banner("STEP 5 / 6  可视化 (ROC / PR / KS / 混淆矩阵 / 概率分布)")
    # 准备 8 个模型的 proba
    proba_dict = {}
    for r in res_a + res_b:
        key = f"{r['name']}-{'CW' if r['use_smote'] == False else 'SMOTE'}"
        proba_dict[key] = r["proba"]
    plot_roc(proba_dict, y_test)
    plot_pr(proba_dict, y_test)
    plot_ks(proba_dict, y_test)
    plot_model_bar(metrics_df)

    # 找到最佳策略对应的 proba 与模型
    best_key = f"{best['model']}-{'CW' if best['strategy'] == 'class_weight' else 'SMOTE'}"
    best_proba = proba_dict[best_key]
    # 取对应 fitted model
    src_list = res_a if best["strategy"] == "class_weight" else res_b
    best_model = next(r["model"] for r in src_list if r["name"] == best["model"])
    joblib.dump(best_model, OUT / "best_model.pkl")
    print(f"  Saved best model -> {OUT / 'best_model.pkl'}")

    # 阈值 0.5 下的混淆矩阵
    plot_confusion(y_test, best_proba, threshold=0.5, name=best_key, save_name="14_confusion_best_t05.png")
    # 业务推荐阈值
    recommend_t = 0.2 if best["strategy"] == "SMOTE" else 0.5
    plot_confusion(y_test, best_proba, threshold=recommend_t, name=best_key, save_name="15_confusion_best_recommend.png")
    plot_proba_dist(y_test, best_proba, name=best_key, save_name="16_proba_dist.png")

    banner("STEP 6 / 6  SHAP 可解释性分析")
    # 用原始训练集（不做 SMOTE）取子样本，避免内存问题
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(X_train.shape[0], size=min(1500, X_train.shape[0]), replace=False)
    X_sample = X_train[sample_idx]
    try:
        shap_res = shap_analysis(best_model, X_sample, feat_cols, model_name=best_key)
        print("  Top 10 features by mean |SHAP|:")
        for f, v in shap_res["top_features"]:
            print(f"    {f:>10s}  {v:.4f}")
    except Exception as e:
        print(f"  [WARN] SHAP failed: {e}")
        shap_res = {"figures": [], "top_features": []}

    # 写入总览 JSON 供报告生成使用
    overview = {
        "data_info": info,
        "eda_summary": eda_res,
        "metrics": metrics_df.to_dict(orient="records"),
        "best": {
            "name": best["model"],
            "strategy": best["strategy"],
            "key": best_key,
            "metrics": {k: v for k, v in best.items() if k not in ("model",)},
            "shap_top": shap_res.get("top_features", []),
        },
        "feature_count": len(feat_cols),
        "feat_cols": feat_cols,
    }
    with open(OUT / "overview.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Saved overview -> {OUT / 'overview.json'}")

    banner("DONE")
    print(f"  Total time: {time.time() - t_total:.1f}s")
    print(f"  All artifacts: {OUT}")


if __name__ == "__main__":
    main()
