"""
模型训练与对比
- 4 个模型：LogisticRegression / RandomForest / XGBoost / LightGBM
- 每模型分别训练：(a) 不平衡原比例 + class_weight / scale_pos_weight
                 (b) SMOTE 后训练
- 输出: 每个模型在测试集上的预测概率
"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

OUT_DIR = Path(__file__).parent / "output"


def build_models(use_smote: bool):
    """
    返回模型字典；不平衡处理方案取决于 use_smote。
    - use_smote=True: class_weight 不再额外配置（SMOTE 已平衡）
    - use_smote=False: 显式设置 class_weight / scale_pos_weight
    """
    if use_smote:
        # 数据已经平衡，无需 class_weight
        models = {
            "LR": LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42),
            "RF": RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42),
            "XGB": XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                eval_metric="aucpr", n_jobs=-1, random_state=42, verbosity=0
            ),
            "LGB": LGBMClassifier(
                n_estimators=300, max_depth=-1, learning_rate=0.05, num_leaves=31,
                objective="binary", n_jobs=-1, random_state=42, verbose=-1
            ),
        }
    else:
        models = {
            "LR": LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=42),
            "RF": RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                        n_jobs=-1, random_state=42),
        # XGB 用 scale_pos_weight (578 ≈ 正负比)
            "XGB": XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                eval_metric="aucpr", n_jobs=-1, random_state=42, verbosity=0,
                scale_pos_weight=578
            ),
            # LGB 用 class_weight={0:1, 1:50} 而不是 scale_pos_weight=578
            # 原因：scale_pos_weight=578 会把叶子输出分数放大到 sigmoid 饱和，
            # 反而让所有测试样本预测概率集中在接近 0 的区间，PR-AUC 退化到接近随机
            # 这里用 1:50 经验上更适合"原始数据 + 显式加权"场景
            "LGB": LGBMClassifier(
                n_estimators=400, learning_rate=0.05, num_leaves=63, max_depth=-1,
                objective="binary", n_jobs=-1, random_state=42, verbose=-1,
                class_weight={0: 1, 1: 50},
                min_child_samples=20, reg_alpha=0.1
            ),
        }
    return models


def train_all(X_train, y_train, X_test, use_smote: bool = False):
    """训练所有模型，返回 (model_name, fitted_model, predict_proba_test, fit_time) 列表"""
    results = []
    models = build_models(use_smote=use_smote)
    for name, m in models.items():
        t0 = time.time()
        m.fit(X_train, y_train)
        elapsed = time.time() - t0
        proba = m.predict_proba(X_test)[:, 1]
        results.append({
            "name": name,
            "model": m,
            "proba": proba,
            "fit_time": elapsed,
            "use_smote": use_smote,
        })
        print(f"  [{name}] fit_time={elapsed:.2f}s")
    return results


def get_best(results, y_test, metric: str = "pr_auc"):
    """根据指定指标选择最佳模型"""
    from evaluate import compute_metrics
    best = None
    for r in results:
        m = compute_metrics(y_test, r["proba"])
        r["metrics"] = m
        score = m[metric]
        if best is None or score > best["score"]:
            best = {"name": r["name"], "model": r["model"], "score": score, "proba": r["proba"], "metrics": m}
    return best


if __name__ == "__main__":
    from data_loader import load_data
    from preprocess import feature_engineer, split_and_scale, apply_smote
    df = feature_engineer(load_data())
    X_train, X_test, y_train, y_test, _, _ = split_and_scale(df)

    print(">>> Strategy A: class_weight (no SMOTE)")
    res_a = train_all(X_train, y_train, X_test, use_smote=False)
    print(">>> Strategy B: SMOTE")
    X_res, y_res = apply_smote(X_train, y_train)
    res_b = train_all(X_res, y_res, X_test, use_smote=True)

    print("Done.")
