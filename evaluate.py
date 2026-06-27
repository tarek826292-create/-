"""
风控评估指标
- AUC (ROC)
- PR-AUC (Average Precision)
- KS 值（风控专用）
- Recall@K% / Precision@K%
- F1
- 业务指标：拦截率 (Recall) / 误报率 (FPR)
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, roc_curve, classification_report
)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ks = float(np.max(tpr - fpr))

    # Recall@K: 关注 top K% 评分能否捕获 K% 欺诈
    n = len(y_true)
    order = np.argsort(-y_proba)
    sorted_y = y_true[order]
    cum_fraud = np.cumsum(sorted_y)
    total_fraud = y_true.sum()
    recall_at_1pct = float(cum_fraud[int(0.01 * n)] / total_fraud) if total_fraud > 0 else 0.0
    recall_at_5pct = float(cum_fraud[int(0.05 * n)] / total_fraud) if total_fraud > 0 else 0.0

    # 基础 confusion
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    return {
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "ks": ks,
        "f1@0.5": float(f1),
        "recall@1%": recall_at_1pct,
        "recall@5%": recall_at_5pct,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def ks_statistic(y_true, y_proba) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def classification_report_str(y_true, y_proba, threshold: float = 0.5) -> str:
    y_pred = (y_proba >= threshold).astype(int)
    return classification_report(y_true, y_pred, digits=4, zero_division=0)


if __name__ == "__main__":
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_proba = np.array([0.1, 0.4, 0.7, 0.9, 0.3, 0.6, 0.2, 0.8])
    print(compute_metrics(y_true, y_proba))
