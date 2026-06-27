"""
可视化集合：ROC 曲线、PR 曲线、KS 曲线、模型对比柱图、混淆矩阵、概率分布
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

OUT_DIR = Path(__file__).parent / "output" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_roc(results_dict: dict, y_test, save_name: str = "10_roc_compare.png"):
    """
    results_dict: {model_name: y_proba}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, proba in results_dict.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        from sklearn.metrics import auc
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Test Set)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = OUT_DIR / save_name
    plt.savefig(p, dpi=120)
    plt.close()
    return p


def plot_pr(results_dict: dict, y_test, save_name: str = "11_pr_compare.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    from sklearn.metrics import average_precision_score
    for name, proba in results_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.4f})")
    base = float(np.mean(y_test))
    ax.axhline(base, color="k", linestyle="--", alpha=0.4, label=f"Base={base:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = OUT_DIR / save_name
    plt.savefig(p, dpi=120)
    plt.close()
    return p


def plot_ks(results_dict: dict, y_test, save_name: str = "12_ks_compare.png"):
    """单图展示最佳模型的 KS"""
    best_name, best_proba = max(
        results_dict.items(),
        key=lambda kv: float(np.max([roc_curve(y_test, kv[1])[1] - roc_curve(y_test, kv[1])[0]]))
    )
    fpr, tpr, _ = roc_curve(y_test, best_proba)
    ks = float(np.max(tpr - fpr))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fpr, label="FPR")
    ax.plot(tpr, label="TPR")
    ax.fill_between(range(len(tpr)), tpr, fpr, alpha=0.2)
    ax.set_xlabel("Threshold index")
    ax.set_ylabel("Rate")
    ax.set_title(f"KS Curve ({best_name}, KS={ks:.4f})")
    ax.legend()
    plt.tight_layout()
    p = OUT_DIR / save_name
    plt.savefig(p, dpi=120)
    plt.close()
    return p


def plot_model_bar(metrics_df, save_name: str = "13_model_metrics_bar.png"):
    """模型对比柱图：AUC / PR-AUC / KS"""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics_df))
    width = 0.25
    metrics_df = metrics_df.copy()
    metrics_df["label"] = metrics_df["model"] + " (" + metrics_df["strategy"] + ")"
    ax.bar(x - width, metrics_df["auc"], width, label="AUC", color="#4C72B0")
    ax.bar(x, metrics_df["pr_auc"], width, label="PR-AUC", color="#55A868")
    ax.bar(x + width, metrics_df["ks"], width, label="KS", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["label"], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    p = OUT_DIR / save_name
    plt.savefig(p, dpi=120)
    plt.close()
    return p


def plot_confusion(y_true, y_proba, threshold: float, name: str, save_name: str):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {name} (t={threshold})")
    plt.tight_layout()
    p = OUT_DIR / save_name
    plt.savefig(p, dpi=120)
    plt.close()
    return p


def plot_proba_dist(y_true, y_proba, name: str, save_name: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(y_proba[y_true == 0], bins=60, alpha=0.6, color="#4C72B0", label="Normal", density=True)
    ax.hist(y_proba[y_true == 1], bins=60, alpha=0.6, color="#C44E52", label="Fraud", density=True)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Probability Distribution - {name}")
    ax.legend()
    plt.tight_layout()
    p = OUT_DIR / save_name
    plt.savefig(p, dpi=120)
    plt.close()
    return p
