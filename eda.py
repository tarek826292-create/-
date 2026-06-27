"""
探索性分析 (EDA)
- 类别分布
- 金额 / 时间分布
- 相关性分析
- 分组统计
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from data_loader import load_data

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).parent / "output" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def class_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    cnt = df["Class"].value_counts()
    bars = ax.bar(["Normal (0)", "Fraud (1)"], cnt.values, color=["#4C72B0", "#C44E52"])
    for b, v in zip(bars, cnt.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2000, f"{v:,}", ha="center", fontsize=10)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()
    p = OUT_DIR / "01_class_distribution.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


def amount_distribution(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # 正常
    axes[0].hist(df.loc[df["Class"] == 0, "Amount"], bins=50, color="#4C72B0", alpha=0.7)
    axes[0].set_title("Amount - Normal")
    axes[0].set_xlabel("Amount")
    axes[0].set_yscale("log")
    # 欺诈
    axes[1].hist(df.loc[df["Class"] == 1, "Amount"], bins=50, color="#C44E52", alpha=0.7)
    axes[1].set_title("Amount - Fraud")
    axes[1].set_xlabel("Amount")
    axes[1].set_yscale("log")
    plt.tight_layout()
    p = OUT_DIR / "02_amount_distribution.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


def time_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.loc[df["Class"] == 0, "Time"].values, df.loc[df["Class"] == 0, "Amount"].values,
            ".", color="#4C72B0", alpha=0.3, markersize=1, label="Normal")
    ax.plot(df.loc[df["Class"] == 1, "Time"].values, df.loc[df["Class"] == 1, "Amount"].values,
            ".", color="#C44E52", alpha=0.8, markersize=3, label="Fraud")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amount")
    ax.set_title("Transactions over Time")
    ax.legend()
    plt.tight_layout()
    p = OUT_DIR / "03_time_distribution.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


def correlation_heatmap(df: pd.DataFrame) -> Path:
    """与 Class 相关性 top 20"""
    corr = df.corr()["Class"].drop("Class").sort_values(key=abs, ascending=False)
    top20 = corr.head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in top20.values]
    ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1])
    ax.set_xlabel("Correlation with Class")
    ax.set_title("Top 20 Features by |Correlation|")
    plt.tight_layout()
    p = OUT_DIR / "04_correlation_top20.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


def fraud_by_hour(df: pd.DataFrame) -> Path:
    """按 6 小时分桶统计欺诈率"""
    df = df.copy()
    df["hour"] = (df["Time"] // 21600).astype(int)  # 21600s = 6h
    grp = df.groupby("hour")["Class"].agg(["count", "sum", "mean"])
    grp["fraud_rate"] = grp["mean"] * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax2 = ax.twinx()
    ax.bar(grp.index, grp["count"], color="#4C72B0", alpha=0.6, label="Total Count")
    ax2.plot(grp.index, grp["fraud_rate"], "o-", color="#C44E52", label="Fraud Rate (%)")
    ax.set_xlabel("6h Bucket (0~96h, about 2 days)")
    ax.set_ylabel("Transaction Count")
    ax2.set_ylabel("Fraud Rate (%)")
    ax.set_title("Fraud Rate over Time Buckets")
    plt.tight_layout()
    p = OUT_DIR / "05_fraud_by_hour.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


def amount_boxplot(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    sub = df[["Class", "Amount"]].copy()
    sub["Class"] = sub["Class"].map({0: "Normal", 1: "Fraud"})
    sns.boxplot(data=sub, x="Class", y="Amount", ax=ax, palette=["#4C72B0", "#C44E52"])
    ax.set_yscale("log")
    ax.set_title("Amount by Class (log scale)")
    plt.tight_layout()
    p = OUT_DIR / "06_amount_boxplot.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


def v_features_violin(df: pd.DataFrame) -> Path:
    """Top 4 相关性 V 特征小提琴图"""
    corr = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False)
    top4 = corr.head(4).index.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, c in enumerate(top4):
        sns.violinplot(data=df, x="Class", y=c, ax=axes[i], palette=["#4C72B0", "#C44E52"], inner="quartile")
        axes[i].set_title(f"{c}")
    plt.suptitle("Top 4 V Features by Class", fontsize=14)
    plt.tight_layout()
    p = OUT_DIR / "07_v_features_violin.png"
    plt.savefig(p, dpi=120)
    plt.close()
    return p


def eda_summary(df: pd.DataFrame) -> dict:
    """供报告使用的统计摘要"""
    fraud = df[df["Class"] == 1]
    normal = df[df["Class"] == 0]
    summary = {
        "normal_mean_amount": float(normal["Amount"].mean()),
        "fraud_mean_amount": float(fraud["Amount"].mean()),
        "normal_median_amount": float(normal["Amount"].median()),
        "fraud_median_amount": float(fraud["Amount"].median()),
        "normal_zero_amount_ratio": float((normal["Amount"] == 0).mean()),
        "fraud_zero_amount_ratio": float((fraud["Amount"] == 0).mean()),
    }
    corr = df.corr()["Class"].drop("Class").sort_values(key=abs, ascending=False)
    summary["top5_corr_features"] = corr.head(5).to_dict()
    return summary


def run_eda() -> dict:
    df = load_data()
    paths = [
        class_distribution(df),
        amount_distribution(df),
        time_distribution(df),
        correlation_heatmap(df),
        fraud_by_hour(df),
        amount_boxplot(df),
        v_features_violin(df),
    ]
    summary = eda_summary(df)
    summary["figures"] = [str(p) for p in paths]
    out_json = Path(__file__).parent / "output" / "eda_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[EDA] Generated {len(paths)} figures.")
    return summary


if __name__ == "__main__":
    s = run_eda()
    print(json.dumps(s, ensure_ascii=False, indent=2))
