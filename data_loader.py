"""
数据加载与基础信息检查
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "creditcard.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """加载原始 CSV"""
    df = pd.read_csv(path)
    return df


def basic_info(df: pd.DataFrame) -> dict:
    """返回数据集基础信息（用于日志和报告）"""
    info = {
        "n_samples": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        "n_fraud": int(df["Class"].sum()),
        "n_normal": int((df["Class"] == 0).sum()),
        "fraud_ratio": float(df["Class"].mean()),
        "missing_total": int(df.isnull().sum().sum()),
        "amount_min": float(df["Amount"].min()),
        "amount_max": float(df["Amount"].max()),
        "amount_mean": float(df["Amount"].mean()),
        "time_min": float(df["Time"].min()),
        "time_max": float(df["Time"].max()),
    }
    return info


if __name__ == "__main__":
    df = load_data()
    info = basic_info(df)
    for k, v in info.items():
        print(f"{k}: {v}")
