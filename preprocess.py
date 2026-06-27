"""
特征工程与不平衡处理
- Time / Amount 标准化
- 衍生特征：Hour-of-day, amount_log, is_zero_amount
- 三种不平衡策略：原比例 / class_weight / SMOTE
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """构造衍生特征"""
    df = df.copy()
    # 1. 时间处理：第二笔交易开始的小时数 + 派生 day, night
    seconds_per_day = 86400
    df["Hour"] = (df["Time"] % seconds_per_day) // 3600
    df["Day"] = (df["Time"] // seconds_per_day).astype(int)
    df["IsNight"] = ((df["Hour"] >= 0) & (df["Hour"] < 6)).astype(int)
    # 2. 金额处理
    df["AmountLog"] = np.log1p(df["Amount"])
    df["IsZeroAmount"] = (df["Amount"] == 0).astype(int)
    return df


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    返回:
        X_train, X_test, y_train, y_test, scaler
    """
    feat_cols = [c for c in df.columns if c != "Class"]
    X = df[feat_cols].values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, OUT_DIR / "scaler.pkl")
    return X_train, X_test, y_train, y_test, scaler, feat_cols


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """对训练集做 SMOTE 过采样"""
    sm = SMOTE(random_state=random_state, sampling_strategy="auto")
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res


if __name__ == "__main__":
    from data_loader import load_data, basic_info
    df = load_data()
    print("Before FE:", df.shape)
    df = feature_engineer(df)
    print("After FE:", df.shape)
    X_train, X_test, y_train, y_test, scaler, feat_cols = split_and_scale(df)
    print("Train:", X_train.shape, "Fraud in train:", int(y_train.sum()))
    X_res, y_res = apply_smote(X_train, y_train)
    print("After SMOTE:", X_res.shape, "Fraud:", int(y_res.sum()))
