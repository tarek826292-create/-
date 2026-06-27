"""
SHAP 可解释性分析
- 训练后对最佳模型做 SHAP 分析
- 生成: 特征重要性 Bar / Summary Plot
- 关键点：XGBoost 1.7+ 的 sklearn 兼容层在训练 ndarray 时会推断
  feature_names_in_=['[5.003087E-1]', ...] 这种 cell 字符串，导致 SHAP
  TreeExplainer 在 path-dependent 模式下解析失败。
  解决方案：直接使用底层 booster 构造 explainer，绕过 sklearn 包装。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shap
import warnings

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).parent / "output" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_xgb_booster(model):
    """拿到 XGBoost 的底层 booster 句柄"""
    if hasattr(model, "get_booster"):
        return model.get_booster()
    if hasattr(model, "booster"):
        return model.booster()
    return None


def shap_analysis(model, X_sample: np.ndarray, feat_names: list, model_name: str = "best"):
    """
    对模型做 SHAP 分析。
    - X_sample: numpy.ndarray, shape (n, d)
    - feat_names: 长度 d 的特征名列表
    """
    X_df = pd.DataFrame(X_sample, columns=feat_names)
    cls_name = type(model).__name__

    # ---- XGBoost: 走底层 booster 接口 ----
    if cls_name == "XGBClassifier":
        booster = _to_xgb_booster(model)
        if booster is not None:
            # 用底层 booster，避免 sklearn 包装层引入的列名问题
            explainer = shap.TreeExplainer(booster)
            shap_values = explainer.shap_values(X_df)
            expected_value = float(explainer.expected_value)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
            expected_value = float(np.array(explainer.expected_value).flatten()[-1])
    # ---- LightGBM: 同样用 booster ----
    elif cls_name == "LGBMClassifier" or hasattr(model, "booster_"):
        if hasattr(model, "booster_"):
            explainer = shap.TreeExplainer(model.booster_)
            shap_values = explainer.shap_values(X_df)
            expected_value = float(np.array(explainer.expected_value).flatten()[-1])
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
            ev = explainer.expected_value
            expected_value = float(np.array(ev).flatten()[-1])
    # ---- RandomForest: 直接用 sklearn estimator，XGB/LGB 才需要绕路 ----
    elif cls_name == "RandomForestClassifier":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        ev = explainer.expected_value
        expected_value = float(np.array(ev).flatten()[-1])
    # ---- 线性模型 ----
    else:
        explainer = shap.LinearExplainer(model, X_df)
        shap_values = explainer.shap_values(X_df)
        expected_value = float(explainer.expected_value)

    # Bar plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X_df, feature_names=feat_names,
        plot_type="bar", show=False, max_display=15
    )
    plt.title(f"SHAP Feature Importance ({model_name})")
    p1 = OUT_DIR / "08_shap_bar.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close()

    # Beeswarm
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X_df, feature_names=feat_names,
        show=False, max_display=15
    )
    plt.title(f"SHAP Summary ({model_name})")
    p2 = OUT_DIR / "09_shap_summary.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs)
    top_features = [(feat_names[i], float(mean_abs[i])) for i in order[:10]]

    return {
        "figures": [str(p1), str(p2)],
        "top_features": top_features,
    }
