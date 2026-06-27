"""
生成交互式 HTML demo 页面
- 展示模型对比表
- 嵌入关键图表
- 支持客户在浏览器中查看
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "output"
OVERVIEW = OUT / "overview.json"
FIG = OUT / "figures"


def img_b64(path: Path) -> str:
    """图片转 base64 嵌入 HTML（保证 self-contained）"""
    import base64
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def build():
    if not OVERVIEW.exists():
        print(f"[WARN] {OVERVIEW} not found. Run main.py first.")
        return
    with open(OVERVIEW, "r", encoding="utf-8") as f:
        ov = json.load(f)

    data_info = ov.get("data_info", {})
    metrics = ov.get("metrics", [])
    best = ov.get("best", {})

    # 读取图像
    imgs = {}
    for name in [
        "01_class_distribution.png",
        "04_correlation_top20.png",
        "10_roc_compare.png",
        "11_pr_compare.png",
        "12_ks_compare.png",
        "13_model_metrics_bar.png",
        "16_proba_dist.png",
        "08_shap_bar.png",
        "09_shap_summary.png",
    ]:
        imgs[name] = img_b64(FIG / name)

    # 模型表行
    rows_html = ""
    for m in metrics:
        rows_html += f"""
        <tr>
            <td><b>{m['model']}</b></td>
            <td>{m['strategy']}</td>
            <td>{m['auc']:.4f}</td>
            <td>{m['pr_auc']:.4f}</td>
            <td>{m['ks']:.4f}</td>
            <td>{m['f1@0.5']:.4f}</td>
            <td>{m['recall@1%']*100:.1f}%</td>
            <td>{m['recall@5%']*100:.1f}%</td>
            <td>{m['fit_time_s']:.1f}s</td>
        </tr>
        """

    # SHAP Top
    shap_html = ""
    for i, (fname, val) in enumerate(best.get("shap_top", [])[:10], 1):
        shap_html += f"<tr><td>{i}</td><td>{fname}</td><td>{val:.4f}</td></tr>"

    # 关键指标卡
    bm = best.get("metrics", {}) if best else {}
    kpi_cards = f"""
    <div class="kpi"><div class="kpi-label">PR-AUC</div><div class="kpi-value">{bm.get('pr_auc', 0):.4f}</div></div>
    <div class="kpi"><div class="kpi-label">AUC</div><div class="kpi-value">{bm.get('auc', 0):.4f}</div></div>
    <div class="kpi"><div class="kpi-label">KS</div><div class="kpi-value">{bm.get('ks', 0):.4f}</div></div>
    <div class="kpi"><div class="kpi-label">Recall@1%</div><div class="kpi-value">{bm.get('recall@1%', 0)*100:.1f}%</div></div>
    <div class="kpi"><div class="kpi-label">Recall@5%</div><div class="kpi-value">{bm.get('recall@5%', 0)*100:.1f}%</div></div>
    """

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>金融风控数据建模 - Demo</title>
<style>
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    margin: 0; padding: 24px; background: #f6f8fa; color: #24292e;
}}
h1 {{ margin: 0 0 4px 0; font-size: 26px; }}
h2 {{ margin: 24px 0 12px; padding-left: 10px; border-left: 4px solid #4C72B0; font-size: 18px; }}
.subtitle {{ color: #6a737d; margin-bottom: 20px; font-size: 13px; }}
.kpi-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
.kpi {{ flex: 1; min-width: 140px; background: #fff; border: 1px solid #e1e4e8;
       border-radius: 8px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
.kpi-label {{ font-size: 12px; color: #6a737d; margin-bottom: 4px; }}
.kpi-value {{ font-size: 22px; font-weight: 600; color: #4C72B0; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 16px; }}
.card {{ background: #fff; border: 1px solid #e1e4e8; border-radius: 8px;
         padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
.card img {{ width: 100%; height: auto; border-radius: 4px; }}
.card-title {{ font-weight: 600; font-size: 14px; margin-bottom: 8px; color: #24292e; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; font-size: 13px;
         box-shadow: 0 1px 3px rgba(0,0,0,0.05); border-radius: 8px; overflow: hidden; }}
th, td {{ padding: 8px 12px; text-align: center; border-bottom: 1px solid #e1e4e8; }}
th {{ background: #f6f8fa; font-weight: 600; color: #24292e; }}
tr:hover {{ background: #f6f8fa; }}
.tag {{ display: inline-block; padding: 2px 8px; background: #4C72B0; color: #fff;
        border-radius: 10px; font-size: 12px; }}
.info {{ background: #fff; padding: 12px 16px; border-radius: 8px; border: 1px solid #e1e4e8;
         margin-bottom: 16px; font-size: 13px; line-height: 1.6; }}
</style>
</head>
<body>
<h1>金融风控数据建模 - 信用卡欺诈检测</h1>
<div class="subtitle">《大数据分析与挖掘》课程大作业 | 数据：Kaggle Credit Card Fraud Detection | 最佳模型：<span class="tag">{best.get('name', '-')} ({best.get('strategy', '-')})</span></div>

<div class="info">
<b>数据集概要：</b>
样本数 {data_info.get('n_samples', '-')} | 特征数 {data_info.get('n_features', '-')} |
正常 {data_info.get('n_normal', '-')} | 欺诈 {data_info.get('n_fraud', '-')} |
欺诈占比 {data_info.get('fraud_ratio', 0)*100:.4f}% | 缺失值 {data_info.get('missing_total', '-')}
</div>

<h2>最佳模型核心指标</h2>
<div class="kpi-row">{kpi_cards}</div>

<h2>模型对比（4 模型 × 2 策略）</h2>
<table>
    <thead><tr>
        <th>模型</th><th>策略</th><th>AUC</th><th>PR-AUC</th><th>KS</th>
        <th>F1@0.5</th><th>Recall@1%</th><th>Recall@5%</th><th>训练耗时</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
</table>

<h2>关键可视化</h2>
<div class="cards">
    <div class="card">
        <div class="card-title">类别分布（极端不平衡）</div>
        <img src="data:image/png;base64,{imgs.get('01_class_distribution.png', '')}">
    </div>
    <div class="card">
        <div class="card-title">Top 20 相关特征</div>
        <img src="data:image/png;base64,{imgs.get('04_correlation_top20.png', '')}">
    </div>
    <div class="card">
        <div class="card-title">模型对比柱图</div>
        <img src="data:image/png;base64,{imgs.get('13_model_metrics_bar.png', '')}">
    </div>
    <div class="card">
        <div class="card-title">ROC 曲线</div>
        <img src="data:image/png;base64,{imgs.get('10_roc_compare.png', '')}">
    </div>
    <div class="card">
        <div class="card-title">PR 曲线</div>
        <img src="data:image/png;base64,{imgs.get('11_pr_compare.png', '')}">
    </div>
    <div class="card">
        <div class="card-title">KS 曲线</div>
        <img src="data:image/png;base64,{imgs.get('12_ks_compare.png', '')}">
    </div>
    <div class="card">
        <div class="card-title">预测概率分布</div>
        <img src="data:image/png;base64,{imgs.get('16_proba_dist.png', '')}">
    </div>
    <div class="card">
        <div class="card-title">SHAP 特征重要性</div>
        <img src="data:image/png;base64,{imgs.get('08_shap_bar.png', '')}">
    </div>
    <div class="card">
        <div class="card-title">SHAP 摘要图</div>
        <img src="data:image/png;base64,{imgs.get('09_shap_summary.png', '')}">
    </div>
</div>

<h2>SHAP 全局 Top 10 重要特征</h2>
<table>
    <thead><tr><th>排名</th><th>特征名</th><th>平均 |SHAP| 值</th></tr></thead>
    <tbody>{shap_html}</tbody>
</table>

<div class="info" style="margin-top:24px">
<b>业务建议：</b>在保持 Recall≥80% 的前提下，将高风险拦截比例控制在 1%~2% 之间，建议以模型概率 + 规则引擎融合方式上线，并配套 PSI/KS 漂移监控。
</div>
</body>
</html>"""

    out_path = OUT / "demo.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] Demo HTML saved: {out_path}")


if __name__ == "__main__":
    build()
