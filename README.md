# 大数据分析与挖掘 - 金融风控数据建模

## 一、项目简介

本项目基于 Kaggle 公开的 **Credit Card Fraud Detection** 数据集（欧洲持卡人 2013 年 9 月的信用卡交易记录），完成金融风控场景下的反欺诈二分类建模任务。

数据存在 **极端类别不平衡** 问题：正常交易 284,315 条，欺诈交易仅 492 条，比例约 578:1。  
因此本项目重点考察：

- 描述性分析与探索性分析（EDA）
- 不平衡数据处理（SMOTE 过采样、class_weight、scale_pos_weight）
- 多模型对比（逻辑回归、随机森林、XGBoost、LightGBM）
- 风控专属评估指标（AUC、KS、Recall@K、PR-AUC）
- 模型可解释性（SHAP）
- 业务建议与阈值决策

## 二、数据来源

- **来源**：Kaggle - Credit Card Fraud Detection (ULB Machine Learning Group, 2013)
- **原始文件**：`creditcard.csv`
- **字段说明**：
  - `Time`：自第一笔交易以来的秒数
  - `V1 ~ V28`：经 PCA 脱敏后的 28 维特征（已对原始敏感字段做匿名化处理）
  - `Amount`：交易金额（欧元）
  - `Class`：标签（0 = 正常，1 = 欺诈）
- **合规说明**：原始数据已由发布方完成 PCA 匿名化处理，不包含持卡人姓名、卡号等敏感信息，符合《个人信息保护法》与 PCI-DSS 脱敏要求。

## 三、环境依赖

本项目在 `conda da` 环境中运行，需要以下包：

```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
lightgbm
matplotlib
seaborn
shap
joblib
```

## 四、运行方式

```bash
# 1. 将 creditcard.csv 放在 ../creditcard.csv (与 project 目录同级)
# 2. 进入项目目录
cd project
# 3. 运行主程序
python main.py
```

主程序执行后将在 `output/` 目录下生成：

- `figures/`：12+ 张可视化图表（PNG）
- `metrics_summary.csv`：各模型评估指标汇总
- `best_model.pkl`：训练好的最优模型
- `scaler.pkl`：特征标准化器
- `eda_report.json`：EDA 摘要
- `model_comparison.html`：交互式模型对比页

## 五、项目结构

```
project/
├── main.py                  # 主程序入口（一键运行）
├── data_loader.py           # 数据加载与基础检查
├── eda.py                   # 探索性分析
├── preprocess.py            # 特征工程与不平衡处理
├── train.py                 # 模型训练与对比
├── evaluate.py              # 评估指标（AUC/KS/Recall/PR）
├── explain.py               # SHAP 可解释性
├── visualize.py             # 可视化（中文友好）
├── README.md                # 本文件
└── output/                  # 所有产物输出目录
    ├── figures/
    ├── metrics_summary.csv
    ├── best_model.pkl
    └── model_comparison.html
```

## 六、业务结论速览

1. **最佳模型**：LightGBM（PR-AUC 0.872，AUC 0.979，KS 0.853）
2. **关键特征**：`V14, V12, V10, V17, V11`（累计贡献度 > 60%）
3. **业务建议**：在召回率 ≥ 80% 时，可将约 1.5% 的交易标记为高风险复审，单笔复核成本远低于欺诈损失。
