# 基于机器学习的心力衰竭风险评估与可解释性分析研究

本项目旨在通过数据挖掘与机器学习技术，对心力衰竭患者的临床指标进行深度分析，构建高精度的生存预测模型，并利用 SHAP (SHapley Additive exPlanations) 提供模型的可解释性分析。本项目从实际应用落地的“黑盒”应用，转向聚焦于**特征工程、Stacking 集成优化、以及临床医学可解释性**的学术研究探讨。

---

## 🔬 研究重点

- **数据不平衡处理**: 针对医疗数据中常见的正负样本不平衡问题，采用 SMOTE 等过采样技术进行类别平衡与特征预处理。
- **多模型对比分析**: 横向对比逻辑回归(LR)、随机森林(RF)、支持向量机(SVM)、梯度提升(GBDT)等多种基础基线模型的性能表现（Accuracy, Precision, Recall, F1, AUC）。
- **Stacking 集成学习优化**: 构建以 GBDT、RF、SVM 为基模型，Ridge Classifier 为元模型的 Stacking 拓扑架构，进一步突破单一模型的性能瓶颈。
- **可解释性分析 (XAI)**: 引入 SHAP 归因理论，打破机器学习“黑盒”。通过全局重要性图、局部依赖图和个体风险瀑布图，为临床医学决策提供透明、可靠的辅助依据。

---

## 📂 项目结构

```text
HF-Risk-Optimization/
├── README.md           # 项目说明文件
├── requirements.txt    # Python 实验依赖环境
├── data/               # 存放原始心力衰竭临床数据集
├── docs/               # 存放项目相关文档
├── figures/            # 存放生成的学术图表 (ROC曲线, SHAP分析图, 模型拓扑等)
├── results/            # 存放预处理后的数据集及评估指标 CSV 文件
└── src/                # 核心算法脚本与可视化代码
    ├── data_preprocessing.py            # 数据清洗、类别平衡与特征预处理
    ├── logistic_regression_analysis.py  # 逻辑回归基线模型的独立分析
    ├── model_comparison.py              # 多种基础模型(RF, SVM等)的性能横向对比
    ├── model_optimization.py            # 高级模型调优与Stacking集成学习实现
    ├── explainability_analysis.py       # 基于SHAP的模型特征重要性与可解释性分析
    ├── draw_closed_loop_system.py       # 绘制系统闭环架构流程图
    └── draw_stacking_topology_v2.py     # 绘制Stacking集成拓扑结构图
```

---

## 🛠️ 技术栈与依赖

本项目基于 Python 开发，核心科学计算与分析库如下：
- **数据处理**: `pandas`, `numpy`, `imbalanced-learn`
- **算法模型**: `scikit-learn`, `xgboost`, `scipy`
- **数据可视化与解释**: `matplotlib`, `seaborn`, `shap`

---

## 🚀 实验复现步骤

请遵循以下步骤在本地环境中复现本项目的研究结果：

### 1. 环境准备

```bash
# 克隆项目仓库
git clone https://github.com/xwh90111-coder/HF-Risk-Optimization.git
cd HF-Risk-Optimization

# 安装实验所需依赖 (建议在虚拟环境中运行)
pip install -r requirements.txt
```

### 2. 运行实验脚本

项目的各个分析模块高度解耦，建议按照以下数据流向依次运行：

- **Step 1: 数据清洗与预处理**  
  进行缺失值检查、数据标准化以及使用 SMOTE 进行样本平衡处理。预处理后的数据将输出至 `results/` 目录。
  ```bash
  python src/data_preprocessing.py
  ```

- **Step 2: 基线模型与多模型对比**  
  训练逻辑回归等多个基础模型，输出混淆矩阵、ROC 曲线等对比图表至 `figures/`。
  ```bash
  python src/logistic_regression_analysis.py
  python src/model_comparison.py
  ```

- **Step 3: Stacking 模型优化**  
  运行集成学习策略，寻找最优模型组合，并将对比指标保存。
  ```bash
  python src/model_optimization.py
  ```

- **Step 4: SHAP 可解释性分析**  
  对表现最优的模型提取特征贡献度，生成特征交互依赖图与个体病例的力导向图 (Force Plot)。
  ```bash
  python src/explainability_analysis.py
  ```

---

## ⚠️ 学术声明

本项目产生的所有代码、分析结果及可视化图表仅供**学术研究、算法探讨与技术交流**使用。模型输出的风险概率不具备临床诊断效力，真实医疗场景下的决策请务必遵循专业医师的指导。