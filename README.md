# 心衰竭预测系统

一个基于机器学习的Web应用，通过分析患者的临床指标，实时预测心力衰竭事件的风险。

---

## 项目结构

```text
HF-Risk-Optimization/
├── app.py              # Flask后端应用核心
├── README.md           # 项目说明文件
├── requirements.txt    # Python依赖列表
├── data/               # 存放原始数据集
├── docs/               # 存放项目文档 (.gitkeep用于保持空目录被git追踪)
├── figures/            # 存放各种模型分析和结果的可视化图表
├── results/            # 存放预处理后的数据和训练好的模型文件
├── src/                # 存放核心算法脚本与可视化代码
│   ├── data_preprocessing.py            # 数据清洗、类别平衡与特征预处理
│   ├── logistic_regression_analysis.py  # 逻辑回归基线模型的独立分析
│   ├── model_comparison.py              # 多种基础模型(RF, SVM等)的性能横向对比
│   ├── model_optimization.py            # 高级模型调优与Stacking集成学习实现
│   ├── explainability_analysis.py       # 基于SHAP的模型特征重要性与可解释性分析
│   ├── draw_closed_loop_system.py       # 绘制系统闭环架构流程图
│   └── draw_stacking_topology_v2.py     # 绘制Stacking集成拓扑结构图
└── web/                # 存放所有前端文件
    ├── index.html      # 前端页面
    ├── styles.css      # 页面样式
    ├── script.js       # 页面交互逻辑
    └── 前端使用说明.md # 前端专项说明
```

---

## 核心功能

- **直观的Web界面**: 用户可以通过一个美观、响应式的网页轻松输入12项临床指标。
- **多模型预测**: 集成了五种业界主流的机器学习模型（逻辑回归、随机森林、SVM、梯度提升、K-近邻），用户可自由选择。
- **即时风险评估**: 提交数据后，系统立即返回预测结果，包括风险等级（高/低风险）、风险概率以及主要风险因素分析。
- **全栈技术实现**: 项目涵盖了从数据分析、模型训练到Web后端API和前端界面开发的全过程。

## 技术栈

- **后端**: Python, Flask, Scikit-learn, Pandas, NumPy
- **前端**: HTML5, CSS3, JavaScript (Vanilla JS)
- **数据集**: UCI机器学习库

---

## 安装与运行

请遵循以下步骤来在您的本地环境中运行此项目。

### 1. 环境准备

- **克隆项目**:
  ```bash
  git clone https://github.com/xwh90111-coder/HF-Risk-Optimization.git
  cd HF-Risk-Optimization
  ```
- **安装依赖**: 建议在Python虚拟环境中执行。项目所需的全部依赖都已在 `requirements.txt` 中列出。
  ```bash
  pip install -r requirements.txt
  ```

### 2. 启动Web应用 

这是体验本项目的最佳方式。此命令将启动后端服务器，并提供完整的前端交互功能。

- **运行服务器**:
  ```bash
  python app.py
  ```
- **访问应用**: 服务器启动后，在您的浏览器中打开以下地址：
  ```
  http://localhost:5000
  ```

现在，您应该能看到项目的主界面，可以开始输入数据进行预测了。

### 3. (可选) 运行分析脚本

如果您想重现数据分析和模型评估的过程，可以运行 `src/` 目录下的脚本。请注意，`app.py` 在启动时会自动在内存中完成模型训练，因此运行这些脚本对于启动Web应用不是必需的。

- **数据预处理**:
  ```bash
  python src/data_preprocessing.py
  ```
- **模型对比分析**:
  ```bash
  python src/model_comparison.py
  ```

---


## API接口说明

应用后端提供了一个RESTful API用于预测，您也可以通过编程方式调用。

- **端点**: `POST /predict`
- **请求格式**: `application/json`
- **请求体示例**:

  ```json
  {
      "age": 75,
      "anaemia": 0,
      "creatinine_phosphokinase": 582,
      "diabetes": 0,
      "ejection_fraction": 20,
      "high_blood_pressure": 1,
      "platelets": 265000,
      "serum_creatinine": 1.9,
      "serum_sodium": 130,
      "sex": 1,
      "smoking": 0,
      "time": 4,
      "model": "random_forest"
  }
  ```

- **成功响应示例**:
  ```json
  {
    "model_used": "random_forest",
    "prediction": 1,
    "probability": 0.52,
    "risk_factors": [
      "高龄",
      "射血分数偏低",
      "血清肌酐升高",
      "高血压",
      "随访时间较短"
    ]
  }
  ```

---

## ⚠️ 重要声明

本系统是一个用于技术演示和学术研究的机器学习项目，**其预测结果不具备专业医疗诊断的资格**。任何与健康相关的问题，请务必咨询专业医生。
