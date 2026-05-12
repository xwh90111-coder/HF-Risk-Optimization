import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import os
import warnings

# 屏蔽冗余警告
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据和模型
data_path = '../results/preprocessed_balanced_data.csv'
if not os.path.exists(data_path):
    data_path = 'results/preprocessed_balanced_data.csv'

model_path = '../results/best_stacking_model.pkl'
if not os.path.exists(model_path):
    model_path = 'results/best_stacking_model.pkl'

data = pd.read_csv(data_path)
X = data.drop('DEATH_EVENT', axis=1)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

print("--- [学术补强] 开始深度可解释性 SHAP 分析 ---")

# 2. 初始化 SHAP 解释器
background = shap.sample(X, 100)
predict_fn = getattr(model, "decision_function", model.predict)
explainer = shap.KernelExplainer(predict_fn, background)

# 解释测试样本
test_sample = shap.sample(X, 50)
shap_values = explainer.shap_values(test_sample)

if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[1]
    shap_values_single = shap_values[1][0, :]
else:
    shap_values_to_plot = shap_values
    shap_values_single = shap_values[0, :]

# 映射简短名称
feature_mapping = {
    'age': 'Age', 'creatinine_phosphokinase': 'CPK', 'ejection_fraction': 'EF',
    'platelets': 'Platelets', 'serum_creatinine': 'Cr', 'serum_sodium': 'Na',
    'time': 'Time', 'age_creatinine': 'Age*Cr', 'ef_hbp': 'EF*HBP',
    'anaemia': 'Anaemia', 'diabetes': 'Diabetes', 'high_blood_pressure': 'HBP',
    'sex': 'Sex', 'smoking': 'Smoking'
}
test_sample_renamed = test_sample.rename(columns=feature_mapping)

# 3. 核心可视化方案
# (1) Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_to_plot, test_sample_renamed, show=False)
plt.title('心力衰竭风险预测：特征全局贡献度 (SHAP Summary)', pad=20)
plt.savefig('../figures/shap_summary_academic.png', bbox_inches='tight')
plt.close()

# (2) Dependence Plot
plt.figure(figsize=(10, 6))
shap.dependence_plot("Age", shap_values_to_plot, test_sample_renamed, 
                     interaction_index="Cr", show=False)
plt.title('年龄(Age)与血清肌酐(Cr)的交互效应分析', pad=20)
plt.savefig('../figures/shap_dependence_interaction.png', bbox_inches='tight')
plt.close()

# (3) Force Plot (最终美化版：斜向标题+精准对齐)
if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
    expected_value = explainer.expected_value[1]
else:
    expected_value = explainer.expected_value

# 核心优化逻辑：
# - text_rotation=45: 采用 45 度斜向排列，符合审美
# - contribution_threshold=0.05: 隐藏非核心特征，确保引用线有足够的物理空间保持垂直对齐
# - figsize=(20, 3): 优化纵横比
plt.figure(figsize=(20, 3))
shap.force_plot(expected_value, shap_values_single, test_sample_renamed.iloc[0, :], 
                matplotlib=True, show=False, figsize=(20, 3), 
                text_rotation=45, contribution_threshold=0.05)

# 调整标题位置
plt.title('典型病例风险因子剖析图 (Force Plot)', pad=50, fontsize=14)
plt.savefig('../figures/shap_individual_force_plot.png', bbox_inches='tight')
plt.close()

print("\nSHAP 图表已生成并保存。")
