import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

# 设置中文字体（适配论文图表）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
data_path = '../data/heart_failure_clinical_records_dataset.csv'
if not os.path.exists(data_path):
    # 适配不同运行路径
    data_path = 'data/heart_failure_clinical_records_dataset.csv'

data = pd.read_csv(data_path)

# 2. 特征工程：构建医学交互特征 (Interaction Features)
# 学术理由：血清肌酐水平与年龄的乘积能更好地反映老年人的肾功能风险
data['age_creatinine'] = data['age'] * data['serum_creatinine']
# 射血分数与血压的交互
data['ef_hbp'] = data['ejection_fraction'] * data['high_blood_pressure'].astype(float)

print("特征工程完成，新增交互特征：age_creatinine, ef_hbp")

# 3. 数据标准化
scaler = StandardScaler()
numerical_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 
                      'serum_creatinine', 'serum_sodium', 'time', 'age_creatinine', 'ef_hbp']

# 先保存标签，再对数值特征进行缩放
X_numerical = data[numerical_features]
y = data['DEATH_EVENT']
X_categorical = data[['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']]

X_scaled = pd.DataFrame(scaler.fit_transform(X_numerical), columns=numerical_features)
data_preprocessed = pd.concat([X_scaled, X_categorical, y], axis=1)

# 保存常规预处理数据
os.makedirs('../results', exist_ok=True)
data_preprocessed.to_csv('../results/preprocessed_heart_failure_data.csv', index=False)
print("常规预处理数据已保存。")

import pickle
# 保存 scaler，确保 Web 端标准化参数与训练时完全一致
with open('../results/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("标准化模型已保存至 results/scaler.pkl")

# 4. 类别平衡处理 (SMOTE)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(data_preprocessed.drop('DEATH_EVENT', axis=1), y)

data_balanced = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled, name='DEATH_EVENT')], axis=1)

# 保存 SMOTE 平衡后的数据
data_balanced.to_csv('../results/preprocessed_balanced_data.csv', index=False)
print(f"SMOTE 平衡完成。原始样本数: {len(data)}, 平衡后样本数: {len(data_balanced)}")

# 5. 可视化对比（仅展示平衡效果）
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='DEATH_EVENT', data=data_preprocessed)
plt.title('原始类别分布')

plt.subplot(1, 2, 2)
sns.countplot(x='DEATH_EVENT', data=data_balanced)
plt.title('SMOTE 平衡后分布')

plt.tight_layout()
# 如果在服务器运行，建议保存为图片
plt.savefig('../figures/sampling_comparison.png')
print("采样对比图已保存至 figures/sampling_comparison.png")
