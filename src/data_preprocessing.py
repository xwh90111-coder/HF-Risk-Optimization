import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
import pickle

# 设置中文字体（适配论文图表）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
data_path = 'archive/cardio_train.csv'
if not os.path.exists(data_path):
    data_path = 'archive/cardio_train.csv'

data = pd.read_csv(data_path, sep=';')

# 2. 数据清洗与特征工程
# 删除无用的 id 列
if 'id' in data.columns:
    data = data.drop('id', axis=1)

# 将年龄从天数转换为年
data['age'] = data['age'] / 365.25

# 构建新特征
# BMI = 体重(kg) / 身高(m)^2
data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)

# 脉压差 = 收缩压 - 舒张压
data['pulse_pressure'] = data['ap_hi'] - data['ap_lo']

print("特征工程完成，新增特征：bmi, pulse_pressure")

# 去除明显离谱的异常值（例如血压负数或超过300，身高低于50cm）
data = data[(data['ap_hi'] <= 300) & (data['ap_hi'] >= 50)]
data = data[(data['ap_lo'] <= 300) & (data['ap_lo'] >= 40)]
data = data[(data['height'] >= 100) & (data['weight'] >= 30)]

# 3. 数据标准化
scaler = StandardScaler()
numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure']
categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

X_numerical = data[numerical_features]
y = data['cardio']
X_categorical = data[categorical_features]

X_scaled = pd.DataFrame(scaler.fit_transform(X_numerical), columns=numerical_features)
data_preprocessed = pd.concat([X_scaled.reset_index(drop=True), 
                               X_categorical.reset_index(drop=True), 
                               y.reset_index(drop=True)], axis=1)

# 保存常规预处理数据
os.makedirs('results', exist_ok=True)
data_preprocessed.to_csv('results/preprocessed_heart_failure_data.csv', index=False)
print("常规预处理数据已保存。")

# 保存 scaler
with open('results/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("标准化模型已保存至 results/scaler.pkl")

# 4. 类别平衡处理 (SMOTE)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(data_preprocessed.drop('cardio', axis=1), data_preprocessed['cardio'])

data_balanced = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled, name='cardio')], axis=1)

# 保存 SMOTE 平衡后的数据
data_balanced.to_csv('results/preprocessed_balanced_data.csv', index=False)
print(f"SMOTE 平衡完成。原始样本数: {len(data)}, 平衡后样本数: {len(data_balanced)}")

# 5. 可视化对比（仅展示平衡效果）
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='cardio', data=data_preprocessed)
plt.title('原始类别分布')

plt.subplot(1, 2, 2)
sns.countplot(x='cardio', data=data_balanced)
plt.title('SMOTE 平衡后分布')

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/sampling_comparison.png')
print("采样对比图已保存至 figures/sampling_comparison.png")
