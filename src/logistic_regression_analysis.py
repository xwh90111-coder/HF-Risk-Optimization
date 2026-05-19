import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图片保存目录
os.makedirs('../figures', exist_ok=True)

# 读取预处理后的数据
data_path = '../results/preprocessed_heart_failure_data.csv'
if not os.path.exists(data_path):
    data_path = 'results/preprocessed_heart_failure_data.csv'
data = pd.read_csv(data_path)

# 选择特征和目标变量
X = data.drop('cardio', axis=1)
y = data['cardio']

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 选择逻辑回归模型
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 1. 数值型变量的直方图
numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig('../figures/lr_numerical_distribution.png')
plt.close()

# 2. 分类变量的计数图
categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(2, 4, i)
    sns.countplot(x=feature, data=data)
    plt.title(f'Count of {feature}')
plt.tight_layout()
plt.savefig('../figures/lr_categorical_count.png')
plt.close()

# 3. 目标变量与其他变量的关系图 (箱线图)
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='cardio', y=feature, data=data)
    plt.title(f'{feature} by cardio')
plt.tight_layout()
plt.savefig('../figures/lr_boxplots.png')
plt.close()

# 4. 相关性热力图
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../figures/lr_correlation_heatmap.png')
plt.close()

# 5. 特征重要性分析图
feature_importances = pd.Series(model.coef_[0], index=X.columns)
plt.figure(figsize=(10, 6))
feature_importances.sort_values(ascending=False).plot(kind='bar')
plt.title('Feature Importances (Logistic Regression)')
plt.tight_layout()
plt.savefig('../figures/lr_feature_importances.png')
plt.close()

# 6. ROC曲线
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Logistic Regression)')
plt.legend()
plt.savefig('../figures/lr_roc_curve.png')
plt.close()

# 7. 混淆矩阵
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Logistic Regression)')
plt.savefig('../figures/lr_confusion_matrix.png')
plt.close()

print("逻辑回归分析完成，所有图片已保存至 figures/ 目录。")
