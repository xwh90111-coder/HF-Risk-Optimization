import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import os

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建图片保存目录
os.makedirs('../figures', exist_ok=True)

# 读取预处理后的数据
data_path = '../results/preprocessed_heart_failure_data.csv'
if not os.path.exists(data_path):
    data_path = 'results/preprocessed_heart_failure_data.csv'
data = pd.read_csv(data_path)

# 特征和目标变量
X = data.drop('cardio', axis=1)
y = data['cardio']

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练与评估
# 逻辑回归
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)
y_prob_logistic = logistic_model.predict_proba(X_test)[:, 1]

# 随机森林
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# 支持向量机
svm_model = SVC(probability=True, random_state=42, max_iter=1000)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]

# 梯度提升
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
y_prob_gb = gb_model.predict_proba(X_test)[:, 1]

# K近邻
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
y_prob_knn = knn_model.predict_proba(X_test)[:, 1]

# 特征重要性分析
def feature_importance_analysis(model, model_name, X):
    if model_name == "逻辑回归":
        importances = pd.Series(model.coef_[0], index=X.columns)
    elif model_name == "随机森林":
        importances = pd.Series(model.feature_importances_, index=X.columns)
    elif model_name == "梯度提升":
        importances = pd.Series(model.feature_importances_, index=X.columns)
    else:
        importances = None
    return importances

models = {
    "逻辑回归": logistic_model,
    "随机森林": rf_model,
    "梯度提升": gb_model
}

feature_importances = {}
for model_name, model in models.items():
    importances = feature_importance_analysis(model, model_name, X)
    if importances is not None:
        importances = importances.sort_values(ascending=False)
        feature_importances[model_name] = importances

# 1. 特征重要性可视化
plt.figure(figsize=(15, 12))
for i, (model_name, importances) in enumerate(feature_importances.items(), 1):
    plt.subplot(3, 1, i)
    importances.plot(kind='bar')
    plt.title(f'特征重要性分析 ({model_name})')
plt.tight_layout()
plt.savefig('../figures/comparison_feature_importance.png')
plt.close()

# 2. ROC曲线对比
def plot_roc_curve(name, y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

fpr_logistic, tpr_logistic, roc_auc_logistic = plot_roc_curve("逻辑回归", y_test, y_prob_logistic)
fpr_rf, tpr_rf, roc_auc_rf = plot_roc_curve("随机森林", y_test, y_prob_rf)
fpr_svm, tpr_svm, roc_auc_svm = plot_roc_curve("支持向量机", y_test, y_prob_svm)
fpr_gb, tpr_gb, roc_auc_gb = plot_roc_curve("梯度提升", y_test, y_prob_gb)
fpr_knn, tpr_knn, roc_auc_knn = plot_roc_curve("K近邻", y_test, y_prob_knn)

plt.figure(figsize=(10, 8))
plt.plot(fpr_logistic, tpr_logistic, label=f'逻辑回归 (AUC = {roc_auc_logistic:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'随机森林 (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_svm, tpr_svm, label=f'支持向量机 (AUC = {roc_auc_svm:.2f})')
plt.plot(fpr_gb, tpr_gb, label=f'梯度提升 (AUC = {roc_auc_gb:.2f})')
plt.plot(fpr_knn, tpr_knn, label=f'K近邻 (AUC = {roc_auc_knn:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('多模型 ROC 曲线对比')
plt.legend()
plt.savefig('../figures/comparison_roc_curves.png')
plt.close()

# 3. 混淆矩阵对比
def plot_confusion_matrix_save(name, conf_matrix, i):
    plt.subplot(1, 5, i+1)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{name}')

plt.figure(figsize=(25, 5))
plot_confusion_matrix_save("逻辑回归", confusion_matrix(y_test, y_pred_logistic), 0)
plot_confusion_matrix_save("随机森林", confusion_matrix(y_test, y_pred_rf), 1)
plot_confusion_matrix_save("支持向量机", confusion_matrix(y_test, y_pred_svm), 2)
plot_confusion_matrix_save("梯度提升", confusion_matrix(y_test, y_pred_gb), 3)
plot_confusion_matrix_save("K近邻", confusion_matrix(y_test, y_pred_knn), 4)
plt.tight_layout()
plt.savefig('../figures/comparison_confusion_matrices.png')
plt.close()

print("模型对比分析完成，所有图片已保存至 figures/ 目录。")
