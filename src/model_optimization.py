import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve, auc, confusion_matrix)
import os
import pickle
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 路径自适应逻辑
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DATA_FILE = os.path.join(RESULTS_DIR, 'preprocessed_balanced_data.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. 加载数据
data = pd.read_csv(DATA_FILE)
X = data.drop('cardio', axis=1)
y = data['cardio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8, stratify=y)

print("开始模型训练与评估...")

def evaluate_model_comprehensive(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred
    return {
        '准确率(Acc)': accuracy_score(y_test, y_pred),
        '精确率(Pre)': precision_score(y_test, y_pred),
        '召回率(Rec)': recall_score(y_test, y_pred),
        'F1值': f1_score(y_test, y_pred),
        'AUC值': roc_auc_score(y_test, y_prob)
    }

# 3. 稳健基模型配置
rf_base = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
gb_base = GradientBoostingClassifier(random_state=42)
svc_base = SVC(probability=True, random_state=42, max_iter=1000)

# 4. 构建 Stacking (元学习器采用 RidgeClassifier)
estimators = [('rf', rf_base), ('gb', gb_base), ('svc', svc_base)]
stacking_model = StackingClassifier(
    estimators=estimators, 
    final_estimator=RidgeClassifier(), 
    cv=5, 
    passthrough=True
)
stacking_model.fit(X_train, y_train)

# 5. 训练对比基准
lr_baseline = LogisticRegression(max_iter=1000, random_state=1).fit(X_train, y_train)
rf_base.fit(X_train, y_train)
gb_base.fit(X_train, y_train)

# 6. 汇总真实数据
results = {
    '基准模型(LR)': evaluate_model_comprehensive(lr_baseline, X_test, y_test),
    '随机森林(RF)': evaluate_model_comprehensive(rf_base, X_test, y_test),
    '梯度提升树(GBDT)': evaluate_model_comprehensive(gb_base, X_test, y_test),
    'Stacking(本文方案)': evaluate_model_comprehensive(stacking_model, X_test, y_test)
}
df_res = pd.DataFrame(results).T

# 7. 绘图优化：柱状图
plt.figure(figsize=(15, 9))
ax = df_res.plot(kind='bar', rot=0, width=0.82, ax=plt.gca(),
                 color=['#3366CC', '#109618', '#FF9900', '#DC3912', '#990099'])

# Y轴起点自适应
all_values = df_res.values.flatten()
min_val = np.min(all_values)
plt.ylim(max(0.6, min_val - 0.08), 1.05) 

# 添加数值标注
for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(f'{h:.3f}', (p.get_x() + p.get_width() / 2., h), 
                    ha='center', va='center', xytext=(0, 10), 
                    textcoords='offset points', fontsize=8, fontweight='bold')

plt.title('心力衰竭风险预测模型多维度性能横向对比', fontsize=14, pad=20)
plt.ylabel('Score')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='性能指标')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()

# 保存柱状图
plt.savefig(os.path.join(FIGURES_DIR, 'academic_model_comparison.png'), dpi=300)
plt.close()

# 8. 补充绘图：ROC 曲线对比
models_dict = {
    '逻辑回归(LR)': lr_baseline,
    '随机森林(RF)': rf_base,
    '梯度提升树(GBDT)': gb_base,
    'Stacking(本文方案)': stacking_model
}

plt.figure(figsize=(10, 8))
for name, model in models_dict.items():
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = model.predict(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)
plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)
plt.title('各算法受试者工作特征(ROC)曲线对比', fontsize=14)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'comparison_roc_curves.png'), dpi=300)
plt.close()

# 9. 补充绘图：混淆矩阵对比
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for idx, (name, model) in enumerate(models_dict.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx], annot_kws={"size": 14})
    axes[idx].set_title(name, fontsize=14)
    axes[idx].set_xlabel('预测值 (Predicted)', fontsize=12)
    if idx == 0:
        axes[idx].set_ylabel('真实值 (Actual)', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'comparison_confusion_matrices.png'), dpi=300)
plt.close()

df_res.to_csv(os.path.join(RESULTS_DIR, 'model_metrics_comparison.csv'))
with open(os.path.join(RESULTS_DIR, 'best_stacking_model.pkl'), 'wb') as f:
    pickle.dump(stacking_model, f)

print("\n模型性能评估结果：")
print(df_res.round(4))
print("\n图表与模型已保存。")
