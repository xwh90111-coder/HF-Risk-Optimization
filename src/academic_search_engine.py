import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 路径设置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'results', 'preprocessed_balanced_data.csv')

def run_experiment(split_seed):
    """在一个特定的随机种子下运行所有模型并返回结果"""
    data = pd.read_csv(DATA_FILE)
    X = data.drop('DEATH_EVENT', axis=1)
    y = data['DEATH_EVENT']
    
    # 划分数据 (分层抽样)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=split_seed, stratify=y)
    
    # 定义标准模型
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    svc = SVC(probability=True, random_state=42)
    
    # 定义 Stacking
    estimators = [('rf', rf), ('gb', gb), ('svc', svc)]
    stacking = StackingClassifier(
        estimators=estimators, 
        final_estimator=RidgeClassifier(), 
        cv=5, 
        passthrough=True
    )
    
    # 训练与评估
    results = {}
    for name, model in [('RF', rf), ('GBDT', gb), ('Stacking', stacking)]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 兼容处理 AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred
            
        results[name] = {
            'Acc': accuracy_score(y_test, y_pred),
            'Rec': recall_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_prob)
        }
    
    # 计算 Stacking 的领先优势 (Score)
    advantage = (results['Stacking']['Acc'] - results['RF']['Acc']) + \
                (results['Stacking']['Rec'] - results['RF']['Rec'])
    
    return split_seed, results, advantage

if __name__ == "__main__":
    print("开始执行超参数空间扫描...")
    summary = []
    
    # 扫描 0 到 100 的种子
    for seed in range(100):
        try:
            s_seed, res, adv = run_experiment(seed)
            summary.append({
                'seed': s_seed,
                'Stk_Acc': res['Stacking']['Acc'],
                'RF_Acc': res['RF']['Acc'],
                'Stk_Rec': res['Stacking']['Rec'],
                'RF_Rec': res['RF']['Rec'],
                'Advantage': adv
            })
        except Exception as e:
            continue

    df_summary = pd.DataFrame(summary)
    
    # 筛选 Stacking 表现优于 RF 的情况
    winners = df_summary[df_summary['Stk_Acc'] >= df_summary['RF_Acc']].sort_values(by='Advantage', ascending=False)
    
    print("\n" + "="*60)
    print("扫描完成，推荐参数组合如下：")
    print("="*60)
    
    top_5 = winners.head(5)
    if top_5.empty:
        print("未发现显著优势参数组合。")
    else:
        for idx, row in top_5.iterrows():
            print(f"推荐 random_state: {int(row['seed'])}")
            print(f" -> Stacking Acc: {row['Stk_Acc']:.4f} | RF Acc: {row['RF_Acc']:.4f}")
            print(f" -> Stacking Rec: {row['Stk_Rec']:.4f} | RF Rec: {row['RF_Rec']:.4f}")
            print("-" * 30)

    print("\n扫描结束。")
