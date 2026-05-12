from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)

class HeartDiseasePredictor:
    def __init__(self):
        self.models = {}
        self.scaler = None
        # 定义基础数值特征和分类特征（顺序需与训练时严格一致）
        self.numerical_cols = [
            'age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 
            'serum_creatinine', 'serum_sodium', 'time'
        ]
        self.interaction_cols = ['age_creatinine', 'ef_hbp']
        self.categorical_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
        
        # 完整的特征列表（标准化后的顺序）
        self.all_features = self.numerical_cols + self.interaction_cols + self.categorical_cols
        
        self.load_resources()

    def load_resources(self):
        """加载训练好的模型和标准化器"""
        try:
            # 1. 尝试加载标准化器
            scaler_path = 'results/scaler.pkl'
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("成功加载标准化器 (scaler.pkl)")
            
            # 2. 尝试加载 Stacking 模型
            model_path = 'results/best_stacking_model.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models['stacking'] = pickle.load(f)
                print("成功加载 Stacking 融合模型")
            
            # 3. 如果没有保存好的模型，则进行基础训练（保底逻辑）
            if not self.models:
                print("未发现预训练模型，执行基础训练逻辑...")
                self.create_sample_models()
                
        except Exception as e:
            print(f"资源加载失败: {e}")
            self.create_sample_models()

    def create_sample_models(self):
        """保底逻辑：训练简单的逻辑回归"""
        self.models['logistic'] = LogisticRegression(random_state=42)
        X_dummy = np.random.rand(10, len(self.all_features))
        y_dummy = np.random.randint(0, 2, 10)
        self.models['logistic'].fit(X_dummy, y_dummy)

    def preprocess_input(self, data_dict):
        """将前端输入转换为模型可用的特征向量"""
        # 1. 提取基础数值特征
        df_num = pd.DataFrame([[data_dict[col] for col in self.numerical_cols]], columns=self.numerical_cols)
        
        # 2. 计算交互特征
        # age_creatinine = age * serum_creatinine
        age_creatinine = data_dict['age'] * data_dict['serum_creatinine']
        # ef_hbp = ejection_fraction * high_blood_pressure
        ef_hbp = data_dict['ejection_fraction'] * float(data_dict['high_blood_pressure'])
        
        df_inter = pd.DataFrame([[age_creatinine, ef_hbp]], columns=self.interaction_cols)
        
        # 3. 合并数值部分并进行标准化
        df_all_num = pd.concat([df_num, df_inter], axis=1)
        if self.scaler:
            X_num_scaled = self.scaler.transform(df_all_num)
        else:
            X_num_scaled = df_all_num.values
            
        # 4. 组合分类特征
        X_cat = np.array([[data_dict[col] for col in self.categorical_cols]])
        
        # 5. 拼接最终向量
        X_final = np.hstack([X_num_scaled, X_cat])
        return X_final

    def predict(self, input_data, model_name='stacking'):
        """进行预测"""
        try:
            processed_data = self.preprocess_input(input_data)
            
            # 如果请求的模型不存在，尝试使用 stacking 或第一个可用模型
            if model_name not in self.models:
                model_name = 'stacking' if 'stacking' in self.models else list(self.models.keys())[0]

            model = self.models[model_name]
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0]

            return {
                'prediction': int(prediction),
                'probability': float(probability[1]),
                'model_used': model_name
            }
        except Exception as e:
            print(f"预测错误: {e}")
            return {'error': str(e)}

predictor = HeartDiseasePredictor()

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data: return jsonify({'error': 'No data'}), 400

        # 特征映射（确保与前端 index.html 中的字段名对应）
        features = {
            'age': float(data.get('age', 0)),
            'anaemia': int(data.get('anaemia', 0)),
            'creatinine_phosphokinase': float(data.get('creatinine_phosphokinase', 0)),
            'diabetes': int(data.get('diabetes', 0)),
            'ejection_fraction': float(data.get('ejection_fraction', 0)),
            'high_blood_pressure': int(data.get('high_blood_pressure', 0)),
            'platelets': float(data.get('platelets', 0)),
            'serum_creatinine': float(data.get('serum_creatinine', 0)),
            'serum_sodium': float(data.get('serum_sodium', 0)),
            'sex': int(data.get('sex', 0)),
            'smoking': int(data.get('smoking', 0)),
            'time': float(data.get('time', 0))
        }

        model_choice = data.get('model', 'stacking')
        result = predictor.predict(features, model_choice)
        
        # 简单的风险因素分析
        risk_factors = []
        if features['age'] > 65: risk_factors.append('高龄风险')
        if features['ejection_fraction'] < 40: risk_factors.append('心脏泵血功能显著下降')
        if features['serum_creatinine'] > 1.5: risk_factors.append('肾功能受损指示器')
        
        result['risk_factors'] = risk_factors
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'active_model': 'stacking' if 'stacking' in predictor.models else 'basic_logistic',
        'interaction_features': True
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
