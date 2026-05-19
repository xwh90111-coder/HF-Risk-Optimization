import os

# Update src/model_comparison.py
path_mc = 'src/model_comparison.py'
with open(path_mc, 'r', encoding='utf-8') as f:
    content_mc = f.read()
content_mc = content_mc.replace("data.drop('DEATH_EVENT', axis=1)", "data.drop('cardio', axis=1)")
content_mc = content_mc.replace("data['DEATH_EVENT']", "data['cardio']")
with open(path_mc, 'w', encoding='utf-8') as f:
    f.write(content_mc)

# Update src/logistic_regression_analysis.py
path_lr = 'src/logistic_regression_analysis.py'
with open(path_lr, 'r', encoding='utf-8') as f:
    content_lr = f.read()
content_lr = content_lr.replace("data.drop('DEATH_EVENT', axis=1)", "data.drop('cardio', axis=1)")
content_lr = content_lr.replace("data['DEATH_EVENT']", "data['cardio']")
content_lr = content_lr.replace("['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']", "['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure']")
content_lr = content_lr.replace("['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']", "['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']")
content_lr = content_lr.replace("x='DEATH_EVENT'", "x='cardio'")
content_lr = content_lr.replace("by DEATH_EVENT", "by cardio")
with open(path_lr, 'w', encoding='utf-8') as f:
    f.write(content_lr)

# Update src/explainability_analysis.py
path_ea = 'src/explainability_analysis.py'
with open(path_ea, 'r', encoding='utf-8') as f:
    content_ea = f.read()
content_ea = content_ea.replace("data.drop('DEATH_EVENT', axis=1)", "data.drop('cardio', axis=1)")
content_ea = content_ea.replace("""feature_mapping = {
    'age': 'Age', 'creatinine_phosphokinase': 'CPK', 'ejection_fraction': 'EF',
    'platelets': 'Platelets', 'serum_creatinine': 'Cr', 'serum_sodium': 'Na',
    'time': 'Time', 'age_creatinine': 'Age*Cr', 'ef_hbp': 'EF*HBP',
    'anaemia': 'Anaemia', 'diabetes': 'Diabetes', 'high_blood_pressure': 'HBP',
    'sex': 'Sex', 'smoking': 'Smoking'
}""", """feature_mapping = {
    'age': 'Age', 'height': 'Height', 'weight': 'Weight',
    'ap_hi': 'Sys_BP', 'ap_lo': 'Dia_BP', 'bmi': 'BMI',
    'pulse_pressure': 'Pulse_Pressure', 'gender': 'Gender',
    'cholesterol': 'Chol', 'gluc': 'Glucose', 'smoke': 'Smoke',
    'alco': 'Alcohol', 'active': 'Active'
}""")
content_ea = content_ea.replace("interaction_index=\"Cr\"", "interaction_index=\"Sys_BP\"")
content_ea = content_ea.replace("年龄(Age)与血清肌酐(Cr)的交互效应分析", "年龄(Age)与收缩压(Sys_BP)的交互效应分析")
content_ea = content_ea.replace("心力衰竭风险预测", "心血管疾病风险预测")
with open(path_ea, 'w', encoding='utf-8') as f:
    f.write(content_ea)

print("Scripts updated successfully.")
