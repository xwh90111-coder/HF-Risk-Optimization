import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_closed_loop_system():
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    
    # 颜色配置
    c_terminal = '#E8F5E9' # 终端-绿色
    c_cloud = '#E3F2FD'    # 云端-蓝色
    c_physician = '#FFF3E0'# 医生-橙色
    c_line = '#455A64'

    # --- 1. 终端采集层 (Data Acquisition) ---
    terminal_box = patches.FancyBboxPatch((0.5, 3.5), 2.2, 1.5, boxstyle="round,pad=0.1", ec=c_line, fc=c_terminal)
    ax.add_patch(terminal_box)
    ax.text(1.6, 4.5, "多维终端采集层", ha='center', fontweight='bold', fontsize=11)
    ax.text(1.6, 4.0, "便携式家用医疗设备\n(血压、心率、随访指标)", ha='center', fontsize=9)

    # --- 2. 云端计算层 (Inference & Stacking) ---
    cloud_box = patches.FancyBboxPatch((3.8, 5.5), 2.5, 1.5, boxstyle="round,pad=0.1", ec=c_line, fc=c_cloud)
    ax.add_patch(cloud_box)
    ax.text(5.05, 6.5, "云端智能计算中心", ha='center', fontweight='bold', fontsize=11)
    ax.text(5.05, 6.0, "Stacking 集成模型推理\n(GBDT+RF+SVC -> Ridge)", ha='center', fontsize=9)

    # --- 3. 可解释反馈层 (SHAP Interpretation) ---
    shap_box = patches.FancyBboxPatch((7.2, 3.5), 2.2, 1.5, boxstyle="round,pad=0.1", ec=c_line, fc='#FCE4EC')
    ax.add_patch(shap_box)
    ax.text(8.3, 4.5, "深度可解释反馈层", ha='center', fontweight='bold', fontsize=11)
    ax.text(8.3, 4.0, "SHAP 风险地图归因\n(识别核心致死因子)", ha='center', fontsize=9)

    # --- 4. 临床决策干预层 (Physician Intervention) ---
    physician_box = patches.FancyBboxPatch((3.8, 1.0), 2.5, 1.5, boxstyle="round,pad=0.1", ec=c_line, fc=c_physician)
    ax.add_patch(physician_box)
    ax.text(5.05, 2.0, "临床决策干预层", ha='center', fontweight='bold', fontsize=11)
    ax.text(5.05, 1.5, "临床医生精准诊疗\n(资源配置与预后管理)", ha='center', fontsize=9)

    # --- 绘制闭环线条 (The Loop) ---
    # 1 -> 2 (数据上传)
    ax.annotate('加密数据流', xy=(3.8, 6.25), xytext=(2.7, 5.0),
                arrowprops=dict(arrowstyle='-|>', color=c_line, connectionstyle="angle3,angleA=0,angleB=-90"))
    
    # 2 -> 3 (结果分发)
    ax.annotate('风险预测与分析', xy=(7.2, 4.25), xytext=(6.3, 6.25),
                arrowprops=dict(arrowstyle='-|>', color=c_line, connectionstyle="angle3,angleA=0,angleB=-90"))

    # 3 -> 4 (反馈给医生)
    ax.annotate('归因可视化', xy=(6.3, 1.75), xytext=(8.3, 3.5),
                arrowprops=dict(arrowstyle='-|>', color=c_line, connectionstyle="angle3,angleA=0,angleB=90"))

    # 4 -> 1 (临床干预 - 闭环完成)
    ax.annotate('诊疗方案下达', xy=(1.6, 3.5), xytext=(3.8, 1.75),
                arrowprops=dict(arrowstyle='-|>', color=c_line, connectionstyle="angle3,angleA=0,angleB=90", linestyle='--'))

    # 中央说明
    ax.text(5, 4.25, "SYSTEMIC\nCLOSED-LOOP", ha='center', va='center', fontsize=14, 
            fontweight='black', alpha=0.1, color='gray')

    ax.axis('off')
    plt.title("融合 Stacking 与 SHAP 的智能随访闭环管理系统逻辑框架", fontsize=16, pad=20)
    plt.tight_layout()
    
    # 保存
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/closed_loop_system_logic.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("系统逻辑图已生成至 figures/closed_loop_system_logic.png")

if __name__ == "__main__":
    draw_closed_loop_system()
