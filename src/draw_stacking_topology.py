import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_rich_stacking_topology():
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    
    # 颜色风格
    c_data = '#ECEFF1'  # 数据集颜色
    c_base = '#E3F2FD'  # 基模型颜色
    c_new_feat = '#FFF3E0' # 新特征集颜色
    c_meta = '#FCE4EC'  # 元模型颜色
    c_line = '#546E7A'  # 线条颜色

    # --- 标题与阶段分隔 ---
    ax.text(6, 9.5, "第一阶段：训练基模型 (Base Models Training)", ha='center', fontsize=14, fontweight='bold')
    ax.axhline(y=3.5, color='#B0BEC5', linestyle='--', linewidth=1.5)
    ax.text(6, 3.1, "第二阶段：训练元模型 (Meta Model) & 最终预测", ha='center', fontsize=14, fontweight='bold')

    # --- 1. 原始训练集 ---
    raw_data_box = patches.FancyBboxPatch((0.5, 6), 2, 1.5, boxstyle="round,pad=0.1", ec=c_line, fc=c_data)
    ax.add_patch(raw_data_box)
    ax.text(1.5, 6.75, "原始训练集\n(Original\nTraining Set)", ha='center', va='center', fontsize=11)

    # --- 2. 基模型 (标注参数) ---
    models = [
        {"name": "基模型 A: GBDT", "params": "n_estimators=100\nlearning_rate=0.1", "y": 7.5},
        {"name": "基模型 B: RF", "params": "n_estimators=100\nmax_depth=8", "y": 6.25},
        {"name": "基模型 C: SVC", "params": "C=1.0\nkernel='rbf'", "y": 5.0}
    ]
    
    for m in models:
        m_box = patches.FancyBboxPatch((4, m['y']), 2.5, 1, boxstyle="round,pad=0.1", ec=c_line, fc=c_base)
        ax.add_patch(m_box)
        ax.text(5.25, m['y']+0.65, m['name'], ha='center', fontsize=10, fontweight='bold')
        ax.text(5.25, m['y']+0.25, m['params'], ha='center', fontsize=9, color='#455A64')
        
        # 箭头：原始数据 -> 基模型
        ax.annotate('', xy=(4, m['y']+0.5), xytext=(2.5, 6.75),
                    arrowprops=dict(arrowstyle='->', color=c_line, connectionstyle="arc3,rad=-0.1"))

    # --- 3. 新特征训练集 (关键样式还原) ---
    new_feat_box = patches.FancyBboxPatch((8.5, 5.2), 3, 3, boxstyle="round,pad=0.1", ec=c_line, fc=c_new_feat)
    ax.add_patch(new_feat_box)
    ax.text(10, 7.8, "新特征训练集\n(Stacking Dataset)", ha='center', fontweight='bold', fontsize=11)
    
    # 模拟数据内容
    data_content = (
        "Pred_A | Pred_B | Pred_C | Target\n"
        "-------------------------------\n"
        " 0.85  |  0.72  |  0.91  |   1\n"
        " 0.12  |  0.25  |  0.08  |   0\n"
        " 0.77  |  0.88  |  0.65  |   1\n"
        "  ...  |   ...  |   ...  |  ..."
    )
    ax.text(10, 6.4, data_content, ha='center', va='center', family='monospace', fontsize=9)

    # 箭头：基模型 -> 新特征集
    for m in models:
        ax.annotate('', xy=(8.5, 6.7), xytext=(6.5, m['y']+0.5),
                    arrowprops=dict(arrowstyle='->', color=c_line))

    # --- 4. 元模型 ---
    meta_box = patches.FancyBboxPatch((4.5, 0.8), 3, 1.5, boxstyle="round,pad=0.1", ec=c_line, fc=c_meta)
    ax.add_patch(meta_box)
    ax.text(6, 1.8, "元模型 (Meta Model)", ha='center', fontweight='bold', fontsize=12)
    ax.text(6, 1.2, "Ridge Classifier\n(Passthrough=True)", ha='center', fontsize=10)

    # 箭头：新特征集 -> 元模型
    # 使用弯曲箭头模拟从表格下方流出的效果
    ax.annotate('', xy=(7.5, 1.5), xytext=(10, 5.2),
                arrowprops=dict(arrowstyle='->', color=c_line, connectionstyle="angle3,angleA=0,angleB=-90"))
    
    # 箭头：原始特征 -> 元模型 (Passthrough)
    ax.annotate('', xy=(4.5, 1.5), xytext=(1.5, 6),
                arrowprops=dict(arrowstyle='->', color=c_line, linestyle='--', connectionstyle="angle3,angleA=90,angleB=180"))
    ax.text(2.5, 3.5, "Passthrough\n(特征透传)", ha='center', fontsize=9, color='#455A64')

    # --- 5. 最终预测 ---
    ax.text(10, 1.5, "最终预测结果\n(Final Prediction)", ha='center', va='center', 
            bbox=dict(boxstyle="round", fc='#C8E6C9', ec=c_line))
    ax.annotate('', xy=(9, 1.5), xytext=(7.5, 1.5), arrowprops=dict(arrowstyle='->', color=c_line))

    ax.axis('off')
    plt.tight_layout()
    
    # 保存为 TeX 文件引用的图片名称
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/stacking_topology.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("拓扑图已生成至 figures/stacking_topology.png")

if __name__ == "__main__":
    draw_rich_stacking_topology()
