import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'
# 数据示例
categories = [
            "No Finding", "Enlarged \nCardiomediastinum", "Cardiomegaly",
            "Lung Lesion", "Lung Opacity", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
            "Pleural Other", "Fracture", "Support Devices"
]

# 两组模型的值 (示例数据，可替换为实际值)
# non_causal_model = [0.7339, 0.6119, 0.7051, 0.9729, 0.6847, 0.9695, 0.4881, 0.9966, 0.9390, 0.3881, 0.4102, 0.9915, 0.9678, 0.9610]
# vlci_model =       [0.7000, 0.6356, 0.6966, 0.9695, 0.6966, 0.9593, 0.6373, 0.9831, 0.9254, 0.6237, 0.5797, 0.9780, 0.9458, 0.9508]
non_causal_model = [0.4424, 0.9120, 0.6020, 0.9729, 0.7780, 0.9746, 0.8108, 0.9983, 0.9627, 0.9915, 0.9495, 0.9766, 0.9431, 0.9627]
vlci_model =       [0.5610, 0.9119, 0.7814, 0.9678, 0.8780, 0.9576, 0.9271, 0.9864, 0.9458, 0.9915, 0.9559, 0.9831, 0.9559, 0.9542]

# 将类别的数量均分到圆周
num_vars = len(categories)

# 创建角度（每个类别的起始点）
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合多边形

# 数据闭合（起点和终点相同）
non_causal_model += non_causal_model[:1]
vlci_model += vlci_model[:1]

# 开始绘制
fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(polar=True))

# 绘制每组模型的曲线
ax.plot(angles, non_causal_model, color='green', linewidth=2, label='Q-former')
ax.fill(angles, non_causal_model, color='green', alpha=0.25)

ax.plot(angles, vlci_model, color='orange', linewidth=2, label='Q-former + CI')
ax.fill(angles, vlci_model, color='orange', alpha=0.25)

# 在每个顶点处添加具体数值标注
# for angle, value in zip(angles, non_causal_model):
#     ax.text(angle, value -0.1, f'{value:.3f}', color='black', fontsize=14, ha='center', va='center')

# for angle, value in zip(angles, vlci_model):
#     ax.text(angle, value + 0.08, f'{value:.3f}', color='black', fontsize=14, ha='center', va='center')

# 根据数值大小动态调整位置
for angle, value_nc, value_vl in zip(angles, non_causal_model, vlci_model):
    # Non-causal model 数值位置
    if value_nc >= value_vl:
        ax.text(angle, value_nc + 0.08, f'{value_nc:.3f}', color='black', fontsize=20, ha='center', va='center')  # 外侧
    else:
        ax.text(angle, value_nc - 0.1, f'{value_nc:.3f}', color='black', fontsize=20, ha='center', va='center')  # 内侧

    # VLCI model 数值位置
    if value_vl > value_nc:
        ax.text(angle, value_vl + 0.08, f'{value_vl:.3f}', color='black', fontsize=20, ha='center', va='center')  # 外侧
    else:
        ax.text(angle, value_vl - 0.1, f'{value_vl:.3f}', color='black', fontsize=20, ha='center', va='center')  # 内侧
        
# 添加每个类别的标签
ax.xaxis.set_visible(True)
ax.set_xticks(angles[:-1])
# ax.set_xticklabels(categories, fontsize=10)
# 添加疾病名称并稍微远离圆环
for angle, label in zip(angles[:-1], categories):
    ax.text(angle, 1.4, label, fontsize=23, ha='center', va='center') 
# 隐藏默认的类别标签
# ax.set_xticks([])
ax.set_xticklabels([])

# 设置雷达图的范围
ax.set_ylim(0, 1)
ax.spines['polar'].set_visible(False)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)

# 添加标题
# ax.set_title("Comparison of Non-causal Model and VLCI", size=16, pad=20)

# 显示图例
ax.legend(loc='upper left', bbox_to_anchor=(-0.35, 1.26), fontsize=20, prop={'family': 'Times New Roman', 'size': 23})

# 显示图形
plt.tight_layout()
# plt.show()
output_path = r"D:\code\ubc\paper\new_paper\course\figure\image.png"  # 替换为你的目标保存路径
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 保存图像，dpi=300表示高分辨率
