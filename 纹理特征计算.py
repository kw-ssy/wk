# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互后端
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 兼容新旧版本的导入方式
try:
    from skimage.feature import graycomatrix, graycoprops
    print("使用新版本API: graycomatrix, graycoprops")
except ImportError:
    from skimage.feature import greycomatrix, greycoprops
    print("使用旧版本API: greycomatrix, greycoprops")
    graycomatrix = greycomatrix
    graycoprops = greycoprops

# 图像路径
img_path = "D:/pythonProject3/qimo/3_otsu_segmentation.jpg"

# 检查文件是否存在
if not os.path.exists(img_path):
    print(f"错误：文件不存在 - {img_path}")
    exit(1)

# 读取图像（以灰度图读取）
img = cv2.imread(img_path, 0)
if img is None:
    raise FileNotFoundError(f"无法读取图像: {img_path}")

# 计算GLCM
distances = [1]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
glcm = graycomatrix(img, distances=distances, angles=angles,
                    levels=256, symmetric=True, normed=True)

# 提取纹理特征
feature_names = ['energy', 'contrast', 'homogeneity', 'correlation', 'ASM']
features = {}

for name in feature_names:
    feature_values = graycoprops(glcm, name)
    features[name] = feature_values

# 手动计算熵
def calculate_entropy(glcm):
    glcm = np.maximum(glcm, 1e-10)  # 避免log(0)
    entropy = -np.sum(glcm * np.log(glcm))
    return entropy

entropy_values = []
for angle_idx in range(len(angles)):
    glcm_angle = glcm[:, :, 0, angle_idx]  # 注意索引顺序：(行,列,距离,方向)
    entropy = calculate_entropy(glcm_angle)
    entropy_values.append(entropy)
features['entropy'] = np.array(entropy_values).reshape(-1, 1)

# 手动计算IDM
idm_values = []
for angle_idx in range(len(angles)):
    glcm_angle = glcm[:, :, 0, angle_idx]
    i, j = np.indices(glcm_angle.shape)
    denominator = 1 + np.square(i - j)
    idm = np.sum(glcm_angle / denominator)
    idm_values.append(idm)
features['idm'] = np.array(idm_values).reshape(-1, 1)

# 打印特征结果（修正索引）
print("===== 纹理特征计算结果 =====")
for name, value in features.items():
    # 处理graycoprops返回的特征
    if name in feature_names:
        for angle_idx, angle in enumerate(angles):
            deg = int(angle * 180/np.pi)
            # 修正索引：value[0, angle_idx]
            print(f"{name} (方向{deg}°): {value[0, angle_idx]:.4f}")
    # 处理手动计算的特征（形状为(4,1)）
    else:
        for angle_idx, angle in enumerate(angles):
            deg = int(angle * 180/np.pi)
            print(f"{name} (方向{deg}°): {value[angle_idx, 0]:.4f}")

# 可视化GLCM
plt.figure(figsize=(10, 8))
for i, angle in enumerate(angles):
    deg = int(angle * 180/np.pi)
    plt.subplot(2, 2, i+1)
    sns.heatmap(glcm[:, :, 0, i], cmap='viridis')
    plt.title(f'GLCM (方向{deg}°)')
plt.tight_layout()
plt.savefig("4_glcm_visualization.jpg")
plt.close()

# 特征可视化
plt.figure(figsize=(10, 6))
angles_deg = [int(a*180/np.pi) for a in angles]
for name in feature_names + ['entropy', 'idm']:
    # 修正索引：根据特征来源选择不同的索引方式
    if name in feature_names:
        plt.plot(angles_deg, features[name][0, :], marker='o', label=name)
    else:
        plt.plot(angles_deg, features[name].flatten(), marker='o', label=name)
plt.xlabel('方向(°)'), plt.ylabel('特征值'), plt.title('不同方向的纹理特征')
plt.legend(), plt.grid(True)
plt.savefig("4_texture_features.jpg")
plt.show()

print("任务4完成，纹理特征已计算并可视化")