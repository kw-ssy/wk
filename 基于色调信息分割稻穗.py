# -*- coding: utf-8 -*-
import cv2
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt

img_color_corrected = cv2.imread("1_color_corrected.jpg")
hsv = cv2.cvtColor(img_color_corrected, cv2.COLOR_BGR2HSV)

# 2. 设定稻穗的H范围（需根据实际图像调整，示例值：30-80为偏黄绿色）
h_min, h_max = 30, 80
s_min, s_max = 50, 255  # 饱和度阈值，避免过亮或过暗区域
v_min, v_max = 50, 255  # 亮度阈值

# 3. 创建掩码
mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))

# 4. 应用掩码（保留稻穗，其余置黑）
result = np.zeros_like(img_color_corrected)
result[mask > 0] = img_color_corrected[mask > 0]

# 5. 保存结果
cv2.imwrite("2_hue_segmentation.jpg", result)

# 6. 可视化
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(cv2.cvtColor(img_color_corrected, cv2.COLOR_BGR2RGB)), plt.title("色调增强图")
plt.subplot(122), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title("稻穗分割结果")
plt.tight_layout()
plt.savefig("2_segmentation_compare.jpg")
plt.show()

print("任务2完成，结果已保存")