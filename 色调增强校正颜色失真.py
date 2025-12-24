# -*- coding: utf-8 -*-
import cv2
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# 1. 读取原始图像
img_path = "D:/pythonProject3/qimo/img.png"  # 替换为实际水稻图片路径
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"无法读取图像: {img_path}")

# 2. BGR转HSV（OpenCV默认读取为BGR）
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# 3. 调整色调（示例：增加5，可根据实际失真情况调整）
h_adjusted = h.astype(np.float32) + 5
h_adjusted = np.clip(h_adjusted, 0, 180).astype(np.uint8)  # H范围[0,180]

# 4. 合并通道并转回BGR
hsv_adjusted = cv2.merge([h_adjusted, s, v])
img_color_corrected = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

# 5. 保存结果
cv2.imwrite("1_color_corrected.jpg", img_color_corrected)

# 6. 可视化对比
plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("原图")
plt.subplot(132), plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)), plt.title("HSV原图")
plt.subplot(133), plt.imshow(cv2.cvtColor(img_color_corrected, cv2.COLOR_BGR2RGB)), plt.title("色调增强后")
plt.tight_layout()
plt.savefig("1_color_corrected_compare.jpg")
plt.show()

print("任务1完成，结果已保存")