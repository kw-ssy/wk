# -*- coding: utf-8 -*-
import cv2
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt

img_color_corrected = cv2.imread("1_color_corrected.jpg")
gray = cv2.cvtColor(img_color_corrected, cv2.COLOR_BGR2GRAY)

# 2. 计算灰度直方图（可选，用于辅助确定阈值）
plt.figure(figsize=(8, 6))
plt.hist(gray.ravel(), 256, [0, 256])
plt.title("灰度直方图"), plt.xlabel("灰度值"), plt.ylabel("像素数")
plt.savefig("3_gray_histogram.jpg")
plt.close()

# 3. 方法1：Otsu自动阈值
thresh_otsu, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu自动阈值: {thresh_otsu}")

# 4. 方法2：经验阈值（示例：128，可手动调整）
thresh_manual = 128
binary_manual = cv2.threshold(gray, thresh_manual, 255, cv2.THRESH_BINARY)[1]

# 5. 形态学优化（去噪）
kernel = np.ones((3, 3), np.uint8)
binary_otsu_clean = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, kernel)
binary_manual_clean = cv2.morphologyEx(binary_manual, cv2.MORPH_OPEN, kernel)

# 6. 应用掩码到原图
result_otsu = np.zeros_like(img_color_corrected)
result_manual = np.zeros_like(img_color_corrected)
result_otsu[binary_otsu_clean > 0] = img_color_corrected[binary_otsu_clean > 0]
result_manual[binary_manual_clean > 0] = img_color_corrected[binary_manual_clean > 0]

# 7. 保存结果
cv2.imwrite("3_otsu_segmentation.jpg", result_otsu)
cv2.imwrite("3_manual_segmentation.jpg", result_manual)

# 8. 可视化
plt.figure(figsize=(12, 8))
plt.subplot(221), plt.imshow(gray, cmap='gray'), plt.title("灰度图")
plt.subplot(222), plt.imshow(binary_otsu, cmap='gray'), plt.title(f"Otsu阈值({thresh_otsu})")
plt.subplot(223), plt.imshow(binary_manual, cmap='gray'), plt.title(f"手动阈值({thresh_manual})")
plt.subplot(224), plt.imshow(cv2.cvtColor(result_otsu, cv2.COLOR_BGR2RGB)), plt.title("Otsu分割结果")
plt.tight_layout()
plt.savefig("3_threshold_segmentation.jpg")
plt.show()

print("任务3完成，结果已保存")