#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project  ：SimulationForPinhole 
@File     ：t9.py
@IDE      ：PyCharm 
@Author   ：user
@Email    : zzm_ai@bupt.edu.cn
@Date     ：2023/12/13 10:39 
@Function ：观察曝光时间较小的情况下，成像能否检测到针孔，图像是模拟实际的分辨率拍摄
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.config import *
from utils.gamma import gamma_correction
from utils.visualization3D import visualize_3d_array

image_path = "Dataset/ZZMImgs/ExposureTimeus/70000.bmp"
ExposureTime = float(image_path.split("/")[-1].split(".")[0])
isGamma = 1
isMorphology = 1

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# 检查图像是否成功读取
if image is None:
    print("Error: Could not read the image.")
    exit()

if isGamma > 0:
    image = gamma_correction(image, isGamma)  # 拉伸亮度
    if DEBUG:
        plt.imshow(image, cmap="gray"), plt.title(
            "origin image(Gamma:" + str(isGamma) + ")"
        ), plt.show()
        visualize_3d_array(image, title="origin image(Gamma:" + str(isGamma) + ")")

# 局部阈值分割
adaptive_threshold = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -1
)
if DEBUG:
    plt.imshow(adaptive_threshold, cmap="gray"), plt.title("threshold"), plt.show()
    visualize_3d_array(adaptive_threshold, title="threshold")

if isMorphology == 0:
    # 开运算 去除噪点
    # 闭运算 联通区域
    # 如果本来成像就小的话，应该采用最小的核
    radius = 1.5  # 圆形的半径 过滤直径为6以下的圆形噪点
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(2 * radius + 1), int(2 * radius + 1))
    )
    morphology_image = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_OPEN, kernel)
    radius = 50  # 圆形的半径
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    morphology_image = cv2.morphologyEx(morphology_image, cv2.MORPH_CLOSE, kernel)
    if DEBUG:
        plt.imshow(morphology_image, cmap="gray"), plt.title(
            "threshold-morphology"
        ), plt.show()
        visualize_3d_array(morphology_image, title="threshold-morphology")

elif isMorphology == 1:  # 只有闭运算
    radius = 5  # 圆形的半径
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    morphology_image = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_CLOSE, kernel)
    if DEBUG:
        plt.imshow(morphology_image, cmap="gray"), plt.title(
            "threshold-none"
        ), plt.show()
        visualize_3d_array(morphology_image, title="threshold-close")

elif isMorphology == 2:  # 无形态学操作
    morphology_image = adaptive_threshold
    if DEBUG:
        plt.imshow(morphology_image, cmap="gray"), plt.title(
            "threshold-none"
        ), plt.show()
        visualize_3d_array(morphology_image, title="threshold-none")

# 连通组件标记
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    morphology_image, connectivity=8
)
# 过滤面积大于阈值的连通组件
min_area_threshold = (
    0 if ExposureTime < 1000 else 5
)  # math.pow((322-2.2*diameter)/precision,2) * 0.03
max_area_threshold = 1000  # 此处使用了放大倍率和diameter之间的近似关系 mag = 322/diameter-2.2
filtered_labels = [
    label
    for label, stat in enumerate(stats[1:], start=1)
    if min_area_threshold < stat[4] < max_area_threshold
]
areas = stats[filtered_labels, 4]
# 计算均值和标准差
mean_val = np.mean(areas)
std_dev = np.std(areas)
if std_dev >= 0.01:
    # 计算Z-score
    z_scores = (areas - mean_val) / std_dev
    # 设置阈值（例如，2倍标准差）
    threshold = 1.5
    # 剔除偏离数据
    filtered_areas = areas[abs(z_scores) <= threshold]
else:  # 标准差太小就没有必要过滤
    filtered_areas = areas

# 创建只包含小点的二值图像
mask = np.zeros_like(morphology_image)
for label in filtered_labels:
    mask[labels == label] = 255
    # 在图像上绘制文本, 标注面积
    area = stats[label][4]
    centroid = (int(centroids[label][0]) + 50, int(centroids[label][1] + 50))
    cv2.putText(mask, f"Area: {area}", centroid, cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 1)

rgba_array = np.zeros((*mask.shape, 4), dtype=np.uint8)
rgba_array[:, :, 0] = mask  # 红色通道
rgba_array[:, :, 3] = 50  # 透明度
rgba_mask = Image.fromarray(rgba_array, "RGBA")
rgba_image = Image.fromarray(image).convert("RGBA")
rgba_result = Image.alpha_composite(rgba_image, rgba_mask)
rgba_result_name = (
    "Area Result of " + os.path.basename(image_path).split(".")[0] + ".bmp"
)
output_path = os.path.join(os.path.dirname(image_path), rgba_result_name)
rgba_result.save(output_path, "BMP")

# 显示处理结果：原始图像、二值图像和小点图像
fig = plt.figure(figsize=(7, 10))
plt.suptitle("Result of " + image_path)
# 原图
plt.subplot(2, 1, 1)
plt.imshow(image, cmap="gray")
plt.title("origin image(Gamma:" + str(isGamma) + ")")
plt.axis("off")
# 处理结果
plt.subplot(2, 1, 2)
plt.imshow(mask, cmap="gray")
plt.title("Small Points with filtered_Area\n" + str(filtered_areas))
plt.axis("off")
result_file_name = "Result of " + os.path.basename(image_path).split(".")[0]
output_path = os.path.join(os.path.dirname(image_path), result_file_name)
plt.savefig(output_path, dpi=800)  # 设置 dpi 参数以保存高分辨率图片
if DEBUG:
    plt.show()
plt.close(fig)
