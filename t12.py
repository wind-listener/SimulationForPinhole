#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project  ：SimulationForPinhole 
@File     ：t12.py
@IDE      ：PyCharm 
@Author   ：user
@Email    : zzm_ai@bupt.edu.cn
@Date     ：2023/12/25 11:08 
@Function ：构建数据集
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mypackages.config import *
from mypackages.gamma import gamma_correction
from mypackages.visualization3D import visualize_3d_array


def find_bright_regions(image, attribute_dict, isLoG=1, isGamma=0.5, isMorphology=0):
    """
    isGamma
    """
    if isLoG:
        image_name = "Gaussian_laplacian"
        image_smoothed = cv2.GaussianBlur(image, (7, 7), 0)
        laplacian_operator = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        Gaussian_laplacian = image_smoothed - cv2.Laplacian(
            image_smoothed, cv2.CV_64F, laplacian_operator
        )
        image = Gaussian_laplacian

    if isGamma > 0:
        ET = int(attribute_dict["ExposureTime"])
        if ET < 1000:
            isGamma = 0.01
        elif ET < 10000:
            isGamma = 0.4
        else:
            isGamma = 1

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
        radius = 1  # 开运算的核 圆形的半径 过滤直径为6以下的圆形噪点
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(2 * radius + 1), int(2 * radius + 1))
        )
        morphology_image = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_OPEN, kernel)
        radius = 50  # 闭运算的核 圆形的半径
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
    return num_labels, labels, stats, centroids


def main():
    # 文件夹路径
    folder_path = "Dataset/ZZMImgs/forRegressionTrain"
    rawimgs_path = os.path.join(folder_path, "raw")
    output_folder = os.path.join(folder_path, "cropped_imgs")

    # 1. 遍历文件夹中的文件
    # noinspection PyTypeChecker
    for filename in os.listdir(rawimgs_path):
        attribute_dict = {}
        # 拼接文件的完整路径
        image_path = os.path.join(rawimgs_path, filename)
        # 解析文件名
        filename_without_extension, extension = os.path.splitext(filename)
        attributes = filename_without_extension.split("-")
        # 解析每个属性
        for attribute in attributes:
            key, value = attribute.split("_")
            attribute_dict[key] = value
        if DEBUG:
            # 打印解析后的字典
            print("\n--------------------")
            print(f"File: {filename}")
            print("Attributes:")
            for key, value in attribute_dict.items():
                print(f"  {key}: {value}")

        # 2. 读取图像，寻找区域
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 检查图像是否成功读取
        if image is None:
            print("Error: Could not read the image.")
            exit()
        # 获取图像的高度
        image_height = image.shape[0]
        # 计算每个区间的高度
        region_height = image_height // 4
        # 寻找亮区域
        num_labels, labels, stats, centroids = find_bright_regions(
            image, attribute_dict
        )
        # 过滤面积超出阈值的连通区域
        min_area_threshold = 0 if int(attribute_dict["ExposureTime"]) < 1000 else 5
        max_area_threshold = 1000
        filtered_labels = [
            label
            for label, stat in enumerate(stats[1:], start=1)
            if min_area_threshold < stat[4] < max_area_threshold
        ]
        # 可视化找到的亮区域结果
        mask = np.zeros_like(image)
        for label in filtered_labels:
            mask[labels == label] = 255
            # 在图像上绘制文本, 标注面积
            area = stats[label][4]
            centroid = (int(centroids[label][0]) + 50, int(centroids[label][1] + 50))
            cv2.putText(
                mask, f"Area: {area}", centroid, cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2
            )
        rgba_array = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba_array[:, :, 0] = mask  # 红色通道
        rgba_array[:, :, 3] = 50  # 透明度
        rgba_mask = Image.fromarray(rgba_array, "RGBA")
        rgba_image = Image.fromarray(image).convert("RGBA")
        rgba_result = Image.alpha_composite(rgba_image, rgba_mask)
        rgba_result_name = "Area Result of " + os.path.basename(image_path)
        output_path = os.path.join(folder_path, "result", rgba_result_name)
        rgba_result.save(output_path, "BMP")

        stats = stats[filtered_labels, :]
        # 3.裁剪区域，保存和记录特征值
        for stat in stats:
            # 获取连通区域的坐标信息,x,y是leftTop点坐标
            x, y, w, h, area = stat[0], stat[1], stat[2], stat[3], stat[4]
            # 判断直径
            if y < region_height:
                Diameter = 10
            elif y < 2 * region_height:
                Diameter = 20
            elif y < 3 * region_height:
                Diameter = 50
            else:
                Diameter = 100
            # 裁剪图像
            cropped_region = image[y - 2 * h : y + 3 * h, x - 2 * w : x + 3 * w]
            # 计算最大值、最小值、平均值、中位数和方差
            max_value = np.max(cropped_region)
            min_value = np.min(cropped_region)
            mean_value = np.mean(cropped_region)
            median_value = np.median(cropped_region)
            variance_value = np.var(cropped_region)

            # 打开annotation.cvs记录
            with open(folder_path + "/annotation.csv", "r") as file:
                lines = file.readlines()
                last_image_id = 0
                if len(lines) > 1:  # 确保文件中有至少两行（包括标题行和数据行）
                    last_line = lines[-1].strip()  # 获取最新一行并去除首尾空白
                    last_image_id = int(last_line.split(",")[0])

            with open(folder_path + "/annotation.csv", "a") as file:
                file.write(
                    f"{last_image_id + 1},"
                    f"{attribute_dict['ImagingDistance']},"
                    f"{attribute_dict['ExposureTime']},"
                    f"{attribute_dict['PowerdividedbyArea']},"
                    f"{attribute_dict['Fnumber']},"
                    f"{attribute_dict['Focal']},"
                    f"{attribute_dict['CMOSPixelSize']},"
                    f"{attribute_dict['CMOSQE']},"
                    f"{w},"
                    f"{h},"
                    f"{area},"
                    # 图像的灰度分布信息：最值 平均值 中位数 方差
                    f"{max_value:.4f},"
                    f"{min_value:.4f},"
                    f"{mean_value:.4f},"
                    f"{median_value:.4f},"
                    f"{variance_value:.4f},"
                    f"{x},"
                    f"{y},"
                    f"{Diameter}\n"
                )
            # 保存裁剪的图像
            cv2.imwrite(
                os.path.join(output_folder, f"{last_image_id + 1}.png"),
                cropped_region,
            )


if __name__ == "__main__":
    main()
