# """
# Date: 2024-02-29 16:14:48
# LastEditors: wind-listener 2775174829@qq.com
# LastEditTime: 2024-03-08 22:17:14
# FilePath: \PinholeAnalysis\prepareDataset.py
# """

import os
import sys

# sys.path.append(r"E:\WPSSync\Projects\SimulationForPinhole\utils")
import cv2
import numpy as np
from PIL import Image
from skimage.transform import warp, SimilarityTransform
from utils.findBrightRegions import findBrightRegions
import shutil
from utils.calculate_features_and_annotation import calculate_features_and_annotation
from utils.getDiameterfromPosition import getDiameterfromPosition
from utils.plot_text_on_mask import plot_text_on_mask

# 文件夹路径
folder_path = r"E:\WPSSync\Projects\PinholeAnalysis\Dataset\NewPinholeSampleImage"
rawimgs_path = os.path.join(folder_path, "raw")
output_folder = os.path.join(folder_path, "cropped_imgs")

# conditions.txt存在的话，就读入
txt_file_path = os.path.join(rawimgs_path, "conditions.txt")
print(txt_file_path)

if os.path.exists(txt_file_path):
    # 如果文件存在，则读取文件内容并保存到字符串中
    with open(txt_file_path, "r") as file:
        file_content = file.read()
    attribute_dict = {}
    attributes = file_content.split("-")
    # 解析每个属性
    for attribute in attributes:
        key, value = attribute.split("_")
        attribute_dict[key] = value
    # 输出读取到的文件内容
    print("实验条件：\n")
    for key, value in attribute_dict.items():
        print(f"  {key}: {value}")
else:
    print("实验条件相关文件不存在")

# 先清空output_folder文件夹
shutil.rmtree(output_folder)
os.mkdir(output_folder)
# 打开文件，使用写入模式覆盖并写入表头
annotation_file_path = folder_path + "/annotation.csv"
header = "ImageID,MaxValue,MinValue,Mean,Median,Variance,ImagingDistance,ExposureTime,PowerdividedbyArea,Fnumber,Focal,CMOSPixelSize,CMOSResponsivity,Xcoordinate,Ycoordinate,Width,Height,PixelCount,Diameter\n"
with open(annotation_file_path, "w") as file:
    file.write(header)
print("文件内容已清空")

# 1. 遍历文件夹中的文件
for file_name in os.listdir(rawimgs_path):
    if file_name.endswith(".bmp"):
        # 处理BMP文件的逻辑
        print(f"处理BMP文件: {file_name}")
        
        image_path = os.path.join(
            r"E:\WPSSync\Projects\PinholeAnalysis\Dataset\NewPinholeSampleImage\\raw", file_name
        )
        # 2. 读取图像，寻找区域
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 寻找亮区域
        num_labels, labels, stats, centroids = findBrightRegions(
            image, file_name, attribute_dict, isLoG=0
        )
        # 过滤面积超出阈值的连通区域
        min_area_threshold = 0 if int(attribute_dict["ExposureTime"]) < 1000 else 5
        max_area_threshold = 1000
        # print(f"min_area_threshold = {min_area_threshold}")
        # print(f"max_area_threshold = {max_area_threshold}")
        filtered_labels = [
            label
            for label, stat in enumerate(stats[1:], start=1)
            if min_area_threshold < stat[4] < max_area_threshold
        ]
        stats = stats[filtered_labels, :]

        # mask用于可视化找到的亮区域结果
        mask = np.zeros_like(image)
        # 获取图像的边长, 一张图中可以均分为10个竖条区域，区域内的针孔大小相同
        image_width = image.shape[0]
        margin = 276  # 第一列针孔距离边缘的距离，注意拍摄图像时，尽量不要让多张图片之间的margin差距过大，这是平均值
        region_width = (image_width - 2 * margin) / 9
        # 3.裁剪区域，保存和记录特征值
        for stat, label in zip(stats, filtered_labels):
            # 获取连通区域的坐标信息,x,y是leftTop点坐标
            x, y, w, h, area = stat[0], stat[1], stat[2], stat[3], stat[4]
            attribute_dict["X_coordinate"] = x
            attribute_dict["Y_coordinate"] = y
            attribute_dict["Width"] = w
            attribute_dict["Heigth"] = h
            attribute_dict["PixelCount"] = area
            # 根据位置判断直径
            min_diameter = int(os.path.basename(image_path).split("_")[0])
            Diameter = getDiameterfromPosition(x, region_width, margin, min_diameter)
            # 绘制mask
            mask[labels == label] = 255
            # 在mask上绘制文本
            text = f"{area}pixel\n{Diameter}um"
            plot_text_on_mask(mask, text, label=label, centroid=centroids[label])
            # 裁剪图像
            # # 根据长宽裁剪
            # cropped_region = image[y - 2 * h : y + 3 * h, x - 2 * w : x + 3 * w]
            # 按正方形裁剪
            length = 10 * max(w, h)
            center = [x + w / 2, y + h / 2]
            cropped_region = image[
                int(center[1] - length / 2) : int(center[1] + length / 2),
                int(center[0] - length / 2) : int(center[0] + length / 2),
            ]

            calculate_features_and_annotation(
                region=cropped_region,
                folder_path=folder_path,
                output_folder=output_folder,
                attribute_dict=attribute_dict,
                Diameter=Diameter,
            )

        rgba_array = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba_array[:, :, 0] = mask  # 红色通道
        rgba_array[:, :, 3] = 100  # 透明度
        rgba_mask = Image.fromarray(rgba_array, "RGBA")
        rgba_image = Image.fromarray(image).convert("RGBA")
        rgba_result = Image.alpha_composite(rgba_image, rgba_mask)

        rgba_result_name = "Area Result of " + os.path.basename(image_path)
        output_path = os.path.join(folder_path, "result", rgba_result_name)
        rgba_result.save(output_path, "BMP")
