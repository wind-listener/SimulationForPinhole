# '''
# Date: 2024-03-08 22:12:48
# LastEditors: wind-listener 2775174829@qq.com
# LastEditTime: 2024-03-08 22:28:14
# FilePath: \PinholeAnalysis\utils\calculate_features_and_annotation.py
# '''
import numpy as np 
import cv2,os

def calculate_features_and_annotation(region,folder_path,output_folder,attribute_dict, Diameter):
    
    # 计算最大值、最小值、平均值、中位数和方差
    max_value = np.max(region)
    min_value = np.min(region)
    mean_value = np.mean(region)
    median_value = np.median(region)
    variance_value = np.var(region)

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
            # 图像的灰度分布信息：最值 平均值 中位数 方差
            f"{max_value:.4f},"
            f"{min_value:.4f},"
            f"{mean_value:.4f},"
            f"{median_value:.4f},"
            f"{variance_value:.4f},"
            # 图像的其余特征，拍照条件和成像位置等
            f"{attribute_dict['ImagingDistance']},"
            f"{attribute_dict['ExposureTime']},"
            f"{attribute_dict['PowerdividedbyArea']},"
            f"{attribute_dict['Fnumber']},"
            f"{attribute_dict['Focal']},"
            f"{attribute_dict['CMOSPixelSize']},"
            f"{attribute_dict['CMOSQE']},"
            f"{attribute_dict['X_coordinate']},"
            f"{attribute_dict['Y_coordinate']},"
            f"{attribute_dict['Width']},"
            f"{attribute_dict['Heigth']},"
            f"{attribute_dict['PixelCount']},"
            f"{Diameter}\n"
        )
    # 保存裁剪的图像
    cv2.imwrite(
        os.path.join(output_folder, f"{last_image_id + 1}.png"),
        region,
    )