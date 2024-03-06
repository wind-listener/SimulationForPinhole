"""
Author: wind-listener 2775174829@qq.com
Date: 2023-12-22 10:42:15
LastEditors: wind-listener 2775174829@qq.com
LastEditTime: 2023-12-22 13:55:09
FilePath: \SimulationForPinhole\t10.py
Description: 验证Laplace算子效果

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
"""
import os

import cv2
import numpy as np

from utils.config import DEBUG

# # ————————————————————————————————————————————————————————————————
# # 构建一个255*255大小的8bit灰度图，并在中心设置一个灰度值为255的像素
# # 创建一个255x255大小的8位灰度图
# image = np.zeros((255, 255), dtype=np.uint8)

# # 获取图像中心坐标
# center_x, center_y = image.shape[1] // 2, image.shape[0] // 2

# # 在中心设置一个像素值为255的像素
# image[center_y, center_x] = 255

# # 保存图像为“test.bmp”
# cv2.imwrite("test.bmp", image)
# #——————————————————————————————————————————————————————————————————

# 读取图像
input_path = "ResultImages_t10\\test1pixel.bmp"
output_floder = "ResultImages_t10/"
image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

# 裁剪图像
if 0:  # 是否裁剪图像
    left_top = (300, 100)
    right_bottom = (1450, 600)
    cropped_image = image[left_top[1] : right_bottom[1], left_top[0] : right_bottom[0]]
else:
    cropped_image = image

# 保存用于测试的图像
output_path = output_floder + "test" + os.path.basename(input_path)
cv2.imwrite(output_path, cropped_image)

# 自定义3x3拉普拉斯算子，似乎也没必要用这个
laplacian_operator = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# # 1. 应用Laplace算子
# image_name = "Cropped+laplacian"
# laplacianResult = cropped_image - cv2.Laplacian(cropped_image, cv2.CV_64F, laplacian_operator)
# output_path = output_floder + image_name + ".bmp"
# cv2.imwrite(output_path, laplacian)
# if DEBUG:
#     cv2.imshow("Cropped Image", cropped_image)
#     cv2.imshow("Laplacian", laplacian)
#     cv2.imshow(image_name, laplacianResult)

# 2. 应用Gaussian滤波 和 应用Laplace算子
image_name = "Gaussian_laplacian"
image_smoothed = cv2.GaussianBlur(cropped_image, (7, 7), 0)
Gaussian_laplacian = image_smoothed - cv2.Laplacian(
    image_smoothed, cv2.CV_64F, laplacian_operator
)
if DEBUG:
    cv2.imshow("image_smoothed", image_smoothed)
    cv2.imshow(image_name, Gaussian_laplacian)
output_path = output_floder + image_name + ".bmp"
cv2.imwrite(output_path, Gaussian_laplacian)

# # 3. 应用Laplace算子 和 应用Gaussian滤波
# image_name = "laplacian_Gaussian"
# laplacian = cropped_image - cv2.Laplacian(cropped_image, cv2.CV_64F)
# image_smoothed = cv2.GaussianBlur(laplacian, (5, 5), 0)
# if DEBUG:
#     cv2.imshow("Original Image",cropped_image)
#     cv2.imshow("laplacian", laplacian)
#     cv2.imshow("laplacian_Gaussian", image_smoothed)
# output_path = output_floder + image_name + ".bmp"
# cv2.imwrite(output_path, image_smoothed)

cv2.waitKey(0)
cv2.destroyAllWindows()
