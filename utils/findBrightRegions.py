import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.config import *
from utils.gamma import gamma_correction
from utils.visualization3D import visualize_3d_image, show_image_and_3d_visualization


def findBrightRegions(image, filename, attribute_dict, isLoG=1, isGamma=0.5, Morphology=0):
    """
    isGamma
    """
    # 执行 LoG 边缘检测：将图像进行高斯模糊处理，然后应用拉普拉斯算子进行边缘检测
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
            suptitle1 = filename + f" image after Gamma Trans(Gamma:{isGamma})"
            show_image_and_3d_visualization(image, suptitle=suptitle1)

    # 局部阈值分割
    adaptive_threshold = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -1
    )
    if DEBUG:
        if isGamma > 0:
            suptitle2 = filename + f" (Gamma:{isGamma}) + adaptive_threshold"
        else:
            suptitle2 = filename + f" adaptive_threshold" 
        show_image_and_3d_visualization(adaptive_threshold, suptitle= suptitle2)

    if Morphology != None:
        if Morphology == 0:
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

        elif Morphology == 1:  # 只有闭运算
            radius = 5  # 圆形的半径
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            morphology_image = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_CLOSE, kernel)

        elif Morphology == 2:  # 无形态学操作
            morphology_image = adaptive_threshold
    if DEBUG:
        suptitle3 = suptitle2 +f" + Morphology{Morphology}"
        show_image_and_3d_visualization(adaptive_threshold, suptitle= suptitle3)

    # 连通组件标记
    # 连通组件标记的结果存储在 result 中，result 是一个元组，包含了四个元素：
    # - 第一个元素 num_labels 表示图像中连通组件的数量（包括背景）。
    # - 第二个元素 labels 是一个与输入图像大小相同的数组，每个像素点都被分配了一个标签，标签值表示该像素所属的连通组件。
    # - 第三个元素 stats 是一个二维数组，每一行包含一个连通组件的统计信息，如左上角坐标、宽度、高度等。
    # - 第四个元素 centroids 是一个包含连通组件中心坐标的数组。
    result = cv2.connectedComponentsWithStats(morphology_image, connectivity=8)
    # 返回 result。
    return result
