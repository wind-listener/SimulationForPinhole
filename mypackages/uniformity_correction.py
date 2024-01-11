import cv2
import numpy as np

def uniformity_correction(image):
    """
    均匀度校正算法，使用平均值进行校正。

    参数:
    - image: 输入图像

    返回:
    - 校正后的图像
    """
    # 将图像转换为浮点数类型
    image = image.astype(np.float32)

    # 计算图像的全局平均值
    mean_value = np.mean(image)

    # 应用均匀度校正
    corrected_image = image / mean_value

    # 将图像值限制在合理范围内（例如，0到255）
    corrected_image = np.clip(corrected_image, 0, 255)

    # 将校正后的图像转换为8位无符号整数类型
    corrected_image = corrected_image.astype(np.uint8)

    return corrected_image

