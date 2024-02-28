import numpy as np


def gamma_correction(image, gamma):
    # 避免除以零，确保最大值不为零
    max_value = float(np.max(image)) if np.max(image) != 0 else 1.0
    # 对输入值进行限制，确保不出现负数
    image = np.clip(image, 0, None)
    gamma_corrected = np.power(image / max_value, gamma) * 255.0
    gamma_corrected = np.uint8(gamma_corrected)
    return gamma_corrected
