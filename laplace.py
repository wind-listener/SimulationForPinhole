import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(image, gamma):
    # 避免除以零，确保最大值不为零
    max_value = float(np.max(image)) if np.max(image) != 0 else 1.0
    gamma_corrected = np.power(image / max_value, gamma) * 255.0
    gamma_corrected = np.uint8(gamma_corrected)
    return gamma_corrected

# 图像路径
image_path = 'Dataset/ZZMImgs/HighPrecisionTest5_8.65/10um.bmp'
# image_path = 'test.png'
# 读取灰度图像
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# 检查图像是否成功读取
if image is None:
    print("Error: Could not read the image.")
    exit()
plt.imshow(image, cmap='gray'),plt.title('origin image'),plt.show()

# 局部阈值分割
adaptive_threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, -1)
plt.imshow(adaptive_threshold, cmap='gray'),plt.title('threshold'),plt.show()

# 开运算 闭运算
radius = 10 # 圆形的半径
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
morphology_image = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_CLOSE, kernel)
plt.imshow(morphology_image, cmap='gray'),plt.title('threshold-morphology'),plt.show()
radius = 10 # 圆形的半径
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
morphology_image = cv2.morphologyEx(morphology_image, cv2.MORPH_OPEN, kernel)


# # 腐蚀一次，膨胀两次，腐蚀一次
# radius = 3 # 圆形的半径
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
# eroded_image = cv2.erode(adaptive_threshold, kernel, iterations=1)
# eroded_imagedilated = cv2.dilate(eroded_image, kernel, iterations=2)
# morphology_image = cv2.erode(eroded_imagedilated, kernel, iterations=1)
# plt.imshow(morphology_image, cmap='gray'),plt.title('threshold-morphology'),plt.show()

# morphology_image = adaptive_threshold
# 连通组件标记
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morphology_image, connectivity=8)

# 过滤面积大于阈值的连通组件
min_area_threshold = 10000  # 适当调整阈值
filtered_labels = [label for label, stat in enumerate(stats[1:], start=1) if 100 <stat[4] < min_area_threshold]

# 创建只包含小点的二值图像
small_points_image = np.zeros_like(morphology_image)
for label in filtered_labels:
    small_points_image[labels == label] = 255

# 显示原始图像、二值图像和小点图像
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(morphology_image, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(small_points_image, cmap='gray')
plt.title('Small Points')
plt.axis('off')

plt.show()

