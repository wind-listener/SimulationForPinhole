import matplotlib.pyplot as plt
from PIL import Image
import os
gamma = 0.
image_path ="10.bmp"
# 打开图片
img = Image.open(image_path)

# 进行伽马变换
img_gamma = img.point(lambda x: x ** gamma)

image_save_path = "10_gamma.bmp"
img_gamma.save(image_save_path)
#
# plt.subplot(121)
# plt.imshow(img, cmap='gray')
#
# plt.subplot(122)
# plt.imshow(img_gamma, cmap='gray')