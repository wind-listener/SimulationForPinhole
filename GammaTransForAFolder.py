import os

from PIL import Image

# 定义伽马值，这里选择一个大于1的值, 会提亮图片
gamma = 2  # 可以根据需要调整伽马值

# 定义图片文件路径
image_names = ["10.bmp", "20.bmp", "50.bmp", "100.bmp"]
folder_path = r"Dataset/ZZMImgs/Repeat2_16.24/None_Gamma"
save_path = r"Dataset/ZZMImgs/Repeat2_16.24"
# folder_path = r"Dataset\ZZMImgs\HighPrecisionTest5_8.65\origin"
# save_path = r"Dataset\ZZMImgs\HighPrecisionTest5_8.65"

for image_name in image_names:
    # 构建完整的文件路径
    image_path = os.path.join(folder_path, image_name)

    # 打开图片
    img = Image.open(image_path)

    # 进行伽马变换
    img_gamma = img.point(lambda x: x**gamma)

    # 保存变换后的图片，替换原文件
    image_save_path = os.path.join(save_path, image_name)
    img_gamma.save(image_save_path)

print("伽马变换完成")
