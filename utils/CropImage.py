from PIL import Image


def crop_and_save_image(image_path, crop_coords, title):
    """
    从给定的图像中裁剪指定区域并保存到原路径。

    Parameters:
        image_path (str): 输入图像的路径。
        crop_coords (tuple): 包含左上角和右下角坐标的元组，格式为 ((x1, y1), (x2, y2))。

    Returns:
        str: 返回裁剪后图像的保存路径。
    """
    # 打开图像
    image = Image.open(image_path)

    # 获取裁剪区域的坐标
    (left, top), (right, bottom) = crop_coords

    # 裁剪图像
    cropped_image = image.crop((left, top, right, bottom))

    # 构造保存路径
    output_path = image_path.replace(".bmp", "_" + title + "_Cropped.bmp")

    # 保存裁剪后的图像，保持8位灰度图像格式
    cropped_image.save(output_path, format="BMP", bits=8)

    return output_path


# 示例用法
image_path = (
    "Dataset/ZZMImgs/LowPrecisionTest1_74.24/origin/Image_20231107181437558.bmp"
)
imgs = ["10", "20", "50", "100"]
i = 0
for y in range(800, 2400, 400):
    print(f"(1500,{y})(4000,{y + 300})")
    crop_coordinates = ((1500, y), (4100, y + 300))  # 以左上角(100, 50)和右下角(300, 250)裁剪
    output_path = crop_and_save_image(image_path, crop_coordinates, title=imgs[i])
    i += 1
    print(f"Cropped image saved at: {output_path}")
