'''
根据数输入的四个坐标提取矩形区域

'''
from PIL import Image
from skimage.transform import warp, SimilarityTransform
import numpy as np

def extract_and_square(image_path, coordinates):
    # 读取图像
    image = Image.open(image_path)

    # 获取四个点的坐标
    src = np.array(coordinates)

    # 定义目标正方形的坐标
    dst = np.array([[0, 0], [0, 5000], [5000, 5000], [5000, 0]])

    # 计算透视变换矩阵
    transform = SimilarityTransform()
    transform.estimate(src, dst)

    # 进行透视变换
    warped = warp(np.array(image), transform.inverse, output_shape=(5000, 5000))

    # 将变换后的图像转换为PIL图像对象
    squared_image = Image.fromarray((warped * 255).astype(np.uint8))

    return squared_image


# 读入图片并提取指定区域并矫正为正方形
image_path1 = "Dataset/NewPinholeSampleImage/Image10_100.bmp"
image_path2 = "Dataset/NewPinholeSampleImage/Image110_200.bmp"

fig1 = [[1052, 868], [1002, 2384], [2543, 2405], [2550, 860]]
fig2 = [[2554, 886], [2539, 2433], [4116, 2446], [4079, 891]]
fig3 = [[1053, 890], [1005, 2408], [2539, 2428], [2565, 881]]
fig4 = [[2506, 2428], [4098, 2431], [4075, 862], [2526, 878]]

squared_region1 = extract_and_square(image_path1, fig1)
squared_region1.save("Dataset/NewPinholeSampleImage/region1.bmp")
squared_region2 = extract_and_square(image_path1, fig2)
squared_region2.save("Dataset/NewPinholeSampleImage/region2.bmp")
squared_region3 = extract_and_square(image_path2, fig3)
squared_region3.save("Dataset/NewPinholeSampleImage/region3.bmp")
squared_region4 = extract_and_square(image_path2, fig4)
squared_region4.save("Dataset/NewPinholeSampleImage/region4.bmp")