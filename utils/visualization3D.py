import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_image_and_3d_visualization(image, isGamma = 1, suptitle = ''):
    # 创建一个网格
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))

    # 创建figure
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle(suptitle, fontsize=16)

    # 绘制第一个子图（原始图像）
    ax1 = fig.add_subplot(121)
    ax1.imshow(image, cmap="gray")
    ax1.set_title("Image")
    ax1.axis('on')  # 关闭坐标轴

    # 绘制第二个子图（3D数组）
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(x, y, image, cmap='viridis')
    ax2.set_title("3D Visualization")
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    ax2.set_zlabel('Z-axis')

    plt.show()
    
    

def visualize_3d_image(image,title):
    # 创建一个网格
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))

    # 创建3D图像
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D图像
    ax.plot_surface(x, y, image, cmap='viridis')

    # 设置图像标题和轴标签
    ax.set_title('3D Visualization--'+title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    plt.show()