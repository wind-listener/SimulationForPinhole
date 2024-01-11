import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_array(array,title):
    # 创建一个网格
    x, y = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))

    # 创建3D图像
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D图像
    ax.plot_surface(x, y, array, cmap='viridis')

    # 设置图像标题和轴标签
    ax.set_title('3D Visualization--'+title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    plt.show()