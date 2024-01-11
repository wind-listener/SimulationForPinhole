import numpy as np
import matplotlib.pyplot as plt
from proper import *
from matplotlib.colors import ListedColormap

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体或其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

wavelength = 0.5  # 波长 0.5um
hole_diameter = 10e-6  # 针孔尺寸 10um

(wfo, sampling) = proper.prop_run('WholeSystem', wavelength, 2048, PASSVALUE={'hole_diameter': hole_diameter})



# import numpy as np
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots
#
# # 创建一个二维数组作为示例数据
# data = np.random.rand(10, 10)
#
# # 获取数组的维度
# x_size, y_size = data.shape
# x = np.arange(0, x_size, 1)
# y = np.arange(0, y_size, 1)
#
# # 创建网格
# x, y = np.meshgrid(x, y)
#
# # 创建3D曲面图
# fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'surface'}]])
#
# surface = go.Surface(z=data, x=x, y=y)
#
# # 添加曲面到图形
# fig.add_trace(surface)
#
# # 设置布局
# fig.update_layout(scene=dict(aspectmode='data'))
#
# # 创建FigureWidget，使图形可交互
# fig_widget = go.FigureWidget(fig)
#
# # 显示图形
# fig_widget.show()
#
# # 手动调整视角
# fig_widget.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)))
#
# # 显示交互式图形
# fig_widget




# 显示光场仿真结果



from mayavi import mlab

# 创建一个二维数组作为示例数据
data = wfo

# 获取数组的维度
x_size, y_size = data.shape

# 创建网格
x, y = np.meshgrid(np.arange(0, x_size), np.arange(0, y_size))

# 创建3D可视化
fig = mlab.figure(figure='2D Array Visualization')

# 使用surf函数绘制3D曲面图，并使用数据的值进行上色
surf = mlab.surf(x, y, data, colormap='viridis')

# 添加颜色条
mlab.colorbar(surf, title='Value')

# 显示图形
mlab.show()



# fig = plt.figure(figsize=(30, 10), dpi=100)
# plt.subplot(131)
# # 创建坐标网格
# x = np.linspace(-wfo.shape[1] // 2, wfo.shape[1] // 2, wfo.shape[1]) * sampling
# y = np.linspace(-wfo.shape[0] // 2, wfo.shape[0] // 2, wfo.shape[0]) * sampling
# # 调整坐标原点，使其位于图像中心
# x -= x.mean()
# y -= y.mean()
# # 绘制图像
# plt.imshow(wfo, cmap='gray', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
# plt.xlabel('X (meters)')
# plt.ylabel('Y (meters)')
# text = f'Wavelength: {wavelength} um\nHole Diameter: {hole_diameter} m\nPixel Size: {sampling} m/pixel'
# plt.title('成像仿真\n' + text)
# plt.colorbar()
#
# plt.subplot(132)
# plt.imshow(wfo, cmap='gray', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
# plt.xlabel('X (meters)')
# plt.ylabel('Y (meters)')
# plt.title('成像仿真\n' + "红圈直径为30mm")
# plt.colorbar()
# # 添加一个圆心在图像中心，直径为30mm的圆
# circle = plt.Circle((0, 0), 30e-3 / 2, color='red', fill=False)
# plt.gca().add_patch(circle)
#
# ax = plt.subplot(133, projection='3d')
# from mpl_toolkits.mplot3d import Axes3D
#
# # 创建一个网格
# x = np.linspace(-wfo.shape[0] // 2, wfo.shape[0] // 2, wfo.shape[0])
# y = np.linspace(-wfo.shape[1] // 2, wfo.shape[1] // 2, wfo.shape[1])
# X, Y = np.meshgrid(x, y)
# # 绘制3D图
# surf = ax.plot_surface(X, Y, wfo, cmap='viridis')
# # 添加颜色栏
# fig.colorbar(surf)
#
# # 调整子图之间的间距
# plt.tight_layout()
# # plt.savefig('my_figure.png', dpi=800)  # 保存图像为PNG文件，分辨率为800dpi
# plt.show()