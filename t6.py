import numpy as np
import matplotlib.pyplot as plt

from mypackages.gamma import gamma_correction
from proper import *
from matplotlib.colors import ListedColormap

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体或其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

wavelength = 0.5  # 波长 0.5um
hole_diameter = 10e-6  # 针孔尺寸 10um

(wfo, sampling) = proper.prop_run('WholeSystem', wavelength, 1024, PASSVALUE={'hole_diameter': hole_diameter})


fig = plt.figure(figsize=(30, 10), dpi=100)
plt.subplot(131)
# 创建坐标网格
x = np.linspace(-wfo.shape[1] // 2, wfo.shape[1] // 2, wfo.shape[1]) * sampling
y = np.linspace(-wfo.shape[0] // 2, wfo.shape[0] // 2, wfo.shape[0]) * sampling
# 调整坐标原点，使其位于图像中心
x -= x.mean()
y -= y.mean()
# 绘制图像
plt.imshow(wfo, cmap='gray', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
text = f'Wavelength: {wavelength} um\nHole Diameter: {hole_diameter} m\nPixel Size: {sampling} m/pixel'
plt.title('成像仿真\n' + text)
plt.colorbar()

plt.subplot(132)
wfo_gamma = gamma_correction(wfo,0.2)
plt.imshow(wfo_gamma, cmap='gray', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
# 添加一个圆心在图像中心，直径为65um的圆
circle = plt.Circle((0, 0), 38.4e-6 / 2, color='red', fill=False)
plt.gca().add_patch(circle)
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.title('成像仿真 gamma\n' + "红圈直径为38.4um")
plt.colorbar()

ax = plt.subplot(133, projection='3d')
# 创建一个网格
x = np.linspace(-wfo.shape[0] // 2, wfo.shape[0] // 2, wfo.shape[0])
y = np.linspace(-wfo.shape[1] // 2, wfo.shape[1] // 2, wfo.shape[1])
X, Y = np.meshgrid(x, y)
# 绘制3D图
surf = ax.plot_surface(X, Y, wfo, cmap='viridis')
# 添加颜色栏
fig.colorbar(surf)

# 调整子图之间的间距
plt.tight_layout()
# plt.savefig('my_figure.png', dpi=800)  # 保存图像为PNG文件，分辨率为800dpi
plt.show()