import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import pandas as pd

# 提供的系数
coefficients = [
    [368.820950, -2.601917, 6.007524e-03, -4.115482e-06],
    [-19.663216, 0.139882, -3.269310e-04, 2.254633e-07],
    [0.361028, -0.002571, 6.047093e-06, -4.188682e-09],
    [-0.001994, 0.000014, -3.352547e-08, 2.328587e-11]
]

# 创建 DataFrame
df = pd.DataFrame(coefficients, columns=['a_fit', 'b_fit', 'c_fit', 'd_fit'], index=['a_fit', 'b_fit', 'c_fit', 'd_fit'])

def PlotK_Distance(result, distance):
    CoefficientVector = [0,0,0,0]
    for i in range(result.shape[1]):
        CoefficientVector = CoefficientVector + result.iloc[:, i] * distance**i
    print(f"在距离为{distance}mm时，拟合的多项式系数如下：")
    print(CoefficientVector)
    degree = len(CoefficientVector) - 1  # 多项式的阶数为数据点的数量减一
    # 生成拟合曲线的 x 值
    x_fit = np.linspace(0,100, 100)
    # 计算拟合曲线的 y 值
    ReversedCoefficientVector = CoefficientVector[::-1]
    y_fit = np.polyval(ReversedCoefficientVector, x_fit)
    # 绘制拟合曲线
    plt.plot(x_fit, y_fit, label=f'Fitted Polynomial (degree {degree})', color='red')
    # 设置图形标题和坐标轴标签
    plt.title(f'Polynomial Fitting Distance={distance}mm')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()


# PlotK_Distance(df, 200)

for distance in range(200, 800, 100):
    print(f"距离是：{distance}时:")
    PlotK_Distance(df, distance)