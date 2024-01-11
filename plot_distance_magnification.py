import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体或其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号


df = pd.read_excel("计算结果.xlsx", sheet_name="可靠实验数据分析-2.4um (人工整理)", index_col=0)
Magnifications = df.iloc[:, 0::3]
equivalentDiameter = df.iloc[:, 2::3]

columns =[]
for column in Magnifications.columns:
    columns.append(column.split('_')[0])
Magnifications.columns = columns

columns =[]
for column in equivalentDiameter.columns:
    columns.append(column.split('_')[0])
equivalentDiameter.columns = columns

# 读取 实验的成像距离记录
df = pd.read_excel("放大倍率计算结果.xlsx", sheet_name="实验成像距离记录", usecols=[0, 1])
# 将 A 列和 B 列数据转换为字典
lookup_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
# 使用字典替换数组中的元素
Magnifications.columns = [lookup_dict.get(element, element) for element in Magnifications.columns]
equivalentDiameter.columns = [lookup_dict.get(element, element) for element in equivalentDiameter.columns]

# 按列索引大小排序
Magnifications = Magnifications.sort_index(axis=1)
equivalentDiameter = equivalentDiameter.sort_index(axis=1)

def PlotAndRegressioninOne():
    # 按行在同一坐标中绘制曲线
    plt.figure(figsize=(15, 9))
    # 绘制散点图和回归直线
    space = 0
    for index, row in Magnifications.iterrows():
        plt.scatter(Magnifications.columns, row, label=str(index) +'um', marker='o')
        # 计算回归直线
        slope, intercept, r_value, p_value, std_err = stats.linregress(Magnifications.columns, row)
        line = slope * Magnifications.columns + intercept
        plt.plot(Magnifications.columns, line, label=f'{index}um Regression', linestyle='--')
        plt.text(350, 30-space*1, f'{index}um Regression: Y = {slope}X+{intercept}', ha='center', va='center', color='red', fontsize=12)
        space=space+1

    plt.xlabel('X 成像距离mm\n\n\n\n')
    plt.ylabel('Y 尺度放大倍率')
    plt.legend()
    title = "可靠实验数据分析 成像距离-尺度放大倍率 关系"
    plt.title(title)
    output_path = 'Outputs\\' + title + '.png'
    plt.savefig(output_path, dpi=800)
    plt.show()
    plt.close()

def PlotAndRegression(df, VariableName):
    # 定义子图的行和列
    num_rows = 2
    num_cols = 2
    # 创建图形窗口和子图
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 12))
    # 绘制散点图和回归直线
    space = 0
    RegressionResult = pd.DataFrame()
    for index, row in df.iterrows():
        ax = axs[space//num_cols, space%num_cols]
        # 绘制散点
        ax.scatter(df.columns, row, label=str(index) +'um 原始数据', marker='o')
        # 一次函数回归
        # slope, intercept, r_value, p_value, std_err = stats.linregress(df.columns, row)
        # line = slope * df.columns + intercept
        # plt.plot(df.columns, line, label=f'{index}um Regression', linestyle='--')
        # plt.text(350, 30-space*1, f'{index}um Regression: Y = {slope}X+{intercept}', ha='center', va='center', color='red', fontsize=12)
        # space=space+1

        # 二次函数回归
        # 定义拟合函数（这里选择一个二次多项式）
        def quadratic_function(x, a, b, c, d):
            return a  + b * x + c*x**2 +d*x**3
        # 使用 curve_fit 进行拟合
        params, covariance = curve_fit(quadratic_function, df.columns, row)
        # 拟合后的参数
        RegressionResult[index] = params
        a_fit, b_fit, c_fit, d_fit = params
        # 生成拟合曲线的数据点
        x_fit = np.linspace(min(df.columns), max(df.columns), 100)
        y_fit = quadratic_function(x_fit, a_fit, b_fit, c_fit, d_fit)
        # 绘制原始散点和拟合曲线
        ax.plot(x_fit, y_fit, label=str(index) +'um 拟合曲线', linestyle='--')

        ax.set_xlabel('X 成像距离mm\n'+f'{index}um 拟合曲线: Y = {a_fit:.2e}+{b_fit:.2e}X+{c_fit:.2e}X^2 + {d_fit:.2e}X^3')
        ax.set_ylabel('Y ' + VariableName.split('-')[-1] + 'um')
        ax.legend()
        space=space+1
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
    title = "可靠实验数据分析 " + VariableName + " 关系"
    fig.suptitle(title, fontsize=16)
    output_path = 'Outputs\\' + title + '.png'
    fig.savefig(output_path, dpi=800)
    plt.show()
    plt.close()
    RegressionResult.index = ['a_fit', 'b_fit', 'c_fit', 'd_fit']
    return RegressionResult

Mag_RegressionResult = PlotAndRegression(Magnifications, '成像距离-尺度放大倍率-2.4um')
print(Mag_RegressionResult)

Diameter_RegressionResult = PlotAndRegression(equivalentDiameter, '成像距离-整体光斑等效直径-2.4um')
print(Diameter_RegressionResult)

# result = PlotAndRegression(Mag_RegressionResult, '孔直径-拟合系数-2.4um')
# print(result)

def PlotK_Distance(result, distance):
    for i in len(result[0]):
        CoefficientVector = CoefficientVector + [row[i] for row in result]*distance**i
    print(f"在距离为{distance}mm时，拟合的多项式系数如下：")
    print(CoefficientVector)
    degree = len(CoefficientVector) - 1  # 多项式的阶数为数据点的数量减一
    # 生成拟合曲线的 x 值
    x_fit = np.linspace(0,100, 100)
    # 计算拟合曲线的 y 值
    y_fit = np.polyval(CoefficientVector, x_fit)
    # 绘制拟合曲线
    plt.plot(x_fit, y_fit, label=f'Fitted Polynomial (degree {degree})', color='red')
    # 设置图形标题和坐标轴标签
    plt.title('Polynomial Fitting')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()

# PlotK_Distance(result,200)


