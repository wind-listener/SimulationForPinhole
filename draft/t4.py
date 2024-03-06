
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体或其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

# 读取 Excel 文件中的三个 sheet
file_path = "计算结果.xlsx"
sheets = ["可靠实验数据分析"]
# 读取并拼接数据
dfs = []
for sheet in sheets:
    df = pd.read_excel(file_path, sheet_name=sheet, index_col=0)
    # 获取偶数列的数据
    even_columns = df.iloc[:, 0::2]
    # 获取奇数列的数据
    odd_columns = df.iloc[:, 1::2]
    dfs.append(even_columns)

df_combined = pd.concat(dfs, axis=1)

# 提取列索引中的数字
columns =[]
for column in df_combined.columns:
    columns.append(column.split('_')[0])
df_combined.columns = columns

# 读取 Excel 文件
file_path = "放大倍率计算结果.xlsx"
sheet_name = "实验成像距离记录"

# 读取 A 列和 B 列数据
df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=[0, 1])

# 将 A 列和 B 列数据转换为字典
lookup_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

# 使用字典替换数组中的元素
df_combined.columns = [lookup_dict.get(element, element) for element in df_combined.columns]
# 按列索引大小排序
df_combined = df_combined.sort_index(axis=1)

# 按行在同一坐标中绘制曲线
plt.figure(figsize=(15, 9))
title = "放大倍率-成像距离（原始数据）"
plt.title(title)

# 绘制散点图和回归直线
space = 1
for index, row in df_combined.iterrows():
    plt.scatter(df_combined.columns, row, label=str(index) +'um', marker='o')
    # 计算回归直线
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_combined.columns, row)
    line = slope * df_combined.columns + intercept
    plt.plot(df_combined.columns, line, label=f'{index}um Regression', linestyle='--')
    plt.text(350, 30-space*1, f'{index}um Regression: Y = {slope}X+{intercept}', ha='center', va='center', color='red', fontsize=12)
    space=space+1

plt.xlabel('X 成像距离mm\n\n\n\n')
plt.ylabel('Y 放大倍率')
plt.legend()

output_path = 'Outputs\\' + title + '.png'
plt.savefig(output_path, dpi=800)
plt.show()



# # 按行在同一坐标中绘制曲线
# plt.figure(figsize=(10, 6))
# title ="放大倍率-成像距离（原始数据）"
# plt.title(title)
# for index, row in df_combined.iterrows():
#     # plt.xscale('log')
#     plt.plot(df_combined.columns, row, label=index, marker = 'o')
# plt.xlabel('成像距离mm')
# plt.ylabel('放大倍率')
# plt.legend()
# output_floder = 'Outputs\\'+ title+'.png'
# plt.savefig(output_floder, dpi=800)
# plt.show()



#
# # 使用 Z-score 方法剔除异常数据
# z_scores = stats.zscore(df_combined, axis=0)
# threshold = 1  # 设置阈值，可以根据需要调整
# df_combined = df_combined[(z_scores < threshold).all(axis=1)]
# # 按行在同一坐标中绘制曲线
# plt.figure(figsize=(10, 6))
# title= "放大倍率-精度（Z-score剔除异常据）"
# plt.title(title)
# for index, row in df_combined.iterrows():
#     # plt.xscale('log')
#     plt.plot(df_combined.columns, row, label=index, marker = 'o')
# plt.xlabel('精度um/pixel')
# plt.ylabel('放大倍率')
# plt.legend()
# output_floder = 'Outputs\\'+ title+'.png'
# plt.savefig(output_floder, dpi=800)
# plt.show()


