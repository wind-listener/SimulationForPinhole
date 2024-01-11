import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体或其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

# 读取 Excel 文件中的三个 sheet
file_path = "放大倍率计算结果.xlsx"
sheets = ["放大倍率-孔直径关系（高精度）", "放大倍率-孔直径关系（中精度）", "放大倍率-孔直径关系（低精度）"]

# 读取并拼接数据
dfs = []
for sheet in sheets:
    df = pd.read_excel(file_path, sheet_name=sheet, index_col=0)
    dfs.append(df)

df_combined = pd.concat(dfs, axis=1)

# 提取列索引中的数字
num_columns =[]
for column in df_combined.columns:

    num_columns.append(float(column.split('_')[-1]))
df_combined.columns = num_columns

# 按列索引大小排序
df_combined = df_combined.sort_index(axis=1)

# 按行在同一坐标中绘制曲线
plt.figure(figsize=(10, 6))
title ="放大倍率-精度（原始数据）"
plt.title(title)
for index, row in df_combined.iterrows():
    # plt.xscale('log')
    plt.plot(df_combined.columns, row, label=index, marker = 'o')

plt.xlabel('精度um/pixel')
plt.ylabel('放大倍率')
plt.legend()
output_path = 'Outputs\\'+ title+'.png'
plt.savefig(output_path, dpi=800)
plt.show()

# 使用 Z-score 方法剔除异常数据
z_scores = stats.zscore(df_combined, axis=0)
threshold = 1  # 设置阈值，可以根据需要调整
df_combined = df_combined[(z_scores < threshold).all(axis=1)]
# 按行在同一坐标中绘制曲线
plt.figure(figsize=(10, 6))
title= "放大倍率-精度（Z-score剔除异常据）"
plt.title(title)
for index, row in df_combined.iterrows():
    # plt.xscale('log')
    plt.plot(df_combined.columns, row, label=index, marker = 'o')
plt.xlabel('精度um/pixel')
plt.ylabel('放大倍率')
plt.legend()
output_path = 'Outputs\\'+ title+'.png'
plt.savefig(output_path, dpi=800)
plt.show()


