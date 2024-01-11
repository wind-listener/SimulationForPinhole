import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体或其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号


#----------------------------放大倍率与物料-镜头距离关系----------------------------------
# 读取 Excel 文件中的数据
excel_path = '放大倍率计算结果.xlsx'

# 读取“放大倍率-孔直径关系（高精度）” sheet
df_high_precision = pd.read_excel(excel_path, sheet_name='放大倍率-孔直径关系（高精度）', index_col=0)

# 选择所需的列
columns_high_precision = ['HighPrecisionTest1_10.96', 'HighPrecisionTest3_9.84']
df_high_precision_selected = df_high_precision[columns_high_precision]

# 读取“放大倍率-孔直径关系（中精度）” sheet
df_medium_precision = pd.read_excel(excel_path, sheet_name='放大倍率-孔直径关系（中精度）', index_col=0)

# 选择所需的列
columns_medium_precision = ['MediumPrecisionTest2_22.85', 'MediumPrecisionTest3_14.66']
df_medium_precision_selected = df_medium_precision[columns_medium_precision]

# 合并两个 DataFrame
df_combined = pd.concat([df_high_precision_selected, df_medium_precision_selected], axis=1)
# 修改列索引
new_columns = [400,220,485,320]
df_combined.columns = new_columns
df_combined = df_combined.sort_index(axis=1)
# 绘制折线图
for index, row in df_combined.iterrows():
    plt.plot(row, label='孔直径'+str(index) +'um', marker = 'o')

# 添加图例和标签
plt.legend()
plt.xlabel('物料-镜头距离')
plt.ylabel('放大倍率')
title = '放大倍率与物料-镜头距离关系'
plt.title(title)
output_path = 'Outputs\\'+ title +'.png'
plt.savefig(output_path, dpi=800)
# 显示图像
plt.show()

with pd.ExcelWriter('放大倍率计算结果.xlsx',  mode='a',engine='openpyxl') as writer:
    # 保存整个 DataFrame 到 Excel 文件
    df_combined.to_excel(writer,sheet_name=title,index=True,if_sheet_exists='replace')

