import math

import pandas as pd
from matplotlib import pyplot as plt

from utils.SingleImageAnalysis import SingleImageAnalysis

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 使用黑体或其他支持中文的字体
plt.rcParams["axes.unicode_minus"] = False  # 用于正常显示负号

# 指定文件夹路径
dataset_path = "Dataset/ZZMImgs"
imgs = ["10", "20", "50", "100"]


def SingleExperimentAnalysis(dirs, title="", isMorphology=0, isGamma=0, mode=0):
    # 创建一个空的 DataFrame
    result_df = pd.DataFrame()
    # 创建两个figure
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for dir in dirs:
        true_area = []
        magnifications_1D = []
        for img in imgs:
            img_path = dataset_path + "/" + dir + "/" + img + ".bmp"

            mean_true_area, sqr_mean_magnification = SingleImageAnalysis(
                img_path, isMorphology=isMorphology, isGamma=isGamma, mode=0
            )

            magnifications_1D.append(sqr_mean_magnification)
            true_area.append(mean_true_area)
        equivalentDiameter = [math.sqrt(4 * area / math.pi) for area in true_area]
        # -------------------保存结果------------------------
        df_dir = pd.DataFrame(
            list(zip(magnifications_1D, true_area, equivalentDiameter)),
            columns=[dir + "mag", dir + "area", dir + "dia"],
            index=imgs,
        )
        result_df = pd.concat([result_df, df_dir], axis=1)
        # -------------------绘图------------------------
        ax1.plot(imgs, equivalentDiameter, marker="o", label=dir)
        sub_title = title + " 孔直径-整体光斑等效直径 关系"
        ax1.set_title(sub_title)
        ax1.set_xlabel("孔直径/um")
        ax1.set_ylabel("整体光斑等效直径/um")
        ax1.legend()
        output_path = "Outputs\\" + sub_title + ".png"
        fig1.savefig(output_path, dpi=800)

        ax2.plot(imgs, magnifications_1D, marker="o", label=dir)
        sub_title = title + " 孔直径-尺度放大倍率 关系"
        ax2.set_title(sub_title)
        ax2.set_xlabel("孔直径/um")
        ax2.set_ylabel("尺度放大倍率")
        ax2.legend()
        output_path = "Outputs\\" + sub_title + ".png"
        fig2.savefig(output_path, dpi=800)
    plt.show()

    with pd.ExcelWriter(
        "计算结果.xlsx", mode="a", engine="openpyxl", if_sheet_exists="replace"
    ) as writer:
        # 保存整个 DataFrame 到 Excel 文件
        result_df.to_excel(writer, sheet_name=title, index=True)

    #     # result_df = pd.concat([result_df,df],ignore_index=False)
    #     # 选择一个不同的颜色
    #     color = plt.cm.jet(dirs.index(dir) / len(dirs))
    #     # 绘制折线图
    #     plt.plot(imgs, magnifications_1D, marker='o',label=dir, color=color)
    # plt.title(title)
    # plt.xlabel('孔直径/um')
    # plt.ylabel('放大倍率')
    # plt.legend()
    # output_floder = 'Outputs\\'+ title+'.png'
    # plt.savefig(output_floder, dpi=800)
    # plt.show()
    #
