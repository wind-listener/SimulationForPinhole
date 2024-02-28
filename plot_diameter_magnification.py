from utils.SingleExperimentAnalysis import SingleExperimentAnalysis

# 获取文件夹中的所有文件
# dirs = os.listdir(dataset_path)

title = "可靠实验数据分析-2.4um"
dirs = [
    "HighPrecisionTest4_8.70",
    "HighPrecisionTest6_8.54",
    "MediumPrecisionTest2_22.85",
    "MediumPrecisionTest3_14.66",
    "Repeat1_10.20",
    "Repeat2_16.24",
    "Repeat3_20.85",
    "Repeat4_36.64",
]
SingleExperimentAnalysis(dirs, title)
