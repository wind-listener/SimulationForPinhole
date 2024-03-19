'''
Date: 2024-03-19 12:28:21
LastEditors: wind-listener 2775174829@qq.com
LastEditTime: 2024-03-19 14:27:23
FilePath: \PinholeAnalysis\PinholePrediction\RandomForest\predict.py
'''
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

csv_path = 'Dataset/annotation3.csv'
model_path = 'models/rf_regressor_model_2024-03-19_14-26-57.pkl'

# 读取CSV文件
data = pd.read_csv(csv_path)

# 提取特征
X = data.drop(columns=['ImageID', 'Diameter'])

# 加载模型
rf_regressor = joblib.load(model_path)

# 进行预测
predictions = rf_regressor.predict(X)

# 将预测结果添加到数据集中
data['Predicted_Diameter'] = predictions

# 保存带有预测结果的CSV文件
# 构建带有时间戳的文件名
csv_filename = (
    "predictions/annotation_with_predictions_"
    + os.path.basename(model_path)[19:-4]
    + ".csv"
)
data.to_csv(csv_filename, index=False)
print("预测完成！")

import matplotlib.pyplot as plt

# 提取真实直径和预测直径
true_diameter = data["Diameter"]
predicted_diameter = data["Predicted_Diameter"]

# 计算误差
errors = abs(predicted_diameter - true_diameter) / true_diameter
mean_error = errors.mean()
twenty_percentile_error = 0.2

# 创建主图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 绘制散点图
ax1.scatter(true_diameter, predicted_diameter, color="blue", alpha=0.5)
ax1.plot(
    [true_diameter.min(), true_diameter.max()],
    [true_diameter.min(), true_diameter.max()],
    "k--",
    lw=2,
)
ax1.set_xlabel("True Diameter")
ax1.set_ylabel("Predicted Diameter")
ax1.set_title("True Diameter vs Predicted Diameter")
ax1.grid(True)

# 绘制误差分布图
ax2.hist(errors, bins=30, color="green", alpha=0.7)
ax2.axvline(
    x=mean_error,
    color="red",
    linestyle="--",
    label=f"Mean Error:{mean_error * 100:.2f}%",
)
ax2.axvline(
    x=twenty_percentile_error, color="orange", linestyle="--", label="Max Error: 20%"
)
ax2.set_xlabel("Prediction Error")
ax2.set_ylabel("Frequency")
ax2.set_title("Prediction Error Distribution")
ax2.legend()

# 保存带有子图的图片
combined_pic_filename = (
    "predictions/combined_prediction_result_"
    + os.path.basename(model_path)[19:-4]
    + ".png"
)
plt.savefig(combined_pic_filename)
plt.show()

