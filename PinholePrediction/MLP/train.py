"""
Date: 2024-01-04 13:57:12
LastEditors: wind-listener 2775174829@qq.com
LastEditTime: 2024-03-09 20:11:20
FilePath: \PinholeAnalysis\PinholePrediction\MLP\train.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib
from datetime import datetime

# 读取CSV文件
data = pd.read_csv("Dataset/annotation2.csv")

# 提取特征和标签
X = data.drop(columns=["ImageID", "Xcoordinate", "Ycoordinate", "Diameter"])
y = data["Diameter"]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# (128, 128, 64, 64, 64, 32, 16, 8)
# 训练集得分: 0.79
# 测试集得分: 0.71

# (128, 64, 64, 32, 16, 8)
# 训练集得分: 0.75
# 测试集得分: 0.68

# (64, 32, 16, 8)
# 训练集得分: 0.76
# 测试集得分: 0.69

# 创建MLP回归模型
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16, 8),
    max_iter=2000,
    random_state=42,
    learning_rate="adaptive",
    early_stopping=True,
    validation_fraction=0.2,
)

# 训练模型
mlp.fit(X_train, y_train)

# 评估模型
train_score = mlp.score(X_train, y_train)
test_score = mlp.score(X_test, y_test)

print(f"训练集得分: {train_score:.2f}")
print(f"测试集得分: {test_score:.2f}")

# 保存模型
# 获取当前日期和时间作为文件名的一部分
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# 构建带有时间戳的文件名
model_filename = f"models/mlp_model_{current_time}.pkl"
joblib.dump(mlp, model_filename)
