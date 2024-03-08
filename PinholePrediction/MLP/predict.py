'''
Date: 2024-01-04 13:55:44
LastEditors: wind-listener 2775174829@qq.com
LastEditTime: 2024-03-06 22:52:27
FilePath: \PinholeAnalysis\PinholePrediction\MLP\predict.py
'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project  ：PinholeRegression 
@File     ：predict.py
@IDE      ：PyCharm 
@Author   ：user
@Email    : zzm_ai@bupt.edu.cn
@Date     ：2024/1/2 16:24 
@Function ：针对annotation，只时候MLP实验能不能完成回归
"""
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# 读取CSV文件
data = pd.read_csv('Dataset/annotation2.csv')

# 提取特征
X = data.drop(columns=['ImageID', 'Xcoordinate', 'Ycoordinate', 'Diameter'])

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 加载训练好的MLP模型
model_path = 'models/mlp_model_2024-03-06_22-47-00.pkl'
mlp = joblib.load(model_path)

# 进行预测
predictions = mlp.predict(X_scaled)

# 将预测结果添加到数据集中
data['Predicted_Diameter'] = predictions

# 保存带有预测结果的CSV文件
# 构建带有时间戳的文件名
csv_filename = 'predictions/annotation_with_predictions_' + os.path.basename(model_path)[10:-4] + '.csv'
data.to_csv(csv_filename, index=False)
print("预测完成！")