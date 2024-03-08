#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project  ：PinholeRegression 
@File     ：predict.py
@IDE      ：PyCharm 
@Author   ：user
@Email    : zzm_ai@bupt.edu.cn
@Date     ：2024/1/2 16:24 
@Function ：$END$
"""
import os.path
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from myutils import CustomDataset, PinholeRegressionModel

# 定义超参数
image_channels = 1
image_size = 128
num_features = 7
x_combined_length = 67
hidden_size = 64
output_size = 1
batch_size = 64


# # 加载模型
# model_path = 'models/model_2024-01-02_16-08-56/model.pth'  # 修改为实际的模型路径
# model = PinholeRegressionModel(image_channels, image_size, num_features, hidden_size, output_size,x_combined_length)
# model.load_state_dict(torch.load(model_path))
# model.eval()

# 加载模型
model_path = 'models\model_2024-01-04_11-33-03/model.pth'  # 修改为实际的模型路径
model = torch.load(model_path, map_location=torch.device('cpu'))  # 使用torch.load加载整个模型，并将其赋值给model
model.eval()

# 加载数据集
transform = transforms.Compose([transforms.Grayscale(num_output_channels=image_channels),
                                transforms.Resize((image_size, image_size)),
                                transforms.ToTensor()])

dataset = CustomDataset(csv_file='RegressionDataset/annotation.csv', root_dir='RegressionDataset/cropped_imgs',
                        transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 预测
predictions = []
truths = []
ids = []

with torch.no_grad():
    for batch in data_loader:
        images = batch['image']
        features = batch['features']
        labels = batch['label']

        outputs = model(images, features)
        predictions.extend(outputs.cpu().numpy().flatten())
        truths.extend(labels.numpy())
        ids.extend(batch['ImageID'].numpy())

# 保存结果到 CSV
result_df = pd.DataFrame({'ImageID': ids, 'Truth': truths, 'Prediction': predictions})
save_path = os.path.join(os.path.dirname(model_path), 'predictions.csv')
result_df.to_csv(save_path, index=False)

# 计算标准差
std_dev = result_df['Truth'].sub(result_df['Prediction']).std()
print(f'Standard Deviation between Truth and Prediction: {std_dev:.4f}')
