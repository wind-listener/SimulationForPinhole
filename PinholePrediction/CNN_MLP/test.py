#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project  ：PinholeRegression 
@File     ：test.py
@IDE      ：PyCharm 
@Author   ：user
@Email    : zzm_ai@bupt.edu.cn
@Date     ：2024/1/2 15:04 
@Function ：$END$
"""
# test.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from myutils import CustomDataset,PinholeRegressionModel

# 加载数据集
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.Resize((64, 64)),
                                transforms.ToTensor()])

test_dataset = CustomDataset(csv_file='annotation.csv', root_dir='images/', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 创建模型
model = ImageFeatureModel(image_channels=1, image_size=64, num_features=5, hidden_size=64, output_size=1)

# 加载已经训练好的模型参数
model.load_state_dict(torch.load('model.pth'))

# 测试模型
model.eval()
with torch.no_grad():
    for batch in test_loader:
        images = batch['image']
        features = batch['features']
        labels = batch['label']

        # 正向传播
        outputs = model(images, features)

        # 打印预测结果和真实值
        print(f'Predicted: {outputs.item():.4f}, True: {labels.item():.4f}')
