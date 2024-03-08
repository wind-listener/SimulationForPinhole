#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project  ：PinholeRegression 
@File     ：CustomDataset.py
@IDE      ：PyCharm 
@Author   ：user
@Email    : zzm_ai@bupt.edu.cn
@Date     ：2024/1/2 14:58 
@Function ：$END$
"""
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx, 0]
        img_path = f"{self.root_dir}/{img_name}.png"
        image = Image.open(img_path).convert("L")  # 转为灰度图

        features = self.annotations.iloc[idx, 1:9].values.astype('float32')  # 特征在CSV的第1到第8列
        label = self.annotations.iloc[idx, 11]  # 标签在CSV的第11列

        if self.transform:
            image = self.transform(image)

        return {'ImageID': img_name, 'image': image, 'features': features, 'label': label}
