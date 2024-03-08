#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project  ：PinholeRegression 
@File     ：__init__.py.py
@IDE      ：PyCharm 
@Author   ：user
@Email    : zzm_ai@bupt.edu.cn
@Date     ：2024/1/2 14:57 
@Function ：$END$
"""

from myutils.CustomDataset import CustomDataset
from myutils.PinholeRegressionModel import PinholeRegressionModel

from torchvision import transforms

Debug = False
isEarlyStopping = True

def mirror_padding(img, target_size):
    width, height = img.size
    left_pad = max((target_size[0] - width) // 2, 0)
    top_pad = max((target_size[1] - height) // 2, 0)
    right_pad = max(target_size[0] - width - left_pad, 0)
    bottom_pad = max(target_size[1] - height - top_pad, 0)

    return transforms.functional.pad(img, (left_pad, top_pad, right_pad, bottom_pad), fill=0, padding_mode='constant')
