# '''
# Date: 2024-03-05 22:50:02
# LastEditors: wind-listener 2775174829@qq.com
# LastEditTime: 2024-03-09 18:37:31
# FilePath: \PinholeAnalysis\utils\getDiameterfromPosition.py
# '''
import math


def getDiameterfromPosition(x, region_width, margin, min_diameter):
    
    # x坐标减去margin，再除间隔，减0.5后向上取整，得到列数, 从0开始，0-9共十列
    column = math.ceil(max(0, x - margin) / region_width - 0.5)
    diameter = column%5 * 10 + min_diameter

    return diameter
