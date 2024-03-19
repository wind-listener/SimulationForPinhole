# '''
# Date: 2024-03-08 22:46:00
# LastEditors: wind-listener 2775174829@qq.com
# LastEditTime: 2024-03-08 22:46:23
# FilePath: \PinholeAnalysis\utils\plot_text_on_mask.py
# '''
import cv2


def plot_text_on_mask(mask, text, label, centroid):
    text_lines = text.split("\n")  # 按换行符分割文本
    # text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    # # 调整文本的垂直位置
    # line_height = text_size[1]  # 计算文本行高度
    # total_height = len(text_lines) * line_height  # 计算总高度
    # centroid_y = int(centroid[1] + 1.2 * total_height / 2)
    
    line_height = 0

    # 逐行绘制文本
    for line in text_lines:
        text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        centroid_x = int(centroid[0] - text_size[0] / 4)
        # centroid_x = int(centroid[0])
        line_height += 1.1*text_size[1]
        centroid_y = int(centroid[1] + line_height)
        cv2.putText(
            mask, line, (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1
        )
