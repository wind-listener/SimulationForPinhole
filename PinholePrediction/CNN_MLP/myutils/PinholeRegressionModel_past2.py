import torch
import torch.nn as nn
import torch.nn.functional as F


class PinholeRegressionModel(nn.Module):
    def __init__(self, image_channels, image_size, num_features, hidden_size, output_size, x_combined_length):
        super(PinholeRegressionModel, self).__init__()

        # 图像处理部分
        self.imgFeatureExtractor = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Flatten()
        )

        # 特征参数处理部分
        self.outside_features = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.BatchNorm1d(hidden_size),  # 添加 Batch Normalization
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # 联合处理部分
        self.combined_layers = nn.Sequential(
            nn.Linear(x_combined_length, 16*hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(16*hidden_size, 8*hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(8 * hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4 * hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x_image, x_features):
        # 图像特征提取
        x_image = self.imgFeatureExtractor(x_image)

        # 外部特征处理（如曝光时间等成像因素）
        x_features = self.outside_features(x_features)

        # 合并图像和特征参数
        x_combined = torch.cat((x_image, x_features), dim=1)
        # print(x_combined.size(1))

        # 联合处理，使用nn.Sequential
        output = self.combined_layers(x_combined)

        return output
