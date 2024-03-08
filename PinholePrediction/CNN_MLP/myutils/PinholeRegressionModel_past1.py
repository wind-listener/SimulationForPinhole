import torch
import torch.nn as nn
import torch.nn.functional as F

class PinholeRegressionModel(nn.Module):
    def __init__(self, image_channels, image_size, num_features, hidden_size, output_size):
        super(PinholeRegressionModel, self).__init__()

        # 图像处理部分
        self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc_image = nn.Linear(32 * image_size * image_size, hidden_size)

        # 特征参数处理部分
        self.fc_features = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.BatchNorm1d(hidden_size),  # 添加 Batch Normalization
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # 添加 Batch Normalization
            nn.ReLU(),
            nn.Linear(hidden_size, 3)  # 输出长度设置为3
        )

        # 联合处理部分
        self.combined_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),  # 添加 Batch Normalization
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # 添加 Batch Normalization
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # 添加 Batch Normalization
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # 添加 Batch Normalization
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # 添加 Batch Normalization
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x_image, x_features):
        # 图像处理
        x_image = F.relu(self.conv1(x_image))
        x_image = F.relu(self.conv2(x_image))
        x_image = x_image.view(x_image.size(0), -1)  # 展平
        x_image = F.relu(self.fc_image(x_image))

        # 特征参数处理
        x_features = self.fc_features(x_features)  # 使用新的特征处理层

        # 合并图像和特征参数
        x_combined = torch.cat((x_image, x_features), dim=1)

        # 联合处理，使用nn.Sequential
        output = self.combined_layers(x_combined)

        return output
