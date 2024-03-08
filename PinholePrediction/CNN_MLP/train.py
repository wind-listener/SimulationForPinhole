import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from myutils import Debug, CustomDataset, PinholeRegressionModel, mirror_padding, isEarlyStopping
import wandb
import os
import datetime

if not Debug:
    wandb.init(project='PinholeRegression')

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义超参数
image_channels = 1
image_size = 128
num_features = 8
hidden_size = 64
x_combined_length = 129 * hidden_size
output_size = 1
batch_size = 64
learning_rate = 0.001
num_epochs = 500
# L1正则化和L2正则化的权重参数
l1_lambda = 0.001  # 调整L1正则化的强度
l2_lambda = 0.001  # 调整L2正则化的强度

# 创建一个包含镜像填充的transforms.Compose
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=image_channels),
    transforms.Lambda(lambda img: mirror_padding(img, (image_size, image_size))),  # 使用自定义的填充函数
    transforms.CenterCrop((image_size, image_size)),
    transforms.ToTensor()
])

# 加载数据集并划分为训练集、验证集和测试集
dataset = CustomDataset(csv_file='RegressionDataset2/annotation.csv', root_dir='RegressionDataset2/cropped_imgs',
                        transform=transform)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建模型
model = PinholeRegressionModel(image_channels, image_size, num_features, hidden_size, output_size, x_combined_length)
model.to(device)

# 定义损失函数、优化器、学习率调度器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)  # 添加L2正则化
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

# 定义 Early Stopping 参数
patience = 20
min_delta = 0.0001
early_stopping_counter = 0
best_loss = float('inf')

# 训练模型
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        images = batch['image'].to(device)
        features = batch['features'].to(device)
        labels = batch['label'].float().to(device)

        # 正向传播
        outputs = model(images, features)
        loss = criterion(outputs, labels)

        # 添加L1正则化
        l1_reg = torch.tensor(0.0, device=device)
        for param in model.parameters():
            l1_reg += torch.norm(param, 1)  # L1正则化项
        loss += l1_lambda * l1_reg  # 添加L1正则化损失

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 使用 WandB 记录损失
        wandb.log({'train_loss': loss.item()})

    # 在验证集上评估模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_batch in val_loader:
            val_images = val_batch['image'].to(device)
            val_features = val_batch['features'].to(device)
            val_labels = val_batch['label'].float().to(device)

            val_outputs = model(val_images, val_features)
            val_loss += criterion(val_outputs, val_labels).item()

    val_loss /= len(val_loader)
    wandb.log({'val_loss': val_loss})

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

    # Early Stopping
    if isEarlyStopping:
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break

    # 更新学习率
    scheduler.step(val_loss)

# 在测试集上评估模型
model.eval()
test_loss = 0.0
with torch.no_grad():
    for test_batch in test_loader:
        test_images = test_batch['image'].to(device)
        test_features = test_batch['features'].to(device)
        test_labels = test_batch['label'].float().to(device)

        test_outputs = model(test_images, test_features)
        test_loss += criterion(test_outputs, test_labels).item()

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}')

# 保存整个模型并将其移回CPU
model.to('cpu')  # 将模型移到CPU
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
model_dir = os.path.join('models', f'model_{current_time}')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'model.pth')
torch.save(model, model_path)
print(f'Model saved at: {model_path}')
