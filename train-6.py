import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset

import numpy as np
import os

# 包含四个下采样层和四个上采样层的U-Net结构
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # 定义下采样层
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 定义上采样层
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # 定义输出层
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        # 只使用前四个通道进行下采样和上采样
        x1 = self.down1(x[:, :4, :, :])
        x2 = self.pool1(x1)
        x2 = self.down2(x2)
        x3 = self.pool2(x2)
        x3 = self.down3(x3)
        x4 = self.pool3(x3)
        x4 = self.down4(x4)
        x5 = self.pool4(x4)
        
        x = self.up4(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        
        # 输出sigma平方值
        sigma2 = self.out(x)
        
        # 从后两个通道的目标张量中提取真实的运动场v_t
        v_t = x[:, 4:6, :, :]
        
        return sigma2, v_t

def custom_loss(sigma_sq, v, v_t):
    loss = -0.5 * torch.log(sigma_sq) - (v_t - v) ** 2 / (2 * sigma_sq)
    return loss.mean()

class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_files = sorted(os.listdir(self.data_path))

    def __len__(self):
        return len(self.data_files)

    """
    `__getitem__` 方法会在每次加载一个数据时被调用，
    它会从指定路径中读取 `.npy` 文件，并将其转换为一个 PyTorch 张量。
    然后，使用 PyTorch 提供的 `DataLoader` 类，将数据划分为批次进行训练。
    """
    def __getitem__(self, index):
        # Load data from file
        data = np.load(os.path.join(self.data_path, self.data_files[index]))
        # data = data[0:4]
        # Convert to tensor
        data = torch.from_numpy(data).float()
        return data

def load_data(data_path, batch_size):
    # Create data loader
    dataset = MyDataset(data_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


"""
------------------------------训练部分------------------------------
"""
# 加载数据
data_path = '/Users/panding/code/ur/UR/data'
batch_size = 2

data_loader = load_data(data_path, batch_size)

# 初始化模型、优化器和设备
model = UNet(in_channels=6, out_channels=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练循环
num_epochs = 10

def train(model, optimizer, data_loader, num_epochs, device):
    model.to(device)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_metric = 0.0
        num_batches = 0

        for batch in data_loader:
            # 将数据加载到设备上
            batch = batch.to(device)

            # 将前两个通道作为输入，后两个通道作为目标输出
            inputs = batch[:, :2, :, :]
            targets = batch[:, 2:4, :, :]

            # 将真实的运动场作为额外的目标输出
            v_t = batch[:, 4:6, :, :]

            # 将梯度清零
            optimizer.zero_grad()

            # 前向传递
            sigma2, v = model(inputs)

            # 计算损失和评估指标
            loss = custom_loss(sigma2, v, targets)
            metric = -loss.item()

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 更新损失和评估指标
            epoch_loss += loss.item()
            epoch_metric += metric
            num_batches += 1

        # 计算平均损失和评估指标
        avg_loss = epoch_loss / num_batches
        avg_metric = epoch_metric / num_batches

        # 打印训练进度
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Metric={avg_metric:.4f}")