import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset

import numpy as np
import os


# 包含四个下采样层和四个上采样层的 U-Net 结构
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # 定义下采样层
        self.down = nn.ModuleList()
        self.down.append(self._make_conv_block(in_channels, 64))
        self.down.append(self._make_conv_block(64, 128))
        self.down.append(self._make_conv_block(128, 256))
        self.down.append(self._make_conv_block(256, 512))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义上采样层
        self.up = nn.ModuleList()
        self.up.append(self._make_upconv(512, 256))
        self.up.append(self._make_conv_block(256 + 256, 256))
        self.up.append(self._make_upconv(256, 128))
        self.up.append(self._make_conv_block(128 + 128, 128))
        self.up.append(self._make_upconv(128, 64))
        self.up.append(self._make_conv_block(64 + 64, 64))
        self.up.append(self._make_upconv(64, 32))
        self.up.append(self._make_conv_block(32 + 32, 32))

        # 定义输出层
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for downsample in self.down:
            x = downsample(x)
            skips.append(x)
            x = self.pool(x)

        skips = reversed(skips[:-1])
        for i, upsample in enumerate(self.up):
            skip = next(skips)
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)

        # 输出 sigma 平方值
        sigma2 = self.out(x)

        # 从后两个通道的目标张量中提取真实的运动场 v_t
        v_t = x[:, 4:6, :, :]

        return sigma2, v_t

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _make_upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


def custom_loss(sigma_sq, v, v_t):
    loss = -0.5 * torch.log(sigma_sq) - (v_t - v) ** 2 / (2 * sigma_sq)
    return torch.mean(loss)


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_files = sorted(os.listdir(self.data_path))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_path, self.data_files[idx]))
        target = data[:, :, -2:]  # 取最后两个通道作为目标张量
        input_data = data[:, :, :-2]  # 取前面的通道作为输入张量
        input_data = torch.from_numpy(input_data).float()
        target = torch.from_numpy(target).float()
        return input_data, target


def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        sigma2, v_t = model(inputs)
        loss = custom_loss(sigma2, v_t, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    average_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, average_loss))


if __name__ == '__main__':
    # 定义超参数
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1e-3
    MOMENTUM = 0.9

    # 加载数据集
    train_dataset = MyDataset('data/train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型和优化器
    model = UNet(in_channels=6, out_channels=2)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    # 进行训练
    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer, epoch)