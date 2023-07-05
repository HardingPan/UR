import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np
from torch.utils.data import DataLoader, Dataset

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
根据 U-Net 的结构定义了一个包含输入通道数为 4、输出通道数为 1 的 `UNet` 模型
并使用 Adam 优化器对其进行训练
在训练和测试函数中，我们只需要将 U-Net 的输出 `sigma` 送入损失函数 `custom_loss` 中即可。
"""

# 用于实现两个卷积层和一个批归一化层的组合
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)
    
# 解码器由四个上采样模块 `Up` 组成，每个上采样模块包含一个上采样层和两个卷积层
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # pad x1 to match x2 size
        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_w//2, diff_w-diff_w//2, diff_h//2, diff_h-diff_h//2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
"""
在 `UNet` 类的 `forward` 方法中
首先将输入数据 `x` 送入编码器中，然后将编码器的输出 `x5` 送入解码器中。
在解码器中，我们将 `x5` 与编码器中的输出 `x4`、`x3`、`x2`、`x1` 依次进行拼接，
并送入对应的上采样模块中进行上采样和卷积操作
最终输出一个概率分布中的sigma^2。
"""
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        sigma = torch.sigmoid(x)
        return sigma
    
"""
`sigma` 是模型输出的标准差，
`v_t` 是目标值（即下一帧图像），
`v` 是模型输出的预测值（即当前帧图像加上运动分量）
在 `test` 函数中，需要计算模型在测试集上的损失。
由于测试集中没有目标值 `v_t`，因此我们需要用当前帧图像 `x_t` 作为目标值进行计算。
这样，公式中的 `v_t` 就是当前帧图像 `x_t`，而 `v` 则是模型在当前帧图像 `x_t` 上的预测值
即模型输出的mu。
"""
# 定义损失函数
def custom_loss(sigma, mu, v_t, v):
    loss = -0.5 * torch.log(sigma ** 2) - 0.5 * (v_t - v) ** 2 / sigma ** 2
    return loss.mean()

"""
在下面的的代码中，我们首先使用 `inputs[:, :3, :, :] + inputs[:, 3, :, :].unsqueeze(1)` 
得到模型在当前帧图像上的预测值 `v`。
然后，我们将 `sigma`、`mu` 和 `v` 送入损失函数 `custom_loss` 中进行计算，
得到模型在当前帧图像上的损失 `loss`。最后，我们将所有测试集上的损失相加并平均，得到模型在测试集上的平均损失。
"""
# 定义训练函数
def train(net, optimizer, train_loader, device):
    net.train()
    running_loss = 0.0
    for inputs in train_loader:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        sigma = net(inputs)
        mu = inputs[:, :3, :, :] + inputs[:, 3, :, :].unsqueeze(1)
        v_t = inputs[:, :3, :, :]
        v = mu
        loss = custom_loss(sigma, mu, v_t, v)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

"""
在 `test` 函数中先计算模型在当前帧图像上的输出 `mu`
然后将其与当前帧图像 `x_t` 进行相加
得到模型在当前帧图像上的预测值 `v`。这样，就可以使用公式计算模型在当前帧图像上的损失了
"""
# 定义测试函数
def test(net, test_loader, device):
    net.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            sigma = net(inputs)
            mu = inputs[:, :3, :, :] + inputs[:, 3, :, :].unsqueeze(1)
            v_t = inputs[:, :3, :, :]
            v = mu
            loss = custom_loss(sigma, mu, v_t, v)
            running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(test_loader.dataset)
    return epoch_loss

# 训练和测试模型
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set hyperparameters
    lr = 0.0005
    num_epochs = 10
    batch_size = 1

    # Load data
    path = '/Users/panding/code/ur/UR/data'
    train_loader = load_data(path, batch_size)
    test_loader = load_data(path, batch_size)

    # Create model
    net = UNet(n_channels=4, n_classes=1).to(device)

    # Create optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Train model
    for epoch in range(num_epochs):
        
        train_loss = train(net, optimizer, train_loader, device)
        test_loss = test(net, test_loader, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    # Save model
    torch.save(net.state_dict(), 'model.pt')
    
if "__name__" == "__main__":
    main()