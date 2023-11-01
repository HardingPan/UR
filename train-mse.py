import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import cv2 as cv
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.01,inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(negative_slope=0.01,inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(negative_slope=0.01,inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(UNet, self).__init__()

        # Down sampling
        self.conv1 = double_conv(in_channels, 64)
        self.conv2 = double_conv(64, 128)
        self.conv3 = double_conv(128, 256)
        self.conv4 = double_conv(256, 512)
        
        self.conv1_1 = double_conv(in_channels, 64)
        self.conv2_1 = double_conv(64, 128)
        self.conv3_1 = double_conv(128, 256)
        self.conv4_1 = double_conv(256, 512)
        
        self.conv1_2 = double_conv(in_channels, 64)
        self.conv2_2 = double_conv(64, 128)
        self.conv3_2 = double_conv(128, 256)
        self.conv4_2 = double_conv(256, 512)

        # Up sampling
        # self.up_conv1 = up_conv(512, 256)
        # self.conv5 = double_conv(512, 256)
        # self.up_conv2 = up_conv(256, 128)
        # self.conv6 = double_conv(256, 128)
        # self.up_conv3 = up_conv(128, 64)
        # self.conv7 = double_conv(128, 64)
        
        self.up_conv1_1 = up_conv(512, 256)
        self.conv5_1 = double_conv(512, 256)
        self.up_conv2_1 = up_conv(256, 128)
        self.conv6_1 = double_conv(256, 128)
        self.up_conv3_1 = up_conv(128, 64)
        self.conv7_1 = double_conv(128, 64)
        
        self.up_conv1_2 = up_conv(512, 256)
        self.conv5_2 = double_conv(512, 256)
        self.up_conv2_2 = up_conv(256, 128)
        self.conv6_2 = double_conv(256, 128)
        self.up_conv3_2 = up_conv(128, 64)
        self.conv7_2 = double_conv(128, 64)

        self.final_conv_1 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.final_conv_2 = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Down sampling
        x = F.normalize(x, dim=1)
        x1 = self.conv1(x)
        x2 = self.conv2(F.max_pool2d(x1, 2))
        x3 = self.conv3(F.max_pool2d(x2, 2))
        x4 = self.conv4(F.max_pool2d(x3, 2))

        # Up sampling
        # x = self.up_conv1(x4)
        # x = torch.cat([x3, x], dim=1)
        # x = self.conv5(x)
        # x = self.up_conv2(x)
        # x = torch.cat([x2, x], dim=1)
        # x = self.conv6(x)
        # x = self.up_conv3(x)
        # x = torch.cat([x1, x], dim=1)
        # x = self.conv7(x)
        
        sigma_u = self.up_conv1_1(x4)
        sigma_u = torch.cat([x3, sigma_u], dim=1)
        sigma_u = self.conv5_1(sigma_u)
        sigma_u = self.up_conv2_1(sigma_u)
        sigma_u = torch.cat([x2, sigma_u], dim=1)
        sigma_u = self.conv6_1(sigma_u)
        sigma_u = self.up_conv3_1(sigma_u)
        sigma_u = torch.cat([x1, sigma_u], dim=1)
        sigma_u = self.conv7_1(sigma_u)
        
        sigma_v = self.up_conv1_2(x4)
        sigma_v = torch.cat([x3, sigma_v], dim=1)
        sigma_v = self.conv5_2(sigma_v)
        sigma_v = self.up_conv2_2(sigma_v)
        sigma_v = torch.cat([x2, sigma_v], dim=1)
        sigma_v = self.conv6_2(sigma_v)
        sigma_v = self.up_conv3_2(sigma_v)
        sigma_v = torch.cat([x1, sigma_v], dim=1)
        sigma_v = self.conv7_2(sigma_v)

        sigma_u = self.final_conv_1(sigma_u)
        sigma_v = self.final_conv_2(sigma_v)
        
        # sigma_u = torch.clip(sigma_u, 0.0, 3.0)
        # sigma_v = torch.clip(sigma_v, 0.0, 3.0)
        
        # sigma_u = F.normalize(sigma_u, dim=1)
        # sigma_v = F.normalize(sigma_v, dim=1)
        
        return sigma_u, sigma_v

def custom_loss(sigma, v, v_t, device):
    sigma = sigma.squeeze(1)
    eps = torch.full((len(sigma), 256, 256), 1e-10).to(device)
    # sigma2 = torch.square(sigma) + eps
    sigma = 3 * torch.abs(sigma) + eps
    loss_fn = nn.MSELoss()
    sigma_t = torch.abs(v - v_t)
    loss = loss_fn(sigma, sigma_t)
    # print(f"sigam2.shape: {sigma2.shape}, eps.shape: {eps.shape}, sigma.shape: {sigma.shape}, v_t.shape: {v_t.shape}, sigma_t.shape: {sigma2_t.shape}")
    
    return loss
    
class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_files = glob.glob(os.path.join(self.data_path, '*.npy'))
        # self.data_files = sorted(self.data_path)
        randomidx = np.random.permutation(len(self.data_files))
        self.data_files = [self.data_files[i] for i in randomidx]
        # print(self.data_files)
        
    def __len__(self):
        return len(self.data_files)

    """
    `__getitem__` 方法会在每次加载一个数据时被调用，
    它会从指定路径中读取 `.npy` 文件，并将其转换为一个 PyTorch 张量。
    然后，使用 PyTorch 提供的 `DataLoader` 类，将数据划分为批次进行训练。
    """
    def __getitem__(self, index):
        # Load data from file
        # data = np.load(os.path.join(self.data_path, self.data_files[index]))
        data = np.load(self.data_files[index])
        # data = data[0:4]
        # Convert to tensor
        data = torch.from_numpy(data).float()
        return data
    
def load_data(data_path, batch_size):
    # Create data loader
    dataset = MyDataset(data_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader

def remap(inputs, device):
    inputs = inputs.cpu().numpy()
    N = inputs.shape[0]
    inputs_split_list = np.split(inputs, N, axis=0)
    inputs_split_list = [np.squeeze(i, axis=0) for i in inputs_split_list]
    # print(inputs_split_list[0].shape)
    for i in range(N):
        img0 = inputs_split_list[i][0]
        img1 = inputs_split_list[i][1]
        u = inputs_split_list[i][2]
        v = inputs_split_list[i][3]

        x, y = np.meshgrid(np.arange(img1.shape[1]), np.arange(img1.shape[0]))
        x = np.float32(x)
        y = np.float32(y)
        img0 = cv.remap(img0, x+u, y+v, interpolation = 4)
        
    inputs_new = np.stack(inputs_split_list, axis = 0)
    inputs_new = torch.from_numpy(inputs_new)

    return inputs_new.to(device)

def train(model, optimizer, data_loader, num_epochs, save_name, device):
    
    model.to(device)
    losses = []
    losses_u = []
    losses_v = []
    txt_name = '/home/panding/code/UR/ur-model/'+save_name+'.txt'
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_metric_u = 0.0
        epoch_metric_v = 0.0
        num_batches = 0

        for batch in data_loader:
            batch = batch.to(device)

            inputs = batch[:, :4, :, :]

            v_u = batch[:, 2, :, :]
            v_v = batch[:, 3, :, :]
            v_t_u = batch[:, 4, :, :]
            v_t_v = batch[:, 5, :, :]
            
            inputs = remap(inputs, device)
            # 将梯度清零
            optimizer.zero_grad()
            # 前向传递
            sigma_u, sigma_v = model(inputs)
            # 计算损失和评估指标
            loss_u = custom_loss(sigma_u, v_u, v_t_u, device)
            loss_v = custom_loss(sigma_v, v_v, v_t_v, device)
            
            metric_u = loss_u.item()
            metric_v = loss_v.item()
            
            loss = loss_u + loss_v
            
            # 反向传播和优化
            loss.backward()
            # if epoch % 2 == 0:
            #     loss_v.backward()
            # else:
            #     loss_u.backward()
            optimizer.step()
            # 更新损失和评估指标
            epoch_loss += loss.item()
            epoch_metric_u += metric_u
            epoch_metric_v += metric_v
            num_batches += 1

        # 计算平均损失和评估指标
        avg_loss = epoch_loss / num_batches
        avg_metric_u = epoch_metric_u / num_batches
        avg_metric_v = epoch_metric_v / num_batches
        
        losses.append(avg_loss)
        losses_u.append(avg_metric_u)
        losses_v.append(avg_metric_v)
        # 打印训练进度
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.8f}, Metric_u={avg_metric_u:.8f}, Metric_v={avg_metric_v:.8f}")
        f = open(txt_name,'a')
        f.write(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.8f}, Metric_u={avg_metric_u:.8f}, Metric_v={avg_metric_v:.8f}")
        f.write('\n')   
        f.close()
        save_path = '/home/panding/code/UR/ur-model/' + save_name + '.pth'
        torch.save(model.state_dict(), save_path)
    plt.plot(losses, color='green', label='total loss')
    plt.plot(losses_u, color='red', label='loss of sigma_u')
    plt.plot(losses_v, color='blue', label='loss of sigma_v')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.title('Training Loss')
    plt.savefig('/home/panding/code/UR/ur-model/'+save_name+'.png')    
    save_path = '/home/panding/code/UR/ur-model/' + save_name + '.pth'
    torch.save(model.state_dict(), save_path)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--num', type=int, default=400)
    args = parser.parse_args()
    
    # 加载数据
    data_path = '/home/panding/code/UR/piv-data/unflownet-to-train-muenn'
    batch_size = args.bs
    learning_rate = args.lr

    my_data_loader = load_data(data_path, batch_size)

    # 初始化模型、优化器和设备
    net = UNet(in_channels=4, out_channels=1)
    Adam_optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练循环
    my_num_epochs = args.num
    time = str(datetime.now())
    time = time.split(' ')[0]+'-'+time.split(' ')[1][:8]
    save_name = 'MSE' + '-' + time  + '-' + str(batch_size) + '-' + str(learning_rate) + '-' + str(my_num_epochs)
    print(f"--------- model is training: {save_name} ---------")
    
    train(model=net, optimizer=Adam_optimizer, data_loader=my_data_loader, num_epochs=my_num_epochs, save_name=save_name, device=my_device)
