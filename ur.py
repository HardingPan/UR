import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

"""
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
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
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=2):
        super(UNet, self).__init__()

        # Down sampling
        self.conv1 = double_conv(in_channels, 64)
        self.conv2 = double_conv(64, 128)
        self.conv3 = double_conv(128, 256)
        self.conv4 = double_conv(256, 512)

        # Up sampling
        self.up_conv1 = up_conv(512, 256)
        self.conv5 = double_conv(512, 256)
        self.up_conv2 = up_conv(256, 128)
        self.conv6 = double_conv(256, 128)
        self.up_conv3 = up_conv(128, 64)
        self.conv7 = double_conv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Down sampling
        x1 = self.conv1(x)
        x2 = self.conv2(F.max_pool2d(x1, 2))
        x3 = self.conv3(F.max_pool2d(x2, 2))
        x4 = self.conv4(F.max_pool2d(x3, 2))

        # Up sampling
        x = self.up_conv1(x4)
        x = torch.cat([x3, x], dim=1)
        x = self.conv5(x)
        x = self.up_conv2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv6(x)
        x = self.up_conv3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv7(x)

        x = self.final_conv(x)
        return x
"""

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
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
            nn.ReLU(inplace=True)
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
        
        return sigma_u, sigma_v


def remap(inputs, device):
    # inputs = inputs.numpy()
    
    image0 = inputs[0]
    image1 = inputs[1]
    u = inputs[2]
    v = inputs[3]
    
    x, y = np.meshgrid(np.arange(image0.shape[1]), np.arange(image0.shape[0]))
    x = np.float32(x)
    y = np.float32(y)
    image0 = cv.remap(image0, x+u, y+v, interpolation = 4)
    
    inputs = torch.from_numpy(inputs)
    
    return inputs.to(device)

# def ur(data, path, device):
    
#     data = remap(data, device)
#     data = data.unsqueeze(0)
#     print("remaped!")
#     model = torch.load(path)
#     model = UNet(in_channels=4, out_channels=1)
#     model.load_state_dict(torch.load(path))
#     model.to(device)
#     print("model has loaded!")
#     sigma = model(data)
#     print("sigma has got!!")
    
#     return sigma.squeeze(0, 1)

class Ur():
    def __init__(self, data, path, device):
        
        self.data = remap(data, device)
        self.data = self.data.unsqueeze(0)
        # print(self.data.shape)
        
        model = torch.load(path)
        model = UNet(in_channels=4, out_channels=1)
        model.load_state_dict(torch.load(path))
        model.to(device)
        
        self.sigma_u, self.sigma_v = model(self.data)
        
        self.sigma_u = self.sigma_u.squeeze(0).cpu().detach().numpy()
        self.sigma_v = self.sigma_v.squeeze(0).cpu().detach().numpy()
        print('completed!')
    def get_sigma(self):
        return self.sigma_u, self.sigma_v

if __name__ == '__main__':
    
    # 加载数据
    data_path = '/home/panding/code/UR/piv-data/ur/backstep_Re800_00361.npy'
    data = np.load(data_path)
    data = data[:4]
    # data_tensor = torch.from_numpy(data)
    
    model_path = '/home/panding/code/UR/UR/ur-model/8-21-1.pt'
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test = Ur(data, model_path, my_device)
    sigma = test.get_sigma()
    