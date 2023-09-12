import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import os
import glob
import cv2 as cv
import matplotlib.pyplot as plt

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

# def save(total_loss, recon_loss_1, recon_loss_2, kl_loss):
#     plt.plot(total_loss, color='green', label='total loss')
#     plt.plot(recon_loss_1, color='red', label='loss of sigma_u')
#     plt.plot(recon_loss_2, color='blue', label='loss of sigma_v')
#     plt.plot(kl_loss, color='yellow', label='loss of sigma_v')
#     plt.xlabel('Epoch')
#     plt.ylabel('loss')
#     plt.yscale('log')
#     plt.title('Training Loss')
#     plt.savefig('loss-vae.png')    

# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(32 * (W // 4) * (H // 4), latent_dim)
        self.fc_logvar = nn.Linear(32 * (W // 4) * (H // 4), latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * (W // 4) * (H // 4)),
            nn.Unflatten(1, (32, W // 4, H // 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, output_dim, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
    
def custom_loss(sigma, v, v_t, device):
    sigma = sigma.squeeze(1)
    # print(sigma.shape)
    eps = torch.full((len(sigma), 256, 256), 1e-10).to(device)
    # print(eps.shape)
    # sigma2 = torch.square(sigma) + eps
    sigma = 3 * torch.abs(sigma) + eps
    loss_fn = nn.MSELoss()
    sigma_t = torch.abs(v - v_t)
    loss = loss_fn(sigma, sigma_t)
    # print(f"sigam2.shape: {sigma2.shape}, eps.shape: {eps.shape}, sigma.shape: {sigma.shape}, v_t.shape: {v_t.shape}, sigma_t.shape: {sigma2_t.shape}")
    
    return loss

# 定义训练函数
def train_vae(model, train_loader, num_epochs, learning_rate, device):
    
    model = model.to(device)
    # criterion = nn.MSELoss()  # 重构损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    total_loss_show = []
    recon_loss_1_show = []
    recon_loss_2_show = []
    kl_loss_show = []
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            
            data = data.to(device)
            inputs = data  # 输入数据的shape为（batch_size, 4, H, W）
            
            input_data = inputs[:, :4, :, :]

            v_u = inputs[:, 2, :, :]
            v_v = inputs[:, 3, :, :]
            v_t_u = inputs[:, 4, :, :]
            v_t_v = inputs[:, 5, :, :]
            
            input_data = remap(input_data, device)

            optimizer.zero_grad()

            # 前向传播
            recon_batch, mu, logvar = model(input_data)

            # 计算重构损失和KL散度
            # recon_loss = criterion(recon_batch, input_data)
            recon_loss_1 = custom_loss(recon_batch[:, 0, :, :], v_u, v_t_u, device)
            recon_loss_2 = custom_loss(recon_batch[:, 1, :, :], v_v, v_t_v, device)
            recon_loss = recon_loss_1 + recon_loss_2
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        recon_loss_1_show.append(recon_loss_1.item() / len(train_loader))
        recon_loss_2_show.append(recon_loss_2.item() / len(train_loader))
        kl_loss_show.append(kl_loss.item() / len(train_loader))
        total_loss_show.append(total_loss / len(train_loader))
        
         
        print('Epoch [{}/{}], Total Loss: {:.8f}, recon_loss_1: {:.8f}, recon_loss_2: {:.8f}, kl_loss: {:.8f}'.format(epoch + 1, num_epochs, total_loss, recon_loss_1, recon_loss_2, kl_loss))
        if epoch % 5 == 0:
            save_path = '/home/panding/code/UR/VAE-model/VAE-'+str(epoch)+'.pt'
            torch.save(model.state_dict(), save_path)
        plt.plot(total_loss_show, color='green', label='total loss')
        plt.plot(recon_loss_1_show, color='red', label='loss of sigma_u')
        plt.plot(recon_loss_2_show, color='blue', label='loss of sigma_v')
        plt.plot(kl_loss_show, color='yellow', label='loss of kl')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.yscale('log')
        plt.title('Training Loss')
        plt.savefig('VAE-loss.png')
    plt.plot(total_loss_show, color='green', label='total loss')
    plt.plot(recon_loss_1_show, color='red', label='loss of sigma_u')
    plt.plot(recon_loss_2_show, color='blue', label='loss of sigma_v')
    plt.plot(kl_loss_show, color='yellow', label='loss of kl')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.title('Training Loss')
    plt.savefig('VAE-loss.png')
    torch.save(model.state_dict(), '/home/panding/code/UR/VAE-model/VAE-model-new.pt')

# 设置输入和输出的尺寸
input_dim = 4
output_dim = 2
W = 256
H = 256

# 创建VAE模型实例
latent_dim = 10
vae = VAE(input_dim, latent_dim)


# 定义数据加载器
data_path = '/home/panding/code/UR/piv-data/ur'
batch_size = 20

my_data_loader = load_data(data_path, batch_size)

# 训练VAE模型
num_epochs = 360
learning_rate = 0.0005
my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_vae(vae, my_data_loader, num_epochs, learning_rate, my_device)