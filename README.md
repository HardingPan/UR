# UR(uncertainty of raft)

## 环境配置
python=`3.11`
```python
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install opencv-python
conda install matplotlib
conda install scipy
```
***
## 算法概述    
对RAFT的光流预测的置信度进行估计。
### Raft部分
输入：连续帧 $I_1$ 和 $I_2$   
输出：计算得到的速度场 $\widehat{v}$    
ground truth： $v_t$
### 预处理部分
![ur](ur.png)
将image1，image2，u，v，u_t, v_t进行拼接，这部分的数据集使用的是flyingchair
### Uncertainty部分
取一个期望为 $\mu$ 、方差为 ${\sigma}^2$ 的正态分布 $X \sim N(\mu, \sigma)$ ：  

$$
f(x)=\frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{(x-\mu)^2}{2 \sigma^2}}
$$   

对上式取对数：  

$$
\ln f(x)=-\frac{1}{2}\ln\big(2\pi {\sigma}^2\big)+\ln\big(-\frac{(x-\mu)^2}{2{\sigma}^2}\big)
$$      

化简后需要优化函数：   
$$loss=-\bigg(\frac{1}{2}ln{\sigma}^2+\frac{(v_t-\widehat{v})^2}{2{\sigma}^2+\xi}\bigg)$$
使loss最小，即可完成网络的训练。   
***

## 程序部分
### data
本项目的data基于piv数据
#### uniform
```
raft: 1 - 600
ur: 601 - 900
test: 901 - 1000
```
#### backstep
```
raft: Re800 1 - 300, Re1000 1 - 300, Re1200 1 - 300
ur: Re800 301 - 450, Re1000 301 - 450, Re1200 301 - 450
test: Re800 451 - 500, Re1000 451 - 500, Re1200 451 - 500
```
#### SQG
```
raft: 1 - 600
ur: 601 - 900
test: 901 - 1000
```
#### total
```
raft: 2100
ur: 1050
test: 350
```
#### input
通过`dataset-chair.ipynb`可以把flying-chairs数据集制作为适用于本项目的data。  
data类型为`6wh`的npy张量，其中六个通道的内容分别为：
```
data[0]:image1
data[1]:image2
data[2]:计算得到的光流flow场中的x方向位移量 u
data[3]:计算得到的光流flow场中的y方向位移量 v
data[4]:光流flow_truth场中的x方向位移量真值 u_t
data[5]:光流flow_truth场中的y方向位移量真值 v_t
```
#### remap
上述提到的`data[0]`中的`image1`可以通过remap操作与`image2`对齐，以达到更好的训练效果。
### u-net
#### 结构
包含四个下采样层和四个上采样层的 U-Net 结构。
```python
# 初始化模型
model = UNet(in_channels=6, out_channels=2)
```
#### forward
```python
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
```
## 核心程序说明
### raft文件夹
#### checkpoints文件夹
存放训练的raft-piv模型与其对应的训练日志
#### train-piv.py
训练针对piv图像(256*256)的raft模型。训练指令如下：
```zsh
python train-piv.py --name piv --stage piv --gpus 0 --num_steps 7800 --batch_size 1 --lr 0.00005 --image_size 256 256 --mixed_precision
```
#### dataset-piv-img2npy.py
用于将数据集中的img1、img2、flow文件与raft网络计算的结果打包生成npy文件。dataset指令示例如下：
```zsh
python dataset-piv-img2npy.py --model /home/panding/code/UR/UR/checkpoints/model-8-14-4.pth --path /home/panding/code/UR/piv-data/ur
```
#### dataset-baseline-multimodel.py
baseline对比方案一（使用多个训练的raft模型的结果值计算不确定度）的dataset代码。dataset指令示例如下：
```zsh
python dataset-baseline-multimodel.py --model1 /home/panding/code/UR/UR/raft/checkpoints/model-8-14-1.pth --model2 /home/panding/code/UR/UR/raft/checkpoints/model-8-14-2.pth --model3 /home/panding/code/UR/UR/raft/checkpoints/model-8-14-3.pth --model4 /home/panding/code/UR/UR/raft/checkpoints/model-8-14-4.pth --path /home/panding/code/UR/piv-data/ur
```
