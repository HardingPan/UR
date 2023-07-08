# UR(uncertainty of raft)

## 环境配置
python=`3.11`
```
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
本项目的data基于flying-chairs数据集[flying-chairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html)。
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
### net
#### u-net

