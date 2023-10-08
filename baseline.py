"""
baseline比较方案
方案1: 多个参数模型对同一组图像对进行测量, 并求出不确定度
方案2: 使用单个模型对一组图像对经过增强后的四组图像对进行测量, 并求出不确定度
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

class MultiMethod():
    def __init__(self, data_path, method):
        """
        method = 0: MultiModel
        method = 1: MultiTransform
        """
        self.data = np.load(data_path)
        self.method = method
        
        if self.method == 1:
            self.detransform()
            print('detransform has competed')
        
        self.data_u = self.data[0:8:2]
        self.data_v = self.data[1:9:2]
        self.u_t = self.data[8]
        self.v_t = self.data[9]

        self.u_avg = np.mean(self.data_u, axis=0)
        self.v_avg = np.mean(self.data_v, axis=0)
        # 残差
        self.u_res = np.abs(np.repeat(self.u_avg[np.newaxis, :, :], 4, 0) - self.data_u)
        self.v_res = np.abs(np.repeat(self.v_avg[np.newaxis, :, :], 4, 0) - self.data_v)
    
    def std(self, show):
        sigma_u = np.sqrt(
            (np.square(self.u_res[0]) + np.square(self.u_res[1]) + np.square(self.u_res[2]) + np.square(self.u_res[3])) / 3
        )
        sigma_v = np.sqrt(
            (np.square(self.v_res[0]) + np.square(self.v_res[1]) + np.square(self.v_res[2]) + np.square(self.v_res[3])) / 3
        )
        
        if show == 1:
            plt.figure(figsize=(12,8))
            plt.subplot(1, 2, 1)
            plt.title('sigma_u')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(sigma_u)
            plt.colorbar(fraction=0.05)
            
            plt.subplot(1, 2, 2)
            plt.title('sigma_v')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(sigma_v)
            plt.colorbar(fraction=0.05)
            
        return sigma_u, sigma_v
    
    def std_truth(self, show=0):
        u_res_truth = np.abs(np.repeat(self.u_t[np.newaxis, :, :], 4, 0) - self.data_u)
        v_res_truth = np.abs(np.repeat(self.v_t[np.newaxis, :, :], 4, 0) - self.data_v)
        
        sigma_u_truth = np.sqrt(
            (np.square(u_res_truth[0]) + np.square(u_res_truth[1]) + np.square(u_res_truth[2]) + np.square(u_res_truth[3])) / 4
        )
        
        sigma_v_truth = np.sqrt(
            (np.square(v_res_truth[0]) + np.square(v_res_truth[1]) + np.square(v_res_truth[2]) + np.square(v_res_truth[3])) / 4
        )
        
        if show == 1:
            plt.figure(figsize=(12,8))
            plt.subplot(1, 2, 1)
            plt.title('sigma_u_truth')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(sigma_u_truth)
            plt.colorbar(fraction=0.05)
            
            plt.subplot(1, 2, 2)
            plt.title('sigma_v_truth')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(sigma_v_truth)
            plt.colorbar(fraction=0.05)
            
        return sigma_u_truth, sigma_v_truth
    
    def avg_std(self, show=0):
        sigma_u_avg, sigma_v_avg = self.std(show=0)
        sigma_u_avg = sigma_u_avg / 2
        sigma_v_avg = sigma_v_avg / 2
        
        if show == 1:
            plt.figure(figsize=(12,8))
            plt.subplot(1, 2, 1)
            plt.title('sigma_u_avg')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(sigma_u_avg)
            plt.colorbar(fraction=0.05)
            
            plt.subplot(1, 2, 2)
            plt.title('sigma_v_avg')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(sigma_v_avg)
            plt.colorbar(fraction=0.05)
        
        return sigma_u_avg, sigma_v_avg
    
    def uncertainty(self, show=0):
        
        sigma_u = np.sqrt(
            (np.square(self.data_u[0] - self.data_u[1]) + \
             np.square(self.data_u[0] - self.data_u[2]) + \
             np.square(self.data_u[0] - self.data_u[3])) / 2
        )
        
        sigma_v = np.sqrt(
            (np.square(self.data_v[0] - self.data_v[1]) + \
             np.square(self.data_v[0] - self.data_v[2]) + \
             np.square(self.data_v[0] - self.data_v[3])) / 2
        )
        
        if show == 1:
            plt.figure(figsize=(12,8))
            plt.subplot(1, 2, 1)
            plt.title('uncertainty_u')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(sigma_u)
            plt.colorbar(fraction=0.05)
            
            plt.subplot(1, 2, 2)
            plt.title('uncertainty_v')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(sigma_v)
            plt.colorbar(fraction=0.05)
            
        return sigma_u, sigma_v
    
    def detransform(self):
        
        self.data[2] = np.flip(self.data[2], 0)
        self.data[3] = np.negative(np.flip(self.data[3], 0))
        
        self.data[4] = np.rot90(np.negative(self.data[4]), k=2, axes=(0, 1))
        self.data[5] = np.rot90(np.negative(self.data[5]), k=2, axes=(0, 1))

        self.data[6] = np.negative(np.flip(self.data[6], 1))
        self.data[7] = np.flip(self.data[7], 1)

        # self.data[4] = np.negative(np.flip(self.data[4], 1))
        # self.data[5] = np.flip(self.data[5], 1)
        # self.data[6] = np.negative(np.flip(np.flip(self.data[6], 1), 0))
        # self.data[7] = np.negative(np.flip(np.flip(self.data[7], 1), 0))
        pass
    
    def get_data(self):
        return self.data
    
    def show(self):
        plt.figure(figsize=(12,4),facecolor='white')
        
        plt.subplot(2, 5, 1)
        # plt.title('Raft-1')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.data[0])
        cbar = plt.colorbar(fraction=0.05)
        cbar.ax.tick_params(color='black', labelcolor='black')
        
        plt.subplot(2, 5, 2)
        # plt.title('2.u')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.data[2])
        cbar = plt.colorbar(fraction=0.05)
        cbar.ax.tick_params(color='black', labelcolor='black')
        
        plt.subplot(2, 5, 3)
        # plt.title('3.u')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.data[4])
        cbar = plt.colorbar(fraction=0.05)
        cbar.ax.tick_params(color='black', labelcolor='black')
        
        plt.subplot(2, 5, 4)
        # plt.title('4.u')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.data[6])
        cbar = plt.colorbar(fraction=0.05)
        cbar.ax.tick_params(color='black', labelcolor='black')
        
        plt.subplot(2, 5, 5)
        # plt.title('u_truth')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.data[8])
        cbar = plt.colorbar(fraction=0.05)
        cbar.ax.tick_params(color='black', labelcolor='black')
        
        plt.subplot(2, 5, 6)
        # plt.title('1.v')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.data[1])
        cbar = plt.colorbar(fraction=0.05)
        cbar.ax.tick_params(color='black', labelcolor='black')
        
        plt.subplot(2, 5, 7)
        # plt.title('2.v')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.data[3])
        cbar = plt.colorbar(fraction=0.05)
        cbar.ax.tick_params(color='black', labelcolor='black')
        
        plt.subplot(2, 5, 8)
        # plt.title('3.v')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.data[5])
        cbar = plt.colorbar(fraction=0.05)
        cbar.ax.tick_params(color='black', labelcolor='black')
        
        plt.subplot(2, 5, 9)
        # plt.title('4.v')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.data[7])
        cbar = plt.colorbar(fraction=0.05)
        cbar.ax.tick_params(color='black', labelcolor='black')
        
        plt.subplot(2, 5, 10)
        # plt.title('v_truth')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.data[9])
        cbar = plt.colorbar(fraction=0.05)
        cbar.ax.tick_params(color='black', labelcolor='black')
        
        plt.show()
    
    
if __name__ == "__main__":
    
    data_path = '/home/panding/code/UR/piv-data/baseline-multimodel'
    datas = glob.glob(os.path.join(data_path, '*.npy'))
    randomidx = np.random.permutation(len(datas))
    datas = [datas[i] for i in randomidx]
    
    save_path = '/home/panding/code/UR/UR/baseline/res.png'
    test = MultiMethod(datas[0], 0)
    sigma_u, sigma_v = test.std()
    sigma_t_u, sigma_t_v = test.std_truth()

    plt.figure(figsize=(12,6))

    plt.subplot(231)
    plt.title('sigma_u')
    plt.imshow(sigma_u)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(fraction=0.05)

    plt.subplot(232)
    plt.title('sigma_u_truth')
    plt.imshow(sigma_t_u)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(fraction=0.05)

    plt.subplot(233)
    plt.imshow(np.abs(sigma_u-sigma_t_u))
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(fraction=0.05)

    plt.subplot(234)
    plt.title('sigma_v')
    plt.imshow(sigma_v)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(fraction=0.05)

    plt.subplot(235)
    plt.title('sigma_v_truth')
    plt.imshow(sigma_t_v)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(fraction=0.05)

    plt.subplot(236)
    plt.imshow(np.abs(sigma_v-sigma_t_v))
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(fraction=0.05)

    plt.savefig(save_path)
