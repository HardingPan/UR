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
    def __init__(self, method):
        """
        method = 0: MultiModel
        method = 1: MultiTransform
        """
        self.method = method
    def uncertainty(self, data, show=0):
        if self.method == 1:
            self.detransform(data)
        data_u = data[0:8:2]
        data_v = data[1:9:2]
        sigma_u = np.sqrt(
            (np.square(data_u[0] - data_u[1]) + \
             np.square(data_u[0] - data_u[2]) + \
             np.square(data_u[0] - data_u[3])) / 2
        )
        
        sigma_v = np.sqrt(
            (np.square(data_v[0] - data_v[1]) + \
             np.square(data_v[0] - data_v[2]) + \
             np.square(data_v[0] - data_v[3])) / 2
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
    
    def detransform(self, data):
        
        data[2] = np.flip(data[2], 0)
        data[3] = np.negative(np.flip(data[3], 0))
        
        data[4] = np.rot90(np.negative(data[4]), k=2, axes=(0, 1))
        data[5] = np.rot90(np.negative(data[5]), k=2, axes=(0, 1))

        data[6] = np.negative(np.flip(data[6], 1))
        data[7] = np.flip(data[7], 1)

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
    
