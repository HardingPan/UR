'''
用于制作unflownet的数据集, 给muenn训练用
'''

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cv2
from PIL import Image
import math
from scipy.stats import spearmanr

from baseline import MultiMethod

import torch
from torchvision import transforms
import torch.nn as nn
from ur import Ur
from utils import InputPadder

import argparse

DEVICE = 'cuda'

transform_gray = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 彩色图像转灰度图像num_output_channels默认1
    transforms.ToTensor()
])

def load_flow_to_numpy(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    data2D = np.resize(data, (w, h, 2))
    data2D = data2D.transpose(2,0,1)

    return data2D

# 灰度图像读取
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    # img_rgb = img[:, :, np.newaxis]
    img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    img_rgb[:, :, 0] = img
    img_rgb[:, :, 1] = img
    img_rgb[:, :, 2] = img
    # cv2.imwrite('/home/panding/code/UR/UR/raft/rgb.png', img_rgb)
    img_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
    return img_rgb[None].to(DEVICE)

def cat(imfiles_1, imfiles_2, imfiles_flo, imfiles_un):
    
    for i in range(0, len(imfiles_1)):
        if i % int(len(imfiles_1)/100) == 0:
            print(f"dataset is loading: {i / int(len(imfiles_1)/100)} / 100 %, num {i+1}.")
        imfile_1 = imfiles_1[i]
        imfile_2 = imfiles_2[i]
        imfile_flo = imfiles_flo[i]
        imfile_un = imfiles_un[i]
        
        save_path = '/home/panding/code/UR/piv-data/unflownet-to-train-muenn/'
        file_name = imfile_un.split('/')[-1]
        # print(file_name)
        img_1 = np.expand_dims(np.array(Image.open(imfile_1)).astype(np.uint8), 0)
        img_2 = np.expand_dims(np.array(Image.open(imfile_2)).astype(np.uint8), 0)
        flo = load_flow_to_numpy(imfile_flo)
        un = np.load(imfile_un)[:2]
        # print(img_1.shape, img_2.shape, flo.shape, un.shape)
        res = np.concatenate((img_1, img_2, un, flo), 0)
        # print(res.shape)
        # print(save_path+file_name)
        np.save(save_path+file_name, res)


if __name__ == "__main__":
    
    data_path_truth = '/home/panding/code/UR/piv-data/raft'
    data_path_un = '/home/panding/code/UR/piv-data/unflownet-result'
    datas_truth_img1 = sorted(glob.glob(os.path.join(data_path_truth, '*1.tif')))
    datas_truth_img2 = sorted(glob.glob(os.path.join(data_path_truth, '*2.tif')))
    datas_truth_flo = sorted(glob.glob(os.path.join(data_path_truth, '*.flo')))
    datas_un = glob.glob(os.path.join(data_path_un, '*.npy'))
    assert len(datas_truth_img1) == len(datas_truth_img2) == len(datas_truth_flo) == len(datas_un)
    
    cat(datas_truth_img1, datas_truth_img2, datas_truth_flo, datas_un)
    
    pass
    # parser = argparse.ArgumentParser(description='set data or get avg')
    # parser.add_argument('--data', action='store_true', help='set data')
    # parser.add_argument('--avg', action='store_true', help='get avg')

    # args = parser.parse_args()
    # isSetData = args.data
    # isGetAvg = args.avg

    # if isSetData:
    #     cat(datas_ur_img_1, datas_ur_img_2, datas_multimodel, datas_truth)
    # if isGetAvg:
    #     avg()