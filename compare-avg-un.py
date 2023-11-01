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
from UR.muenn import Ur
from utils import InputPadder

import argparse

DEVICE = 'cuda'

transform_gray = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 彩色图像转灰度图像num_output_channels默认1
    transforms.ToTensor()
])

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

def cat(imfiles_1, imfiles_2, imfiles_flo, imfiles_truth):
    for i in range(0, len(imfiles_1)):
        imfile_1 = imfiles_1[i]
        imfile_2 = imfiles_2[i]
        imfile_flo = imfiles_flo[i]
        imfile_truth = imfiles_truth[i]
        save_path = '/home/panding/code/UR/piv-data/ur-un'
        file_name = imfile_flo.split('/')[-1]
        # print(path)
        image1_rgb_tensor = load_image(imfile_1)
        padder = InputPadder(image1_rgb_tensor.shape)
        # 2 * torch.Size([1, 256, 256])
        image1_gray_tensor = transform_gray(Image.open(imfile_1))
        image2_gray_tensor = transform_gray(Image.open(imfile_2))
        image1_gray_tensor, image2_gray_tensor = padder.pad(image1_gray_tensor, image2_gray_tensor)
        
        image1_gray = image1_gray_tensor.numpy()
        image2_gray = image2_gray_tensor.numpy()
        flo = np.load(imfile_flo)
        truth = np.load(imfile_truth)
        result = np.vstack((image1_gray, image2_gray, flo[:2], truth[2:4]))
        # print(save_path+'/'+file_name)
        np.save(save_path+'/'+file_name, result)

def avg():
    
    mm_u = []
    mm_v = []

    mt_u = []
    mt_v = []

    muenn_u = []
    muenn_v = []
    
    model_path = '/home/panding/code/UR/unet-model/best-1.pt'
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(0, len(datas_un)):
        truth_u = np.abs(np.load(datas_un[i])[4] - np.load(datas_multimodel[i])[0])
        truth_v = np.abs(np.load(datas_un[i])[5] - np.load(datas_multimodel[i])[1])
        baseline_1 = MultiMethod(datas_multimodel[i], 0)
        uncertainty_u_mm, uncertainty_v_mm = baseline_1.uncertainty(show=0)

        baseline_2 = MultiMethod(datas_multitransform[i], 1)
        uncertainty_u_mt, uncertainty_v_mt = baseline_2.uncertainty(show=0)

        data = np.load(datas_un[i])
        data = data[:4]
        uncertainty = Ur('unet', data, path=model_path, device=my_device)
        sigma_u_ur_2show, sigma_v_ur_2show = uncertainty.get_sigma2show()
        sigma_u_ur, sigma_v_ur = uncertainty.get_sigma()
        # print(truth_u.shape, uncertainty_u_mm.shape)
        mm_u.append(SSIM(truth_u, uncertainty_u_mm))
        mm_v.append(SSIM(truth_v, uncertainty_v_mm))
        
        mt_u.append(SSIM(truth_u, uncertainty_u_mt))
        mt_v.append(SSIM(truth_v, uncertainty_v_mt))
        
        muenn_u.append(SSIM(truth_u, sigma_u_ur))
        muenn_v.append(SSIM(truth_v, sigma_v_ur))
        
        # mm_u.append(PSNR(truth_u, uncertainty_u_mm))
        # mm_v.append(PSNR(truth_v, uncertainty_v_mm))
        
        # mt_u.append(PSNR(truth_u, uncertainty_u_mt))
        # mt_v.append(PSNR(truth_v, uncertainty_v_mt))
        
        # muenn_u.append(PSNR(truth_u, sigma_u_ur))
        # muenn_v.append(PSNR(truth_v, sigma_v_ur))
        
    print(f"\n ————————— result ————————— \n \
        mm_u: {np.mean(mm_u)}, mm_v: {np.mean(mm_v)}, \n \
        mt_u: {np.mean(mt_u)}, mt_v: {np.mean(mt_v)}, \n \
        muenn_u: {np.mean(muenn_u)}, muenn_v: {np.mean(muenn_v)}")
        
def MSE(arr_1, arr_2):
    mse = np.mean( (arr_1 - arr_2) ** 2 )
    return mse

# PSNR越大，代表着图像质量越好
def PSNR(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# SSIM取值范围为[0,1]，值越大表示输出图像和无失真图像的差距越小，即图像质量越好。
def SSIM(y_true , y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    R = 255
    c1 = np.square(0.01*R)
    c2 = np.square(0.03*R)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom
    

data_path_multimodel = '/home/panding/code/UR/piv-data/unflownet-mm'
data_path_multitransform = '/home/panding/code/UR/piv-data/unflownet-mt'
data_path_truth = '/home/panding/code/UR/piv-data/truth'
data_path_un = '/home/panding/code/UR/piv-data/ur-un'

data_path_ur = '/home/panding/code/UR/piv-data/raft-test'

cls = 'b'

datas_multimodel = glob.glob(os.path.join(data_path_multimodel, cls+'*.npy'))
datas_multitransform = glob.glob(os.path.join(data_path_multitransform, cls+'*.npy'))
datas_truth = glob.glob(os.path.join(data_path_truth, cls+'*.npy'))
# datas_ur = glob.glob(os.path.join(data_path_ur, 'S*.npy'))
datas_ur_img_1 = glob.glob(os.path.join(data_path_ur, cls+'*img1.tif'))
datas_ur_img_2 = glob.glob(os.path.join(data_path_ur, cls+'*img2.tif'))
datas_un = glob.glob(os.path.join(data_path_un, cls+'*.npy'))

datas_multimodel = sorted(datas_multimodel)
datas_multitransform = sorted(datas_multitransform)
datas_truth = sorted(datas_truth)
datas_un = sorted(datas_un)

# datas_ur = sorted(datas_ur)
datas_ur_img_1 = sorted(datas_ur_img_1)
datas_ur_img_2 = sorted(datas_ur_img_2)
assert len(datas_multimodel) == len(datas_ur_img_1) == len(datas_ur_img_2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='set data or get avg')
    parser.add_argument('--data', action='store_true', help='set data')
    parser.add_argument('--avg', action='store_true', help='get avg')

    args = parser.parse_args()
    isSetData = args.data
    isGetAvg = args.avg

    if isSetData:
        cat(datas_ur_img_1, datas_ur_img_2, datas_multimodel, datas_truth)
    if isGetAvg:
        avg()