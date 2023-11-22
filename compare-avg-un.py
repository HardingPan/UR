import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cv2
from PIL import Image
import math
from scipy.stats import spearmanr
from tqdm.autonotebook import tqdm

from baseline import MultiMethod

import torch
from torchvision import transforms
import torch.nn as nn
from muenn import MueNN
from utils import InputPadder

import argparse

# DEVICE = 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def load_data(cls):

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
    
    return datas_multimodel, datas_multitransform, datas_truth, datas_un

def compute_avg(mms, mts, uns, mm, mt, mue):
    # mm_u_ssim, mm_v_ssim, mt_u_ssim, mt_v_ssim, muenn_u_ssim, muenn_v_ssim = [], [], [], [], [], []
    # mm_u_psnr, mm_v_psnr, mt_u_psnr, mt_v_psnr, muenn_u_psnr, muenn_v_psnr = [], [], [], [], [], []
    mm_psnr, mt_psnr, muenn_psnr = [], [], []
    mm_ssim, mt_ssim, muenn_ssim = [], [], []
    mm_pdf, mt_pdf, muenn_pdf = [], [], []
    for i in tqdm(range(0, len(uns))):
        res_u = np.abs(np.load(uns[i])[-2] - np.load(uns[i])[2])
        res_v = np.abs(np.load(uns[i])[-1] - np.load(uns[i])[3])
        u_mm, v_mm = mm.uncertainty(np.load(mms[i]))
        u_mt, v_mt = mt.uncertainty(np.load(mts[i]))
        data_mue = np.load(uns[i])[:4]
        data_mue = torch.from_numpy(data_mue).to(DEVICE)
        u_mue, v_mue = mue.get_sigma(data_mue)
        
        res_2d = np.stack((res_u, res_v), axis=-1).reshape((256, 256, 2))
        mm_2d = np.stack((u_mm, v_mm), axis=-1).reshape((256, 256, 2))
        mt_2d = np.stack((u_mt, v_mt), axis=-1).reshape((256, 256, 2))
        mue_2d = np.stack((u_mue, v_mue), axis=-1).reshape((256, 256, 2))    
        
        mm_psnr.append(PSNR(mm_2d, res_2d))
        mt_psnr.append(PSNR(mt_2d, res_2d))
        muenn_psnr.append(PSNR(mue_2d, res_2d))
        
        mm_ssim.append(SSIM(res_2d, mm_2d))
        mt_ssim.append(SSIM(res_2d, mt_2d))
        muenn_ssim.append(SSIM(res_2d, mue_2d))
        
        pdf_mm_2d = np.sqrt(np.square(pdf(res_u, u_mm)) + np.square(pdf(res_v, v_mm)))
        pdf_mt_2d = np.sqrt(np.square(pdf(res_u, u_mt)) + np.square(pdf(res_v, v_mt)))
        pdf_mue_2d = np.sqrt(np.square(pdf(res_u, u_mue)) + np.square(pdf(res_v, v_mue)))
        
        mm_pdf.append(pdf_mm_2d)
        mt_pdf.append(pdf_mt_2d)
        muenn_pdf.append(pdf_mue_2d)
        
    print(np.mean(mm_psnr), np.mean(mt_psnr), np.mean(muenn_psnr))
    print(np.mean(mm_ssim), np.mean(mt_ssim), np.mean(muenn_ssim))
    print(np.mean(mm_pdf), np.mean(mt_pdf), np.mean(muenn_pdf))
    
    f = open('/home/panding/code/UR/UR/data.txt','a')
    f.write(np.mean(mm_ssim), np.mean(mt_ssim), np.mean(muenn_ssim))
    f.write('\n')   
    f.close()
        
        # mm_u_ssim.append(SSIM(res_u, u_mm))
        # mm_v_ssim.append(SSIM(res_v, v_mm))
        # mt_u_ssim.append(SSIM(res_u, u_mt))
        # mt_v_ssim.append(SSIM(res_v, v_mt))
        # muenn_u_ssim.append(SSIM(res_u, u_mue))
        # muenn_v_ssim.append(SSIM(res_v, v_mue))
        # mm_u_psnr.append(PSNR(res_u, u_mm))
        # mm_v_psnr.append(PSNR(res_v, v_mm))
        # mt_u_psnr.append(PSNR(res_u, u_mt))
        # mt_v_psnr.append(PSNR(res_v, v_mt))
        # muenn_u_psnr.append(PSNR(res_u, u_mue))
        # muenn_v_psnr.append(PSNR(res_v, v_mue))
        
        
        
        
    # print(f" \
    # {np.mean(mm_u_psnr)}, {np.mean(mm_v_psnr)}, {np.mean(mm_u_ssim)}, {np.mean(mm_v_ssim)}\n \
    # {np.mean(mt_u_psnr)}, {np.mean(mt_v_psnr)}, {np.mean(mt_u_ssim)}, {np.mean(mt_v_ssim)}\n \
    # {np.mean(muenn_u_psnr)}, {np.mean(muenn_v_psnr)}, {np.mean(muenn_u_ssim)}, {np.mean(muenn_v_ssim)}\n ")

def get_avg():
    
    cls = ['backstep', 'cylinder', 'JHTDB_channel', 'JHTDB_isotropic1024_hd', 'JHTDB_mhd1024_hd', 'SQG']
    model_path = '/home/panding/code/UR/unet-model/best-1.pt'
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mue = MueNN(model_path, my_device)
    mm = MultiMethod(0)
    mt = MultiMethod(1)
    for i in range(len(cls)):
        print(cls[i])
        mms, mts, truths, uns = load_data(cls[i])
        compute_avg(mms, mts, uns, mm, mt, mue)
        

# # PSNR越大，代表着图像质量越好
# def PSNR(img1, img2):
#     mse = np.mean( (img1 - img2) ** 2 )
#     if mse == 0:
#         return 100
#     PIXEL_MAX = 255.0
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def psnr_define(val1, val2):
    val1 = torch.from_numpy(val1).to(DEVICE)
    val2 = torch.from_numpy(val2).to(DEVICE)
    mse = torch.mean((val1 - val2) ** 2)
    if mse == 0:
        return 100  # 信噪比无穷大时的特殊情况
    max_val = torch.max(val1)
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def PSNR(map1, map2):
    psnr_values = torch.zeros(map1.shape[0], map1.shape[1])
    for i in range(map1.shape[0]):
        for j in range(map1.shape[1]):
            psnr_values[i, j] = torch.mean(torch.stack([psnr_define(map1[i, j], map2[i, j])]))

    average_psnr = torch.mean(psnr_values)
    return average_psnr.item()

def SSIM(y_true, y_pred):
    R = 255
    c1 = (0.01 * R) ** 2
    c2 = (0.03 * R) ** 2
    ssim_map = torch.zeros(y_true.shape[0], y_true.shape[1]).to(DEVICE)  # 创建与输入图像相同大小的零张量来存储每个位置上的 SSIM 值

    y_true_torch = torch.from_numpy(y_true.transpose(2, 0, 1)).float().to(DEVICE)
    y_pred_torch = torch.from_numpy(y_pred.transpose(2, 0, 1)).float().to(DEVICE)

    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[1]):
            u_true = y_true_torch[0, i, j]
            u_pred = y_pred_torch[0, i, j]
            var_true = torch.var(y_true_torch[:, i, j])
            var_pred = torch.var(y_pred_torch[:, i, j])
            std_true = torch.sqrt(var_true)
            std_pred = torch.sqrt(var_pred)

            numerator = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
            denominator = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)

            ssim_map[i, j] = numerator / denominator

    average_ssim = torch.mean(ssim_map)  # 计算平均的 SSIM 值
    return average_ssim.item()  # 返回平均的 SSIM 值以及每个位置上的 SSIM 值

def pdf(mu, sigma):
    return (1/sigma*math.sqrt(2*math.pi))*np.exp(-np.square(mu)/(2*np.square(sigma)))
    

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
        get_avg()