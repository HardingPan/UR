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
import torch.nn as nn
from ur import Ur

"""
-----------------------PSNR-----------------------
backstep
mm_u: 73.54842328785632, mm_v: 77.37168452931512, 
mt_u: 66.78228424945453, mt_v: 69.80684929651537, 
muenn_u: 74.62859433313226, muenn_v: 81.67577385648771

cylinder
mm_u: 68.42022170110164, mm_v: 67.69542094964947, 
mt_u: 66.75352014987799, mt_v: 67.25183220238121, 
muenn_u: 68.8276276119546, muenn_v: 71.74810006823458

JHTDB_channel
mm_u: 66.43420460259607, mm_v: 64.51813748153762, 
mt_u: 64.43898715088581, mt_v: 65.95533857428032, 
muenn_u: 64.63706052757746, muenn_v: 68.81093396937085

JHTDB_isotropic1024_hd
mm_u: 65.43902960640503, mm_v: 63.569233157098154, 
mt_u: 63.7768370495447, mt_v: 63.74183603837218, 
muenn_u: 63.41048422101705, muenn_v: 65.55305769806245

JHTDB_mhd1024_hd
mm_u: 64.68010969503942, mm_v: 62.381441944702246, 
mt_u: 62.74508374392447, mt_v: 62.444984576136655, 
muenn_u: 62.84535842568205, muenn_v: 64.45767668098178

SQG
mm_u: 61.55290693906337, mm_v: 60.79030206465822, 
mt_u: 59.82522063994518, mt_v: 59.915213068291266, 
muenn_u: 60.56978907915541, muenn_v: 62.09819218395154

-----------------------SSIM-----------------------
backstep
mm_u: 0.9998147654693925, mm_v: 0.9999013087794179, 
mt_u: 0.9989743271451375, mt_v: 0.9993799049367226, 
muenn_u: 0.999843054403522, muenn_v: 0.9999873927752148

cylinder
mm_u: 0.9997821368307069, mm_v: 0.9995787396662047, 
mt_u: 0.9996973184894007, mt_v: 0.9995806220820934, 
muenn_u: 0.9995989513054413, muenn_v: 0.9997804719829324

JHTDB_channel
mm_u: 0.9997656603974714, mm_v: 0.9980611781525585, 
mt_u: 0.9997517983019203, mt_v: 0.9991539106641981, 
muenn_u: 0.998183552721057, muenn_v: 0.9993565125599523

JHTDB_isotropic1024_hd
mm_u: 0.99970250237522, mm_v: 0.9987470019673, 
mt_u: 0.9998611015822388, mt_v: 0.9992077326384922, 
muenn_u: 0.9976926574233685, muenn_v: 0.9987355968272824

JHTDB_mhd1024_hd
mm_u: 0.9994894715066985, mm_v: 0.9983437375188068, 
mt_u: 0.999689906245185, mt_v: 0.9989016772259214, 
muenn_u: 0.9975920233232877, muenn_v: 0.9985899848849762

SQG
mm_u: 0.9992495956395927, mm_v: 0.9983468385425309, 
mt_u: 0.9995873795213176, mt_v: 0.9984375100885128, 
muenn_u: 0.996333335084804, muenn_v: 0.997684274423435
"""

# data_path = '/home/panding/code/UR/piv-data/baseline-multimodel'
# datas = glob.glob(os.path.join(data_path, '*.npy'))
# randomidx = np.random.permutation(len(datas))
# datas = [datas[i] for i in randomidx]

data_path_multimodel = '/home/panding/code/UR/piv-data/baseline-multimodel'
data_path_multitransform = '/home/panding/code/UR/piv-data/baseline-multitransform'
data_path_ur = '/home/panding/code/UR/piv-data/raft-test'
data_path_truth = '/home/panding/code/UR/piv-data/truth'

datas_multimodel = glob.glob(os.path.join(data_path_multimodel, 'S*.npy'))
datas_multitransform = glob.glob(os.path.join(data_path_multitransform, 'S*.npy'))
datas_ur = glob.glob(os.path.join(data_path_ur, 'S*.npy'))
datas_truth = glob.glob(os.path.join(data_path_truth, 'S*.npy'))

datas_multimodel = sorted(datas_multimodel)
datas_multitransform = sorted(datas_multitransform)
datas_ur = sorted(datas_ur)
datas_truth = sorted(datas_truth)

model_path = '/home/panding/code/UR/unet-model/best-1.pt'
my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def probability(sigma, loss):
    return np.abs(1 / (sigma * np.sqrt(2 * math.pi))) * np.exp(-np.square(loss) / (2 * np.square(sigma)))

mm_u = []
mm_v = []

mt_u = []
mt_v = []

muenn_u = []
muenn_v = []

for i in range(0, len(datas_multimodel)):
    
    truth = np.load(datas_truth[i])
    truth_u = truth[4]
    truth_v = truth[5]
    
    baseline_1 = MultiMethod(datas_multimodel[i], 0)
    uncertainty_u_mm, uncertainty_v_mm = baseline_1.uncertainty(show=0)

    baseline_2 = MultiMethod(datas_multitransform[i], 1)
    uncertainty_u_mt, uncertainty_v_mt = baseline_2.uncertainty(show=0)

    data = np.load(datas_ur[i])
    data = data[:4]

    uncertainty = Ur('unet', data, path=model_path, device=my_device)
    sigma_u_ur_2show, sigma_v_ur_2show = uncertainty.get_sigma2show()
    sigma_u_ur, sigma_v_ur = uncertainty.get_sigma()
    
    # print(f"truth_u: {truth_u.shape}, \
    #       uncertainty_u_mm: {uncertainty_u_mm.shape}, \
    #       uncertainty_u_mt: {uncertainty_u_mt.shape}, \
    #       sigma_u_ur: {sigma_u_ur.shape}")
    
    # mm_u.append(PSNR(truth_u, uncertainty_u_mm))
    # mm_v.append(PSNR(truth_v, uncertainty_v_mm))
    
    # mt_u.append(PSNR(truth_u, uncertainty_u_mt))
    # mt_v.append(PSNR(truth_v, uncertainty_v_mt))
    
    # muenn_u.append(PSNR(truth_u, sigma_u_ur))
    # muenn_v.append(PSNR(truth_v, sigma_v_ur))
    
    mm_u.append(SSIM(truth_u, uncertainty_u_mm))
    mm_v.append(SSIM(truth_v, uncertainty_v_mm))
    
    mt_u.append(SSIM(truth_u, uncertainty_u_mt))
    mt_v.append(SSIM(truth_v, uncertainty_v_mt))
    
    muenn_u.append(SSIM(truth_u, sigma_u_ur))
    muenn_v.append(SSIM(truth_v, sigma_v_ur))
    
print(f"\n ————————— result ————————— \n \
        mm_u: {np.mean(mm_u)}, mm_v: {np.mean(mm_v)}, \n \
        mt_u: {np.mean(mt_u)}, mt_v: {np.mean(mt_v)}, \n \
        muenn_u: {np.mean(muenn_u)}, muenn_v: {np.mean(muenn_v)}")