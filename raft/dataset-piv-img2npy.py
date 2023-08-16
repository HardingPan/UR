import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'

batch_size=3

transform_origin = transforms.ToTensor()

transform_gray = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 彩色图像转灰度图像num_output_channels默认1
    transforms.ToTensor()
])
"""
def load_image2cat(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    # img = np.array(Image.open(imfile))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)
"""
# 灰度图像读取
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    img_rgb[:, :, 0] = img
    img_rgb[:, :, 1] = img
    img_rgb[:, :, 2] = img
    img_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
    return img_rgb[None].to(DEVICE)

# 图像可视化
def viz(flo):
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    # img_flo = np.concatenate([img, flo], axis=0)
 
    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()
    cv2.imwrite('res1.png', flo[:, :, [2,1,0]])

# 将第一帧对其到第二帧
def remap(img,x,y):
    """ a wrapper of opencv remap, adopted from 
    img:NxHxW, x:NxHxW ,y:NxHxW
    out:NxHxW
    Implement Scaling and Squaring
    https://github.com/yongleex/DiffeomorphicPIV/blob/main/deformpiv.py#L180
    """
    
    # convert x,y to grid:NxHxWx2
    grid = torch.stack((x, y), dim=-1)
    
    # normalize grid to (-1,1) for grid_sample
    # under pixel coordination system, x->W, y->H
    grid_shape = grid.shape[1:3]
    grid[:,:,:,0] = (grid[:,:,:,0] / (grid_shape[1] - 1) - 0.5)*2
    grid[:,:,:,1] = (grid[:,:,:,1] / (grid_shape[0] - 1) - 0.5)*2

    # shape img to NxCxHxW for grid_sample
    img = torch.unsqueeze(img, dim=1)
    out = F.grid_sample(img, grid, mode='bicubic', align_corners=True)
    
    return torch.squeeze(out, dim=1)

# 读取flo为tensor
def load_flow_to_numpy(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    data2D = np.resize(data, (w, h, 2))
    data2D = data2D.transpose(2,0,1)
    data2D_tensor = torch.from_numpy(data2D)
    return data2D_tensor

def dataload(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    
    data_path = '/home/panding/code/UR/piv-data/ur'
    
    with torch.no_grad():
        # 读取路径内的成对图像和flow真值
        images1 = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg')) + \
                 glob.glob(os.path.join(args.path, '*.ppm')) + \
                 glob.glob(os.path.join(args.path, '*_img1.tif'))
        
        images2 = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg')) + \
                 glob.glob(os.path.join(args.path, '*.ppm')) + \
                 glob.glob(os.path.join(args.path, '*_img2.tif'))
        flow_truth = glob.glob(os.path.join(args.path, '*.flo'))
        
        images1 = sorted(images1)
        images2 = sorted(images2)
        flow_truth = sorted(flow_truth)
        assert (len(images1) == len(images2))

        idx = np.random.permutation(len(images1))
        images1_shuffled = [images1[i] for i in idx]
        images2_shuffled = [images2[i] for i in idx]
        flows_shuffled = [flow_truth[i] for i in idx]

        images_num = len(images2)
        images_loading_num = 1
        print('\n', '--------------images loading...-------------', '\n')
        for flow, imfile1, imfile2 in zip(flows_shuffled, images1_shuffled, images2_shuffled):
            
            images_loading_num = images_loading_num + 1
            # torch.Size([3, 436, 1024])
            image1_rgb_tensor = load_image(imfile1)
            image2_rgb_tensor = load_image(imfile2)
            
            """
            torch.Size([1, 3, 440, 1024])
            这个pad操作会改变张量的尺寸, 后面灰度张量也需要pad一下才可以和光流张量拼接
            """
            padder = InputPadder(image1_rgb_tensor.shape)
            image1, image2 = padder.pad(image1_rgb_tensor, image2_rgb_tensor)
            
            # torch.Size([1, 2, 440, 1024])
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(flow_up)
            # torch.Size([2, 440, 1024])
            flow_up = torch.squeeze(flow_up)

            flow_up_u, flow_up_v = flow_up.split(1, 0)
            
            # torch.Size([2, 436, 1024])
            image1_gray_tensor = transform_origin(Image.open(imfile1)).to(DEVICE)
            image2_gray_tensor = transform_origin(Image.open(imfile2)).to(DEVICE)
            
            # torch.Size([2, 440, 1024])
            image1_gray_tensor, image2_gray_tensor = padder.pad(image1_gray_tensor, image2_gray_tensor)

            image1_gray_tensor_remap = remap(image1_gray_tensor, flow_up_u, flow_up_v)
            
            # 读取flow的真值
            # flow_path = '/home/panding/code/UR/piv-data/ur' + flow
            flow_truth = load_flow_to_numpy(flow).to(DEVICE)

            """
            torch.Size([6, 440, 1024])
            六通道分别为 灰度后的i1, 灰度后的i2, u, v, u_t, v_t
            """
            result = torch.cat((image2_gray_tensor, image1_gray_tensor_remap, flow_up, flow_truth), 0)
            result = result.cpu()
            result_np = result.numpy()
            # data_path = data_path + '/' + imfile1[6:-4]
            data_path = imfile1[:-9]
            np.save(data_path, result_np)
            # data_path = '/home/panding/code/UR/data-chair'
            if images_loading_num % 5 == 0:
                print('\n', '--------------images loaded: ', images_loading_num, ' / ', images_num, '-------------', '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    res = dataload(args)