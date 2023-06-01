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

from raft.raft import RAFT
from raft.utils import flow_viz
from raft.utils.utils import InputPadder



DEVICE = 'cuda'

batch_size=3

transform_rgb = transforms.ToTensor()

transform_gray = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 彩色图像转灰度图像num_output_channels默认1
    transforms.ToTensor()
])

# rgb图片读取
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    # img = np.array(Image.open(imfile))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

# 图像可视化
def viz(flo):
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    # img_flo = np.concatenate([img, flo], axis=0)

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()
    cv2.imwrite('res1.png', flo[:, :, [2,1,0]])

# def dataload(imfiles1, imfiles2):
#     image1 = load_image(imfiles1)
#     image2 = load_image(imfiles2)
    
#     padder = InputPadder(image1.shape)
#     image1, image2 = padder.pad(image1, image2)
    
#     flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

def dataload(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    
    data_path = 'data'
    
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        images_num = len(images)
        images_loading_num = 1
        print('\n', '--------------images loading...-------------', '\n')
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            
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
            
            # torch.Size([2, 436, 1024])
            image1_gray_tensor = transform_gray(Image.open(imfile1)).to(DEVICE)
            image2_gray_tensor = transform_gray(Image.open(imfile2)).to(DEVICE)
            # torch.Size([2, 440, 1024])
            image1_gray_tensor, image2_gray_tensor = padder.pad(image1_gray_tensor, image2_gray_tensor)
            
            """
            torch.Size([4, 440, 1024])
            四通道分别为 灰度后的i1, 灰度后的i2, u, v
            """
            result = torch.cat((image1_gray_tensor, image2_gray_tensor, flow_up), 0)
            result = result.cpu()
            result_np = result.numpy()
            data_path = data_path + '/' + imfile1[-5:-9:-1][-1::-1]
            
            np.save(data_path, result_np)
            data_path = 'data'
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
