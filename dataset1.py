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
    # img = np.array(Image.open(imfile)).astype(np.uint8)
    img = np.array(Image.open(imfile))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

# 光流可视化
def viz(flo):
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    flo = flow_viz.flow_to_image(flo)

    cv2.imwrite('res1.png', flo[:, :, [2,1,0]])


def dataload(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    data = []
    
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            # 用这种方法得到的张量torch.Size([1, 3, 436, 1024])
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            
            # 用这种方法得到的张量也是torch.Size([1, 3, 436, 1024])
            # image1 = transform_rgb(Image.open(imfile1)).to(DEVICE)
            # image2 = transform_rgb(Image.open(imfile2)).to(DEVICE)

            padder = InputPadder(image1.shape)
            # 这就变成440了，torch.Size([1, 3, 440, 1024])
            image1, image2 = padder.pad(image1, image2)
            
            # torch.Size([1, 2, 440, 1024])
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(flow_up)
            # torch.Size([2, 440, 1024])
            flow_up = torch.squeeze(flow_up)
            
            # torch.Size([1, 436, 1024])
            image1_gray_tensor = transform_gray(Image.open(imfile1)).to(DEVICE)
            image2_gray_tensor = transform_gray(Image.open(imfile2)).to(DEVICE)
            image1_gray_tensor, image2_gray_tensor = padder.pad(image1_gray_tensor, image2_gray_tensor)
            print(image1_gray_tensor.shape)
            # 没法cat, 前面两个w为436, flow_up是440
            result = torch.cat((image1_gray_tensor, image2_gray_tensor, flow_up), 0)
            print(result.shape)
            data.append(result)
            
            return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    res = dataload(args)
    print(res)
