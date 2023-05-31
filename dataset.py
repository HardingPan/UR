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

tensor = transforms.ToTensor()

def to_v(image):
    image = np.array(image)
    img_tensor = tensor(image)
    return img_tensor

# 图片读取
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

# 图像可视化
def viz(flo):
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    # img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()
    cv2.imwrite('res1.png', flo[:, :, [2,1,0]])

# def dataload(imfiles1, imfiles2):
#     image1 = load_image(imfiles1)
#     image2 = load_image(imfiles2)
    
#     padder = InputPadder(image1.shape)
#     image1, image2 = padder.pad(image1, image2)
    
    # flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

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
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            # torch.Size([1, 3, 440, 1024])
            image1, image2 = padder.pad(image1, image2)
            
            # torch.Size([1, 2, 440, 1024])
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(flow_up)
            # torch.Size([3, 440, 1024])
            image1 = torch.squeeze(image1)
            image2 = torch.squeeze(image2)
            flow_up = torch.squeeze(flow_up)
            # torch.Size([8, 440, 1024])
            result = torch.cat((image1, image2, flow_up), 0)
            
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
