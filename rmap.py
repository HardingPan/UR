import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision


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