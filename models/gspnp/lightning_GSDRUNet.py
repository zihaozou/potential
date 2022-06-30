import os
import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim import lr_scheduler
import random
from argparse import ArgumentParser

import torchvision
import numpy as np
import matplotlib.pyplot as plt
from .network_unet import UNetRes
from .dncnn import DnCNN
from .test_utils import test_mode
class StudentGrad(nn.Module):
    '''
    Standard DRUNet model
    '''
    def __init__(self, network='dncnn',numInChan=1,numOutChan=1):
        super(StudentGrad,self).__init__()
        # self.model = UNetRes(in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=DRUNET_nb, act_mode=act_mode,
        #                      downsample_mode='strideconv', upsample_mode='convtranspose')
        if network == 'dncnn':
            self.model = DnCNN(depth=12, in_channels=numInChan+1, out_channels=numOutChan, init_features=64, kernel_size=3)
        elif network == 'unet':
            self.model = UNetRes(in_nc=numInChan+1, out_nc=numOutChan, nc=[64, 128, 256, 512], nb=2, act_mode='E',
                              downsample_mode='strideconv', upsample_mode='convtranspose')


    def forward(self, x, sigma):
        noise_level_map = torch.FloatTensor(x.size(0), 1, x.size(2), x.size(3)).fill_(sigma).to(x.device)
        x = torch.cat((x, noise_level_map), 1)
        n = self.model(x)
        return n
    def grad(self, x, sigma,create_graph=True):
        x = x.float()
        x = x.requires_grad_()
        if x.size(2) % 8 == 0 and x.size(3) % 8 == 0:
            N = self(x, sigma)
        else:
            current_model = lambda v: self(v, sigma)
            N = test_mode(current_model, x, mode=5, refield=64, min_size=256)
        JN = torch.autograd.grad(N, x, grad_outputs=x - N, create_graph=create_graph,only_inputs=True)[0]
        Dg = x - N - JN
        return Dg
    def grad2(self, x, sigma,create_graph=True):
        x = x.float()
        x = x.requires_grad_()
        if x.size(2) % 8 == 0 and x.size(3) % 8 == 0:
            N = self(x, sigma)
        else:
            current_model = lambda v: self(v, sigma)
            N = test_mode(current_model, x, mode=5, refield=64, min_size=256)
        JN = torch.autograd.grad(N, x, grad_outputs=x - N, create_graph=create_graph,only_inputs=True)[0]
        Dg = x - N - JN
        return Dg,N

