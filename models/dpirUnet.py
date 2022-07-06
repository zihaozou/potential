from turtle import forward
import torch
import torch.nn as nn
from .gspnp.network_unet import UNetRes
from .gspnp.dncnn import DnCNN
from .gspnp.test_utils import test_mode
from torch.autograd.functional import vjp
class NNclass(nn.Module):
    def __init__(self,numInChan=3,numOutChan=3):
        super(NNclass,self).__init__()
        self.network=UNetRes(numInChan+1,numOutChan,nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    def forward(self,x,sigma,tau,create_graph):
        noise_level_map = sigma.expand(x.size(0),1,x.size(2),x.size(3))
        x = torch.cat((x, noise_level_map), 1)
        n = 1.0/tau*self.network(tau*x)
        return n
class NNclass2(nn.Module):
    def __init__(self,numInChan=3,numOutChan=3,network='unet',train_network=True):
        super(NNclass2,self).__init__()
        if network=='unet':
            self.network=UNetRes(numInChan+1,numOutChan,nc=[64, 128, 256, 512], nb=2, act_mode='E', downsample_mode="strideconv", upsample_mode="convtranspose")
        elif network=='dncnn':
            self.network=DnCNN(depth=12, in_channels=numInChan+1, out_channels=numOutChan, init_features=64, kernel_size=3)
        for p in self.network.parameters():
            p.requires_grad = train_network
    def forward(self,x,sigma,create_graph):
        noise_level_map = sigma.expand(x.size(0),1,x.size(2),x.size(3))
        x_noise_map = torch.cat((x, noise_level_map), 1)
        N = self.network(x_noise_map)
        JN = torch.autograd.grad(N, x, grad_outputs=x - N, create_graph=create_graph,only_inputs=True)[0]
        return N + JN
    def calculate_grad(self,x,sigma):
        x.requires_grad_()
        noise_level_map = torch.tensor([sigma],device=x.device,dtype=x.dtype).expand(x.size(0),1,x.size(2),x.size(3))
        x_noise_map = torch.cat((x, noise_level_map), 1)
        if x.size(2) % 8 == 0 and x.size(3) % 8 == 0:
            N = self.network(x_noise_map)
        else:
            current_model = lambda v: self.network(v)
            N = test_mode(current_model, x_noise_map, mode=5, refield=64, min_size=256)
        JN = torch.autograd.grad(N, x, grad_outputs=x - N, create_graph=False,only_inputs=True)[0]
        return x-N-JN, N

class NNclass3(nn.Module):
    def __init__(self,numInChan=3,numOutChan=3,network='unet',train_network=True):
        super(NNclass3,self).__init__()
        if network=='unet':
            self.network=UNetRes(numInChan+1,numOutChan,nc=[64, 128, 256, 512], nb=2, act_mode='E', downsample_mode="strideconv", upsample_mode="convtranspose")
        elif network=='dncnn':
            self.network=DnCNN(depth=12, in_channels=numInChan+1, out_channels=numOutChan, init_features=64, kernel_size=3)
        for p in self.network.parameters():
            p.requires_grad = train_network
    def forward(self,x,sigma,create_graph,strict):
        x.requires_grad_()
        def f(input):
            noise_level_map = sigma.expand(input.size(0),1,input.size(2),input.size(3))
            x_noise_map = torch.cat((input, noise_level_map), 1)
            N = self.network(x_noise_map)
            return N
        vjpResult=vjp(f,x,x,create_graph=create_graph, strict=strict)
        return x-0.5*(vjpResult[0]+vjpResult[1])