from turtle import forward
import torch
import torch.nn as nn
from .gspnp.network_unet import UNetRes
from .gspnp.dncnn import DnCNN
from .gspnp.test_utils import test_mode
from torch.autograd.functional import vjp,jacobian,jvp
from torch.autograd import grad



class NNclass(nn.Module):
    def __init__(self,numInChan=3,numOutChan=3,network='unet',train_network=True):
        super(NNclass,self).__init__()
        if network=='unet':
            self.network=UNetRes(numInChan+1,numOutChan,nc=[64, 128, 256, 512], nb=2, act_mode='E', downsample_mode="strideconv", upsample_mode="convtranspose")
        elif network=='dncnn':
            self.network=DnCNN(depth=12, in_channels=numInChan+1, out_channels=numOutChan, init_features=64, kernel_size=3)
        for p in self.network.parameters():
            p.requires_grad = train_network
    def preForward(self,x,**kwargs):
        return x
    def postForward(self,x,input,**kwargs):
        return x
    def forward(self,x,sigma,**kwargs):
        x.requires_grad_()
        noise_level_map = sigma.expand(x.size(0),1,x.size(2),x.size(3))
        x_sigma = torch.cat((x, noise_level_map), 1)
        out = self.postForward(self.network(self.preForward(x_sigma,**kwargs)),x,**kwargs)
        return out

class DPIRNNclass(NNclass):
    def __init__(self, numInChan=3, numOutChan=3, network='unet', train_network=True):
        super(DPIRNNclass, self).__init__(numInChan, numOutChan, network, train_network)
        if network=='unet':
            self.network=UNetRes(numInChan+1,numOutChan,nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        elif network=='dncnn':
            self.network=DnCNN(depth=12, in_channels=numInChan+1, out_channels=numOutChan, init_features=64, kernel_size=3)
        for p in self.network.parameters():
            p.requires_grad = train_network
class GSPNPNNclass(NNclass):
    def __init__(self, numInChan=3, numOutChan=3, network='unet', train_network=True):
        super().__init__(numInChan, numOutChan, network, train_network)
    def postForward(self, N, input,**kwargs):
        JN=torch.autograd.grad(N, input, grad_outputs=input - N, create_graph=kwargs['create_graph'],only_inputs=True)[0]
        return N+JN
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

class REDPotentialNNclass(NNclass):
    def __init__(self, numInChan=3, numOutChan=3, network='unet', train_network=True):
        super().__init__(numInChan, numOutChan, network, train_network)
    def postForward(self, N, input, **kwargs):
        JN=grad(N,input,grad_outputs=input,create_graph=kwargs['create_graph'],only_inputs=True)[0]
        return 0.5*(N+JN)
        


class PotentialNNclass(NNclass):
    def __init__(self, numInChan=3, numOutChan=3, network='unet', train_network=True):
        super().__init__(numInChan, numOutChan, network, train_network)
    def postForward(self, N, input, **kwargs):
        N=N.mean([1,2,3]).sum()
        JN=grad(N,input,grad_outputs=torch.ones_like(input),create_graph=kwargs['create_graph'],only_inputs=True)[0]
        return JN