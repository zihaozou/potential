from turtle import forward
import torch
import torch.nn as nn
from utils import utils_sr
from utils.utils_restoration import array2tensor
import numpy as np
class PNP(nn.Module):
    def __init__(self,tau,lamb,rObj,degradation_mode,sf=None):
        super(PNP,self).__init__()
        self.tau = nn.parameter.Parameter(torch.tensor([tau]))
        #self.tau=tau
        self.lamb = lamb
        self.rObj = rObj
        self.degradation_mode = degradation_mode
    def initialize_prox(self, img, degradation):
        '''
        calculus for future prox computatations
        :param img: degraded image
        :param degradation: 2D blur kernel for deblurring and SR, mask for inpainting
        '''
        if self.degradation_mode == 'deblurring':
            self.k = degradation
            self.k_tensor = array2tensor(np.expand_dims(self.k, 2)).float().to(img.device)
            self.FB, self.FBC, self.F2B, self.FBFy = utils_sr.pre_calculate(img, self.k_tensor, 1)
        elif self.degradation_mode == 'SR':
            self.k = degradation
            self.k_tensor = array2tensor(np.expand_dims(self.k, 2)).float().to(img.device)
            self.FB, self.FBC, self.F2B, self.FBFy = utils_sr.pre_calculate(img, self.k_tensor, self.sf)
        elif self.degradation_mode == 'inpainting':
            self.M = array2tensor(degradation).to(img.device)
            self.My = self.M*img
        else:
            print('degradation mode not treated')

    def calculate_prox(self, img):
        '''
        Calculation of the proximal mapping of the data term f
        :param img: input for the prox
        :return: prox_f(img)
        '''
        if self.degradation_mode == 'deblurring':
            rho = torch.tensor([1/self.tau]).repeat(1, 1, 1, 1).float().to(img.device)
            px = utils_sr.data_solution(img, self.FB, self.FBC, self.F2B, self.FBFy, rho, 1)
        elif self.degradation_mode == 'SR':
            rho = torch.tensor([1 / self.tau]).repeat(1, 1, 1, 1).float().to(img.device)
            px = utils_sr.data_solution(img, self.FB, self.FBC, self.F2B, self.FBFy, rho, self.sf)
        elif self.degradation_mode == 'inpainting':
            if self.noise_level_img > 1e-2:
                px = (self.tau*self.My + img)/(self.tau*self.M+1)
            else :
                px = self.My + (1-self.M)*img
        else:
            print('degradation mode not treated')
        return px

    def forward(self, n_ipt,sigma,create_graph=True):
        '''
        forward pass of the PNP
        :param n_ipt: input image NxCxHxW
        :param n_y: degraded image NxCxHxW
        '''
        Ds= self.rObj.grad(n_ipt, sigma / 255.,create_graph)
        Dx=n_ipt-Ds
        z=(1-torch.sigmoid(self.tau))*n_ipt+torch.sigmoid(self.tau)*Dx
        x=self.calculate_prox(z)
        return x


class DPIRPNP(nn.Module):
    def __init__(self,tau,lamb,rObj,train_tau_lamb,degradation_mode):
        super(DPIRPNP,self).__init__()
        self.rObj=rObj
        self.tau=torch.nn.parameter.Parameter(torch.tensor(tau),requires_grad=train_tau_lamb)
        self.lamb=torch.nn.parameter.Parameter(torch.tensor(lamb),requires_grad=train_tau_lamb)
        self.degradation_mode=degradation_mode
    def initialize_prox(self, img, degradation,noise_level_img,sf):
        '''
        calculus for future prox computatations
        :param img: degraded image
        :param degradation: 2D blur kernel for deblurring and SR, mask for inpainting
        '''
        if self.degradation_mode == 'deblurring':
            self.k = degradation
            self.k_tensor = array2tensor(np.expand_dims(self.k, 2)).float().to(img.device)
            self.FB, self.FBC, self.F2B, self.FBFy = utils_sr.pre_calculate(img, self.k_tensor, 1)
        elif self.degradation_mode == 'SR':
            self.k = degradation
            self.k_tensor = array2tensor(np.expand_dims(self.k, 2)).float().to(img.device)
            self.FB, self.FBC, self.F2B, self.FBFy = utils_sr.pre_calculate(img, self.k_tensor, sf)
        elif self.degradation_mode == 'inpainting':
            self.M = degradation
            self.My = self.M*img
        self.noise_level_img=noise_level_img

    def calculate_prox(self, img):
        '''
        Calculation of the proximal mapping of the data term f
        :param img: input for the prox
        :return: prox_f(img)
        '''
        if self.degradation_mode == 'deblurring':
            proxf = utils_sr.data_solution(img, self.FB, self.FBC, self.F2B, self.FBFy, alpha=1/self.tau, sf=1)
        elif self.degradation_mode == 'SR':
            proxf = utils_sr.data_solution(img, self.FB, self.FBC, self.F2B, self.FBFy, alpha=1/self.tau, sf=self.sf)
        elif self.degradation_mode == 'inpainting':
            if self.noise_level_img > 1e-2:
                proxf = (self.tau*self.My + img)/(self.tau*self.M+1)
            else :
                proxf = self.My + (1-self.M)*img
        return proxf
    def forward(self,x,sigma,create_graph):
        x.requires_grad_()
        Dx=self.rObj(x,sigma/255.,create_graph)
        z=(1.0-self.lamb*self.tau)*x+self.lamb*self.tau*Dx
        xnext=self.calculate_prox(z)
        return xnext
    def denoise(self,x,sigma,create_graph):
        Dx=self.rObj(x,sigma/255.,create_graph)
        z=(1.0-self.lamb*self.tau)*x+self.lamb*self.tau*Dx
        return z
