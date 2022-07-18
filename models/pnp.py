from turtle import forward
import torch
import torch.nn as nn
from utils import utils_sr
from utils.utils_restoration import array2tensor
import numpy as np
from .dpirUnet import DPIRNNclass,GSPNPNNclass,REDPotentialNNclass,PotentialNNclass

def get_rho_sigma(sigma=2.55/255, iter_num=15, modelSigma1=49.0, modelSigma2=2.55, w=1.0):
    '''
    One can change the sigma to implicitly change the trade-off parameter
    between fidelity term and prior term
    '''
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
    modelSigmaS_lin = np.linspace(modelSigma1, modelSigma2, iter_num).astype(np.float32)
    sigmas = (modelSigmaS*w+modelSigmaS_lin*(1-w))/255.
    rhos = list(map(lambda x: 0.23*(sigma**2)/(x**2), sigmas))
    return rhos, sigmas

class PNP(nn.Module):
    def __init__(self,lamb,rObj,train_tau_lamb,degradation_mode):
        super(PNP,self).__init__()
        self.rObj=rObj
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
        self.sf=sf
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
    def forward(self,x,sigma,create_graph=True,strict=True):
        if isinstance(self.rObj,DPIRNNclass):
            vnext=self.calculate_prox(x)
            xnext=self.rObj(vnext,sigma/255.,create_graph=create_graph,strict=strict)
        else:
            Dx=self.rObj(x,sigma/255.,create_graph=create_graph,strict=strict)
            z=(1.0-self.lamb*self.tau)*x+self.lamb*self.tau*Dx
            xnext=self.calculate_prox(z)
        return xnext
    def denoise(self,x,sigma,create_graph=True,strict=True):
        Dx=self.rObj(x,sigma/255.,create_graph=create_graph,strict=strict)
        z=(1.0-self.lamb*self.tau)*x+self.lamb*self.tau*Dx
        return z
    @property
    def tau(self):
        return 1/self.lamb
