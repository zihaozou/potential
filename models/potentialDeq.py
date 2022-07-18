import torch
import torch.nn as nn
import pytorch_lightning as pl
from .pnp import PNP
from .deqFixedPoint import DEQFixedPoint,nesterov,anderson, simpleIter
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torch.nn.functional import mse_loss,conv2d,pad
from hdf5storage import loadmat
from random import choice,uniform
import torchvision
from torch.optim import Adam
from torch.optim import lr_scheduler
from argparse import ArgumentParser
from scipy import ndimage
import numpy as np
from os.path import join
from PIL.Image import open as imopen
from .dpirUnet import DPIRNNclass,REDPotentialNNclass,PotentialNNclass,GSPNPNNclass
from skimage.metrics import peak_signal_noise_ratio as skpsnr
from utils.utils_restoration import matlab_style_gauss2D
class PotentialDEQ(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        if self.hparams.potential=='gspnp':
            model=GSPNPNNclass(numInChan=self.hparams.numInChan,numOutChan=self.hparams.numOutChan,network=self.hparams.network,train_network=self.hparams.train_network)
        elif self.hparams.potential=='red_potential':
            model=REDPotentialNNclass(numInChan=self.hparams.numInChan,numOutChan=self.hparams.numOutChan,network=self.hparams.network,train_network=self.hparams.train_network)
        elif self.hparams.potential=='dpir':
            model=DPIRNNclass(numInChan=self.hparams.numInChan,numOutChan=self.hparams.numOutChan,network=self.hparams.network,train_network=self.hparams.train_network)
        elif self.hparams.potential=='potential':
            model=PotentialNNclass(numInChan=self.hparams.numInChan,numOutChan=self.hparams.numOutChan,network=self.hparams.network,train_network=self.hparams.train_network)
        f=PNP(self.hparams.lamb,model,self.hparams.train_tau_lamb,self.hparams.degradation_mode)
        self.deq=DEQFixedPoint(f,simpleIter,anderson,self.hparams.jbf,self.hparams.sigmaFactor,self.hparams.train_sigmaFactor)
        if self.hparams.enable_pretrained_denoiser:
            self.deq.f.rObj.network.load_state_dict(torch.load(self.hparams.pretrained_denoiser,map_location=torch.device('cpu')))
        self.kernels=loadmat(self.hparams.kernel_path)['kernels']
        self.train_PSNR=PSNR(data_range=1.0)
        for i in range(len(self.hparams.sigma_test_list)*len(self.hparams.kernelLst)):
            exec('self.val_PSNR_%d=PSNR(data_range=1.0)'%i)
        testLst=[]
        imLst=['butterfly.png','leaves.png','starfish.png']
        for i in range(3):
            testLst.append(torch.tensor(np.asarray(imopen(join('miscs','set3c',imLst[i]))),dtype=torch.float32).permute(2,0,1)/255.)
        self.testTensor=torch.stack(testLst,dim=0)
        self.val_PSNR_butterfly=PSNR(data_range=1.0)
        self.val_PSNR_leaves=PSNR(data_range=1.0)
        self.val_PSNR_starfish=PSNR(data_range=1.0)
        # self.input_im = np.asarray(imopen('/export1/project/zihao/GSPnP/PnP_restoration/SR/GS-DRUNet/HQS/CBSD10/7.65/kernel_0/images/img_0_input.png'),dtype=np.float32)/255.0

        # self.blur_im = np.asarray(imopen('/export1/project/zihao/GSPnP/PnP_restoration/SR/GS-DRUNet/HQS/CBSD10/7.65/kernel_0/images/img_0_GSPnP.png'),dtype=np.float32)/255.0
    def forward(self, n_y,kernel,sigma,gtImg,degradMode,sf):
        return self.deq(n_y,kernel,sigma,gtImg,degradMode,sf)
    def selectKernel(self,kernelIdx):
        if self.hparams.kernel_path.find('kernels_12.mat') != -1:
            k = self.kernels[0, kernelIdx]
        else:
            if kernelIdx == 8: # Uniform blur
                k = (1/81)*np.ones((9,9))
            elif kernelIdx == 9:  # Gaussian blur
                k = matlab_style_gauss2D(shape=(25,25),sigma=1.6)
            else: # Motion blur
                k = self.kernels[0, kernelIdx]
        return k
    def makeDegrad(self,gt,kIdx,sigma,sf):
        if self.hparams.degradation_mode=='deblurring':
            kernel=self.selectKernel(kIdx)
            degradLst=[ndimage.filters.convolve(gt[i,...].permute(1,2,0).cpu().numpy(), np.expand_dims(kernel, axis=2), mode='wrap')+np.random.normal(0, sigma/255., (gt.shape[2],gt.shape[3],gt.shape[1])) for i in range(gt.shape[0])]
            degradImg=torch.tensor(np.stack(degradLst,axis=0),dtype=torch.float32,device=gt.device).permute(0,3,1,2)
        elif self.hparams.degradation_mode=='SR':
            kernel=self.selectKernel(kIdx)
            degradLst=[ndimage.filters.convolve(self.modcrop(gt[i,...].permute(1,2,0).cpu().numpy(),sf), np.expand_dims(kernel, axis=2), mode='wrap')[0::sf,0::sf,...]+np.random.normal(0, sigma/255., (gt.shape[2]//sf,gt.shape[3]//sf,gt.shape[1])) for i in range(gt.shape[0])]
            degradImg=torch.tensor(np.stack(degradLst,axis=0),dtype=torch.float32,device=gt.device).permute(0,3,1,2)
        elif self.hparams.degradation_mode=='inpainting':
            kernel=torch.bernoulli(torch.tensor(0.5,dtype=torch.float32,device=gt.device).expand(gt.shape[0],1,gt.shape[2],gt.shape[3])).expand(gt.shape[0],3,gt.shape[2],gt.shape[3])
            degradImg=gt*kernel + (0.5)*(1-kernel)
        return degradImg,kernel

        
    def modcrop(self,img_in, scale):
    # img_in: Numpy, HWC or HW
        img = img_in
        if img.ndim == 2:
            H, W = img.shape
            H_r, W_r = H % scale, W % scale
            img = img[:H - H_r, :W - W_r]
        elif img.ndim == 3:
            H, W, C = img.shape
            H_r, W_r = H % scale, W % scale
            img = img[:int(H-H_r), :int(W-W_r), :]
        else:
            raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
        return img
    def training_step(self, batch, batch_idx):
        gtImg,_ = batch
        sf=choice(self.hparams.sf)
        sigma=uniform(self.hparams.sigma_min,self.hparams.sigma_max)
        degradImg,kernel=self.makeDegrad(gtImg,choice(self.hparams.kernelLst),sigma,sf)
        reconImg=self(degradImg,kernel,sigma,gtImg,self.hparams.degradation_mode,sf)
        loss=mse_loss(reconImg,gtImg)
        self.log('train_loss',loss.detach(), prog_bar=False,on_step=True,logger=True)
        self.train_PSNR.update(gtImg,reconImg)
        psnr=self.train_PSNR.compute()
        self.train_PSNR.reset()
        self.log('train_psnr',psnr.detach(), prog_bar=True,on_step=True,logger=True)
        self.log('lamb',self.deq.f.lamb.detach(), prog_bar=False,on_step=True,logger=True)
        self.log('sigma factor',self.deq.sigmaFactor.detach(), prog_bar=False,on_step=True,logger=True)
        return {'loss':loss}

    def training_epoch_end(self, outputs) -> None:
        
        torch.save(self.deq.state_dict(),join(self.hparams.exp_name,f'epoch_{self.current_epoch}.pt'))
    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        gtImg,_ = batch
        sigma_list = self.hparams.sigma_test_list
        kernelLst=self.hparams.kernelLst
        for i, kernelIdx in enumerate(kernelLst):
            for j, sigma in enumerate(sigma_list):
                for k,sf in enumerate(self.hparams.sf):
                    degradImg,kernel=self.makeDegrad(gtImg,kernelIdx,sigma,sf)
                    reconImg=self(degradImg,kernel,sigma,gtImg,self.hparams.degradation_mode,sf).detach()
                    exec('self.val_PSNR_%d.update(gtImg,reconImg)'%(i*len(sigma_list)+j))
                    if batch_idx == 0: # logging for tensorboard
                        clean_grid = torchvision.utils.make_grid(gtImg.detach(),normalize=True,nrow=2)
                        noisy_grid = torchvision.utils.make_grid(degradImg.detach(),normalize=True,nrow=2)
                        recon_grid = torchvision.utils.make_grid(torch.clamp(reconImg,min=0.0,max=1.0).detach(),normalize=False,nrow=2)
                        self.logger.experiment.add_image(f'val_image/clean/kernel-{kernelIdx}/sigma-{sigma}/sf-{sf}', clean_grid, self.current_epoch)
                        self.logger.experiment.add_image(f'val_image/noisy/kernel-{kernelIdx}/sigma-{sigma}/sf-{sf}', noisy_grid, self.current_epoch)
                        self.logger.experiment.add_image(f'val_image/recon/kernel-{kernelIdx}/sigma-{sigma}/sf-{sf}', recon_grid, self.current_epoch)
    def validation_epoch_end(self, outputs) -> None:
        sigma_list = self.hparams.sigma_test_list
        kernelLst=self.hparams.kernelLst
        for i, kernelIdx in enumerate(kernelLst):
            for j, sigma in enumerate(sigma_list):
                exec('psnr1=self.val_PSNR_%d.compute()'%(i*len(sigma_list)+j))
                exec(f'self.log(f"val_psnr_kernel-{kernelIdx}_sigma-{sigma}", psnr1.detach(), prog_bar=False,logger=True)')
                exec('self.val_PSNR_%d.reset()'%(i*len(sigma_list)+j))
        # kernel=self.kernels[0,0]
        # sigma=7.65
        # testTensor=torch.tensor(self.input_im,dtype=torch.float32,device=self.device).permute(2,0,1).unsqueeze(0)
        # testTensorBlur=torch.tensor(self.blur_im,dtype=torch.float32,device=self.device).permute(2,0,1).unsqueeze(0)
        # reconImg=self(testTensorBlur,kernel,sigma,testTensor,self.hparams.degradation_mode,self.hparams.sf).detach()
        testTensor=self.testTensor.to(self.device)
        for i, kernelIdx in enumerate(kernelLst):
            for j, sigma in enumerate(sigma_list):
                for k,sf in enumerate(self.hparams.sf):
                    degradImg,kernel=self.makeDegrad(testTensor,kernelIdx,sigma,sf)
                    #print(skpsnr(testTensor[0,...].detach().permute(1,2,0).cpu().numpy(),degradImg[0,...].detach().permute(1,2,0).cpu().numpy(),data_range=1.0))
                    reconImg=self(degradImg,kernel,sigma,testTensor,self.hparams.degradation_mode,sf).detach()
                    #print(f'test image psnr: {skpsnr(testTensor[0,...].detach().permute(1,2,0).cpu().numpy(),reconImg[0,...].detach().permute(1,2,0).cpu().numpy(),data_range=1.0)}')
                    self.val_PSNR_butterfly.update(testTensor[0,...].unsqueeze(0),reconImg[0,...].unsqueeze(0))
                    psnr=self.val_PSNR_butterfly.compute()
                    self.log(f'val_butterfly_psnr_kernel-{kernelIdx}_sigma-{sigma}',psnr.detach())
                    self.val_PSNR_butterfly.reset()

                    self.val_PSNR_leaves.update(testTensor[1,...].unsqueeze(0),reconImg[1,...].unsqueeze(0))
                    psnr=self.val_PSNR_leaves.compute()
                    self.log(f'val_leaves_psnr_kernel-{kernelIdx}_sigma-{sigma}',psnr.detach())
                    self.val_PSNR_leaves.reset()

                    self.val_PSNR_starfish.update(testTensor[2,...].unsqueeze(0),reconImg[2,...].unsqueeze(0))
                    psnr=self.val_PSNR_starfish.compute()
                    self.log(f'val_starfish_psnr_kernel-{kernelIdx}_sigma-{sigma}',psnr.detach())
                    self.val_PSNR_starfish.reset()

                    clean_grid = torchvision.utils.make_grid(testTensor.detach(),normalize=True,nrow=2)
                    noisy_grid = torchvision.utils.make_grid(degradImg.detach(),normalize=True,nrow=2)
                    recon_grid = torchvision.utils.make_grid(torch.clamp(reconImg,min=0.0,max=1.0).detach(),normalize=False,nrow=2)
                    self.logger.experiment.add_image(f'test_image/clean/kernel-{kernelIdx}/sigma-{sigma}/sf-{sf}', clean_grid, self.current_epoch)
                    self.logger.experiment.add_image(f'test_image/noisy/kernel-{kernelIdx}/sigma-{sigma}/sf-{sf}', noisy_grid, self.current_epoch)
                    self.logger.experiment.add_image(f'test_image/recon/kernel-{kernelIdx}/sigma-{sigma}/sf-{sf}', recon_grid, self.current_epoch)
    def configure_optimizers(self):
        optim_params=[]
        if self.hparams.train_network:
            optim_params.append({'params': self.deq.f.rObj.parameters(), 'lr': self.hparams.network_lr})
        if self.hparams.train_sigmaFactor:
            optim_params.append({'params': self.deq.sigmaFactor,'lr':self.hparams.sigmaFactor_lr})
        if self.hparams.train_tau_lamb:
            optim_params.append({'params': self.deq.f.lamb,'lr':self.hparams.tau_lamb_lr})
        optimizer = Adam(optim_params, weight_decay=1e-8)
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             self.hparams.scheduler_milestones,
                                             self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--kernel_path', type=str, default='miscs/Levin09.mat', help='path to kernel mat file')
        parser.add_argument('--potential', type=str, default='gspnp', help='select potential function')
        parser.add_argument('--network', type=str, default='unet', help='select network')
        parser.add_argument('--numInChan', type=int, default=3, help='number of input channels')
        parser.add_argument('--numOutChan', type=int, default=3, help='number of output channels')
        parser.add_argument('--sigma_min', type=float, default=2.55, help='noise level')
        parser.add_argument('--sigma_max', type=float, default=7.65, help='noise level')
        parser.add_argument('--lamb', type=float, default=0.1, help='regularization parameter')
        parser.add_argument('--sigmaFactor', type=float, default=2.0, help='sigma factor')
        
        parser.add_argument('--degradation_mode', type=str, default='deblurring', choices=['deblurring','SR','inpainting'],help='select degradation mode')
        #SR
        parser.add_argument('--sf', type=int, nargs='+', default=[2])
        ##
        parser.add_argument('--n_init', type=int, default=10)
        parser.add_argument('--prop_mask', type=float, default=0.5)
        parser.add_argument('--kernelLst', type=int, nargs='+', default=[1,3], help='list of kernel indices')
        parser.add_argument('--sigma_test_list', type=float,nargs='+', default=[2.55,7.65], help='list of sigma values')
        parser.add_argument('--no_train_network',dest='train_network',action='store_false')
        parser.set_defaults(train_network=True)
        parser.add_argument('--network_lr', type=float, default=1e-5, help='network learning rate')
        parser.add_argument('--train_sigmaFactor',dest='train_sigmaFactor',action='store_true')
        parser.set_defaults(train_sigmaFactor=False)
        parser.add_argument('--sigmaFactor_lr', type=float, default=1e-3, help='sigma factor learning rate')
        parser.add_argument('--train_tau_lamb',dest='train_tau_lamb',action='store_true')
        parser.set_defaults(train_tau_lamb=False)
        parser.add_argument('--tau_lamb_lr', type=float, default=1e-3, help='tau lambda learning rate')
        parser.add_argument('--scheduler_milestones', type=int, nargs='+', default=[10,20,25], help='milestones for scheduler')
        parser.add_argument('--scheduler_gamma', type=float, default=0.8, help='gamma for scheduler')
        parser.add_argument('--resume_from_checkpoint', dest='resume_from_checkpoint', action='store_true')
        parser.set_defaults(resume_from_checkpoint=False)
        parser.add_argument('--pretrained_checkpoint', type=str,default='')
        parser.add_argument('--gradient_clip_val', type=float, default=1e-2)
        parser.add_argument('--pretrained_denoiser', type=str, default='')
        parser.add_argument('--enable_pretrained_denoiser', dest='enable_pretrained_denoiser', action='store_true')
        parser.set_defaults(enable_pretrained_denoiser=False)
        parser.add_argument('--jbf',dest='jbf',action='store_true')
        parser.set_defaults(jbf=False)
        return parser
