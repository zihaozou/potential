import torch
import torch.nn as nn
import pytorch_lightning as pl
from gspnp.lightning_GSDRUNet import StudentGrad
from pnp import PNP
from deqFixedPoint import DEQFixedPoint,nesterov,anderson
from torchmetrics import PSNR
from torch.nn.functional import mse_loss,conv2d,pad
from hdf5storage import loadmat
from random import choice
import torchvision
from torch.optim import Adam
from torch.optim import lr_scheduler
from argparse import ArgumentParser
class PotentialDEQ(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        if self.hparams.potential=='gspnp':
            model = StudentGrad(network=self.hparams.network,numInChan=self.hparams.numInChan,numOutChan=self.hparams.numOutChan)
        elif self.hparams.potential=='potential':
            pass #TODO 加入 red potential function
        f=PNP(self.hparams.tau,self.hparams.lamb,model,self.hparams.degradation_mode,self.hparams.sf)
        self.deq=DEQFixedPoint(f,nesterov,anderson)
        self.kernels=loadmat(self.hparams.kernel_path)['kernels']
        self.train_PSNR=PSNR(data_range=1.0)
        self.val_PSNRLst=[PSNR(data_range=1.0) for _ in range(len(self.hparams.sigma_test_list)*len(self.hparams.kernelLst))]


    def forward(self, n_y,kernel):
        return self.deq(n_y,kernel)
    def training_step(self, batch, batch_idx):
        gtImg = batch
        kernel=self.kernels[0,choice(self.hparams.kernelLst)]
        kernelTensor=torch.tensor(kernel,dtype=torch.float32,device=gtImg.device)
        kernelTensor=kernelTensor.unsqueeze(0).unsqueeze(0)
        kernelTensor=kernelTensor.expand(gtImg.shape[1],gtImg.shape[1],kernelTensor.shape[2],kernelTensor.shape[3])
        degradImg=conv2d(pad(gtImg,(kernelTensor.shape[3]//2,kernelTensor.shape[3]//2,kernelTensor.shape[2]//2,kernelTensor.shape[2]//2),mode='circular'),kernelTensor)
        noise=torch.FloatTensor(degradImg.size()).normal_(mean=0, std=self.hparams.sigma/255.)
        degradImg=degradImg+noise
        reconImg=self(degradImg,kernel)
        loss=mse_loss(reconImg,gtImg)
        self.log('train_loss',loss.detach(), prog_bar=False)
        self.train_PSNR.update(gtImg,reconImg)
        psnr=self.train_PSNR.compute()
        self.log('train_psnr',psnr.detach(), prog_bar=True)
        return {'loss':loss}
    def training_epoch_end(self, outputs) -> None:
        self.train_PSNR.reset()
    def validation_step(self, batch, batch_idx):
        batch_dict = {}
        gtImg = batch
        sigma_list = self.hparams.sigma_test_list
        kernelLst=self.hparams.kernelLst
        for i, kernelIdx in enumerate(kernelLst):
            for j, sigma in enumerate(sigma_list):
                kernel=self.kernels[0,kernelIdx]
                kernelTensor=torch.tensor(kernel,dtype=torch.float32,device=gtImg.device)
                kernelTensor=kernelTensor.unsqueeze(0).unsqueeze(0)
                kernelTensor=kernelTensor.expand(gtImg.shape[1],gtImg.shape[1],kernelTensor.shape[2],kernelTensor.shape[3])
                degradImg=conv2d(pad(gtImg,(kernelTensor.shape[3]//2,kernelTensor.shape[3]//2,kernelTensor.shape[2]//2,kernelTensor.shape[2]//2),mode='circular'),kernelTensor,padding='valid')
                noise=torch.FloatTensor(degradImg.size()).normal_(mean=0, std=sigma/255.)
                degradImg=degradImg+noise
                reconImg=self(degradImg,kernel)
                self.val_PSNR[i*len(sigma_list)+j].update(gtImg,reconImg)
                if batch_idx == 0: # logging for tensorboard
                    clean_grid = torchvision.utils.make_grid(gtImg.detach(),normalize=True)
                    noisy_grid = torchvision.utils.make_grid(degradImg.detach(),normalize=True)
                    recon_grid = torchvision.utils.make_grid(reconImg.detach(),normalize=True)
                    self.logger.experiment.add_image(f'val_image/clean/kernel-{i}/sigma-{j}', clean_grid, self.current_epoch)
                    self.logger.experiment.add_image(f'val_image/noisy/kernel-{i}/sigma-{j}', noisy_grid, self.current_epoch)
                    self.logger.experiment.add_image(f'val_image/recon/kernel-{i}/sigma-{j}', recon_grid, self.current_epoch)
    def validation_epoch_end(self, outputs) -> None:
        sigma_list = self.hparams.sigma_test_list
        kernelLst=self.hparams.kernelLst
        for i, kernelIdx in enumerate(kernelLst):
            for j, sigma in enumerate(sigma_list):
                psnr=self.val_PSNR[i*len(sigma_list)+j].compute()
                self.log(f'val_psnr/kernel-{str(i)}/sigma-{str(j)}',psnr.detach(), prog_bar=False)
            self.val_PSNR.reset()
        
    def configure_optimizers(self):
        optim_params = self.deq.parameters()
        
        optimizer = Adam(optim_params, lr=self.hparams.optimizer_lr, weight_decay=0)
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             self.hparams.scheduler_milestones,
                                             self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--kernel_path', type=str, default='miscs/Levin09.mat', help='path to kernel mat file')
        parser.add_argument('--potential', type=str, default='gspnp', help='select potential function')
        parser.add_argument('--network', type=str, default='dncnn', help='select network')
        parser.add_argument('--numInChan', type=int, default=3, help='number of input channels')
        parser.add_argument('--numOutChan', type=int, default=3, help='number of output channels')
        parser.add_argument('--tau', type=float, default=0.5, help='regularization parameter')
        parser.add_argument('--sigma', type=float, default=1.0, help='noise level')
        parser.add_argument('--lamb', type=float, default=1.0, help='regularization parameter')
        parser.add_argument('--degradation_mode', type=str, default='deblurring', choices=['deblurring','SR','inpainting'],help='select degradation mode')
        #SR
        parser.add_argument('--sf', type=int, default=2)
        ##
        parser.add_argument('--kernelLst', type=int, nargs='+', default=[1,3], help='list of kernel indices')
        parser.add_argument('--sigma_test_list', type=float,nargs='+', default=[1.0,5.0,10.0], help='list of sigma values')
        parser.add_argument('--optimizer_lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--scheduler_milestones', type=int, nargs='+', default=[50,100,150], help='milestones for scheduler')
        parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='gamma for scheduler')
        parser.add_argument('--resume_from_checkpoint', dest='resume_from_checkpoint', action='store_true')
        parser.set_defaults(resume_from_checkpoint=False)
        parser.add_argument('--pretrained_checkpoint', type=str,default='')
        return parser
