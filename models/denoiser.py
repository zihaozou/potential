import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from .dpirUnet import DPIRNNclass,PotentialNNclass,REDPotentialNNclass,GSPNPNNclass
from torchmetrics import PeakSignalNoiseRatio as PSNR
from random import uniform,choice
from os.path import join
from PIL.Image import open as imopen
import torchvision
from torchvision.transforms.functional import to_tensor
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.optim import Adam
from torch.optim import lr_scheduler
from torchvision.transforms.functional import center_crop
class Denoiser(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        if self.hparams.potential=='gspnp':
            self.model=GSPNPNNclass(numInChan=self.hparams.numInChan,numOutChan=self.hparams.numOutChan,network=self.hparams.denoiser_name,train_network=True)
        elif self.hparams.potential=='red_potential':
            self.model=REDPotentialNNclass(numInChan=self.hparams.numInChan,numOutChan=self.hparams.numOutChan,network=self.hparams.denoiser_name,train_network=True)
        elif self.hparams.potential=='dpir':
            self.model=DPIRNNclass(numInChan=self.hparams.numInChan,numOutChan=self.hparams.numOutChan,network=self.hparams.denoiser_name,train_network=True)
        elif self.hparams.potential=='potential':
            self.model=PotentialNNclass(numInChan=self.hparams.numInChan,numOutChan=self.hparams.numOutChan,network=self.hparams.denoiser_name,train_network=True)
        self.train_PSNR=PSNR(data_range=1.0)
        for i in range(len(self.hparams.sigma_test_list)):
            exec('self.val_PSNR_%d=PSNR(data_range=1.0)'%i)
        if hparams.loss == 'mse':
            self.lossFunc= torch.nn.MSELoss()
        elif hparams.loss == 'l1':
            self.lossFunc= torch.nn.L1Loss()
        self.testNames=['butterfly.png','leaves.png','starfish.png']
        testLst=[to_tensor(imopen(join('miscs','set3c',n)).convert('RGB')) for n in self.testNames]
        self.testTensor=torch.stack(testLst,dim=0)
        self.cropLst=[(32,32),(64,64),(128,128),(256,256)]
    def forward(self, x,sigma,create_graph=True,strict=True):
        return self.model(x,sigma,create_graph=create_graph,strict=strict)

    def training_step(self, batch, batch_idx):
        gtImg,_ = batch
        gtImg=center_crop(gtImg,choice(self.cropLst))
        sigma= uniform(self.hparams.sigma_min,self.hparams.sigma_max)/255.0
        noise=torch.randn_like(gtImg)*sigma
        noisyImg=gtImg+noise
        predImg=self(noisyImg,torch.tensor([sigma],dtype=gtImg.dtype,device=gtImg.device),create_graph=True,strict=True)
        loss=self.lossFunc(gtImg,predImg)
        self.log('train_loss',loss.detach(), prog_bar=False,on_step=True,logger=True)
        self.train_PSNR.update(gtImg,predImg)
        psnr=self.train_PSNR.compute().detach()
        self.log('train_PSNR',psnr, prog_bar=True,on_step=True,logger=True)
        self.train_PSNR.reset()
        return {'loss':loss,'psnr':psnr}

    def training_epoch_end(self, outputs) -> None:
        torch.save(self.model.network.state_dict(),join(self.hparams.exp_name,f'epoch_{self.current_epoch}.pt'))

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        gtImgs,_ = batch
        sigma_list = self.hparams.sigma_test_list
        for i in range(len(sigma_list)):
            sigma=sigma_list[i]/255.0
            noise=torch.randn_like(gtImgs)*sigma
            noisyImgs=gtImgs+noise
            predImgs=self(noisyImgs,torch.tensor([sigma],dtype=gtImgs.dtype,device=gtImgs.device),create_graph=False,strict=False)
            predNoise=gtImgs-predImgs
            loss=self.lossFunc(predNoise,noise)
            exec('self.val_PSNR_%d.update(gtImgs,predImgs)'%i)
            self.log('val_loss_%d'%i,loss.detach(), prog_bar=False,on_step=False,logger=True)
        return {'val_loss':loss}
    def validation_epoch_end(self, outputs):
        sigma_list = self.hparams.sigma_test_list
        for j, sigma in enumerate(sigma_list):
            exec('psnr1=self.val_PSNR_%d.compute()'%j)
            exec(f'self.log(f"val_psnr_{sigma}", psnr1.detach(), prog_bar=False,logger=True)')
            exec('self.val_PSNR_%d.reset()'%j)
        testTensor=self.testTensor.to(self.device)
        for i in range(len(self.hparams.sigma_test_list)):
            sigma=self.hparams.sigma_test_list[i]/255.0
            noise=torch.randn_like(testTensor)*sigma
            noisyImgs=testTensor+noise
            predImgs=self(noisyImgs,torch.tensor([sigma],dtype=testTensor.dtype,device=testTensor.device),create_graph=False,strict=False)
            for j in range(len(testTensor)):
                gtImg=testTensor[j]
                predImg=predImgs[j]
                outpsnr=psnr(gtImg.detach().cpu().numpy().transpose(1,2,0),predImg.detach().cpu().numpy().transpose(1,2,0),data_range=1.0)
                self.log(f'test_PSNR_sigma:{sigma},img_{self.testNames[j]}',outpsnr, prog_bar=False,on_step=False,logger=True)
            clean_grid = torchvision.utils.make_grid(testTensor.detach(),normalize=True,nrow=2)
            noisy_grid = torchvision.utils.make_grid(noisyImgs.detach(),normalize=True,nrow=2)
            recon_grid = torchvision.utils.make_grid(torch.clamp(predImgs,min=0.0,max=1.0).detach(),normalize=False,nrow=2)
            self.logger.experiment.add_image(f'test_image/clean/sigma-{sigma}', clean_grid, self.current_epoch)
            self.logger.experiment.add_image(f'test_image/noisy/sigma-{sigma}', noisy_grid, self.current_epoch)
            self.logger.experiment.add_image(f'test_image/recon/sigma-{sigma}', recon_grid, self.current_epoch)
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.hparams.optimizer_lr,weight_decay=0)
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             self.hparams.scheduler_milestones,
                                             self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]
    @staticmethod
    def add_denoiser_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--denoiser_name', type=str, default='dncnn')
        parser.add_argument('--potential', type=str, default='red_potential')
        parser.add_argument('--numInChan', type=int, default=3)
        parser.add_argument('--numOutChan', type=int, default=3)
        parser.add_argument('--sigma_min', type=float, default=2.55)
        parser.add_argument('--sigma_max', type=float, default=12.75)
        parser.add_argument('--sigma_test_list', type=float, nargs='+', default=[2.55,7.5,12.75])
        parser.add_argument('--loss', type=str, default='mse')
        parser.add_argument('--resume_from_checkpoint', dest='resume_from_checkpoint', action='store_true')
        parser.set_defaults(resume_from_checkpoint=False)
        parser.add_argument('--pretrained_checkpoint', type=str,default='')
        parser.add_argument('--gradient_clip_val', type=float, default=1e-2)
        parser.add_argument('--scheduler_milestones', type=int, nargs='+', default=[180, 240, 300, 360, 420])
        parser.add_argument('--scheduler_gamma', type=float, default=0.75)
        parser.add_argument('--optimizer_lr', type=float, default=1e-4)
        return parser
