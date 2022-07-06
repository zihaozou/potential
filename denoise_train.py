import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
from dataset.data_module import DataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from os.path import join
from pytorch_lightning.strategies.ddp import DDPStrategy
from models.denoiser import Denoiser
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='/export/project/zihao/potential/test')
    parser.add_argument('--GPULst', type=int,nargs='+', default=[0])
    parser=Denoiser.add_denoiser_specific_args(parser)
    parser=DataModule.add_data_specific_args(parser)
    hparams = parser.parse_args()
    model=Denoiser(hparams)
    dm=DataModule(hparams)
    trainer = pl.Trainer.from_argparse_args(hparams, default_root_dir=hparams.exp_name,gpus=hparams.GPULst,
                                            resume_from_checkpoint=hparams.pretrained_checkpoint if hparams.resume_from_checkpoint else None,
                                            gradient_clip_val=hparams.gradient_clip_val, strategy = DDPStrategy(find_unused_parameters=False) if len(hparams.GPULst)>1  else None,
                                            max_epochs = hparams.max_epochs,num_sanity_val_steps=0,log_every_n_steps=10,val_check_interval=1.0)
    trainer.fit(model, dm)