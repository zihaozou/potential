import pytorch_lightning as pl
from models.potentialDeq import PotentialDEQ
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
from dataset.data_module import DataModule
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='/export/project/zihao/potential/test')
    parser.add_argument('--GPULst', type=int,nargs='+', default=[0])
    parser=PotentialDEQ.add_model_specific_args(parser)
    parser=DataModule.add_data_specific_args(parser)
    hparams = parser.parse_args()
    model=PotentialDEQ(hparams)
    dm=DataModule(hparams)
    trainer = pl.Trainer.from_argparse_args(hparams, default_root_dir=hparams.exp_name,gpus=hparams.GPULst,
                                            resume_from_checkpoint=hparams.pretrained_checkpoint if hparams.resume_from_checkpoint else None,
                                            gradient_clip_val=hparams.gradient_clip_val, accelerator='ddp' if len(hparams.GPULst)>1 else None,
                                            max_epochs = hparams.max_epochs,num_sanity_val_steps=1,log_every_n_steps=10)
    trainer.fit(model, dm)