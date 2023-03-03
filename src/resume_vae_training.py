from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.models.style_transfer_vae import StyleTransferVAE

SEED = 1234
PATH_TO_CHECKPOINT = "/home/kieran/Level5ProjectAudioVAE/src/l5proj_style_vae/renudd7j/checkpoints/epoch=1212-step=758125.ckpt"

if __name__ == "__main__":
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision('medium')

    # callbacks
    wandb_logger = WandbLogger(name='vctk_4dafx_random_settings', project='l5proj_style_vae')

    checkpoint_callback = ModelCheckpoint(monitor="val_loss/loss", mode="min")
    early_stopping = EarlyStopping(
        monitor="val_loss/loss",
        mode="min",
        patience=200)

    # arg parse for config
    parser = ArgumentParser()

    # Add available trainer args and system args
    parser = pl.Trainer.add_argparse_args(parser)

    # Parse
    args = parser.parse_args()

    system = StyleTransferVAE.load_from_checkpoint(PATH_TO_CHECKPOINT)

    system.hparams.vae_beta = 1
    system.hparams.lr = 5e-5

    # Set up trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        reload_dataloaders_every_n_epochs=1,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            early_stopping
        ],
        num_sanity_val_steps=0,
        max_epochs=1300,
        accelerator='gpu',
    )

    # create the System
    print(system)

    # train!
    trainer.fit(system, ckpt_path=PATH_TO_CHECKPOINT)
