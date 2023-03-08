from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.models.spectrogram_vae import SpectrogramVAE

SEED = 1234
CHECKPOINT = "/home/kieran/Level5ProjectAudioVAE/src/l5proj_spectrogram_vae/k20ypr7z/checkpoints/epoch=37-step=7942.ckpt"
MAX_EPOCHS = 400

if __name__ == "__main__":
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision('medium')

    # arg parse for config
    parser = ArgumentParser()

    # Add available trainer args and system args
    parser = pl.Trainer.add_argparse_args(parser)

    # Parse
    args = parser.parse_args()

    # callbacks
    wandb_logger = WandbLogger(name='vctk_4dafx_plus_clean_random_settings', project='l5proj_spectrogram_vae')

    val_checkpoint = ModelCheckpoint(
        monitor="val_loss/loss",
        filename="{epoch}-{step}",
        mode="min"
    )
    recon_checkpoint = ModelCheckpoint(
        monitor="val_loss/reconstruction_loss",
        filename="best_recon-{epoch}-{step}",
        mode="min"
    )
    kl_checkpoint = ModelCheckpoint(
        monitor="val_loss/kl_divergence",
        filename="best_kldiv-{epoch}-{step}",
        mode="min"
    )
    # early_stopping = EarlyStopping(
    #     monitor="val_loss/loss",
    #     mode="min",
    #     patience=200)

    system = SpectrogramVAE.load_from_checkpoint(CHECKPOINT)

    args.beta_end_epoch = MAX_EPOCHS

    # Set up trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        reload_dataloaders_every_n_epochs=1,
        logger=wandb_logger,
        callbacks=[
            val_checkpoint,
            recon_checkpoint,
            kl_checkpoint,
            # early_stopping
        ],
        num_sanity_val_steps=0,
        max_epochs=MAX_EPOCHS,
        accelerator='gpu',
        gradient_clip_val=4.
    )

    # create the System
    print(system)

    # train!
    trainer.fit(system, ckpt_path=CHECKPOINT)
