from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torchsummary
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.callbacks.spectrogram_callback import LogSpectrogramCallback
from src.models.mel_spectrogram_vae import MelSpectrogramVAE

DAFX_TO_USE = [
    # 'mda MultiBand',
    # 'clean',
    'mda Delay',
    'mda Overdrive',
    # 'mda Ambience',
    'mda RingMod',
    # 'mda Leslie',
    # 'mda Combo',
    # 'mda Thru-Zero Flanger',
    # 'mda Loudness',
    # 'mda Limiter',
    'mda Dynamics',
]

SEED = 123
MAX_EPOCHS = 200

if __name__ == "__main__":
    wandb.require("service")
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision('medium')

    # arg parse for config
    parser = ArgumentParser()

    # Add available trainer args and system args
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MelSpectrogramVAE.add_model_specific_args(parser)

    # Parse
    args = parser.parse_args()

    # callbacks
    wandb_logger = WandbLogger(name='256x256_vtck_4fx', project='l5proj_melspec_vae')

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

    # Change settings for training
    args.input_dirs = ['vctk_24000']

    args.dafx_file = "/home/kieran/Level5ProjectAudioVAE/src/dafx/mda.vst3"
    args.dafx_names = DAFX_TO_USE
    args.audio_dir = "/home/kieran/Level5ProjectAudioVAE/src/audio"

    args.latent_dim = 128

    args.lr = 5e-4

    args.min_beta = 0.
    args.max_beta = 2.5e-4
    args.beta_start_epoch = 0
    args.beta_end_epoch = MAX_EPOCHS
    args.beta_cycle_length = 17

    # Set up trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        reload_dataloaders_every_n_epochs=1,
        logger=wandb_logger,
        callbacks=[
            LogSpectrogramCallback(),
            val_checkpoint,
            recon_checkpoint,
            kl_checkpoint,
            # early_stopping
        ],
        num_sanity_val_steps=0,
        max_epochs=MAX_EPOCHS,
        accelerator='gpu',
        gradient_clip_val=5.
    )

    # create the System
    system = MelSpectrogramVAE(**vars(args))

    print(torchsummary.summary(system, input_size=(1, 256, 256), device='cpu'))

    # train!
    trainer.fit(system)
