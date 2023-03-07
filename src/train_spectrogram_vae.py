from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.models.spectrogram_vae import SpectrogramVAE

DAFX_TO_USE = [
    # 'mda MultiBand',
    'clean',
    'mda Delay',
    'mda Overdrive',
    # # 'mda Ambience',
    'mda RingMod',
    # 'mda Leslie',
    # 'mda Combo',
    # 'mda Thru-Zero Flanger',
    # 'mda Loudness',
    # 'mda Limiter'
    'mda Dynamics',
]

SEED = 1234

if __name__ == "__main__":
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision('medium')

    # callbacks
    wandb_logger = WandbLogger(name='vctk_4dafx_plus_clean_random_settings', project='l5proj_spectrogram_vae')
    # wandb_logger = None

    checkpoint_callback = ModelCheckpoint(monitor="val_loss/loss", mode="min")
    early_stopping = EarlyStopping(
        monitor="val_loss/loss",
        mode="min",
        patience=100)

    # arg parse for config
    parser = ArgumentParser()

    # Add available trainer args and system args
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SpectrogramVAE.add_model_specific_args(parser)

    # Parse
    args = parser.parse_args()

    # Change settings for training
    args.input_dirs = ['vctk_24000']

    args.dafx_file = "/home/kieran/Level5ProjectAudioVAE/src/dafx/mda.vst3"
    args.dafx_names = DAFX_TO_USE
    args.audio_dir = "/home/kieran/Level5ProjectAudioVAE/src/audio"

    args.effect_audio = True
    args.dummy_setting = False
    args.normalise_audio = True

    args.num_channels = 1
    args.latent_dim = 256

    args.lr = 1e-4

    args.min_beta = 1e-3
    args.max_beta = 4.
    args.beta_start_epoch = 50
    args.beta_end_epoch = 250

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
        max_epochs=300,
        accelerator='gpu',
        gradient_clip_val=4.
    )

    # create the System
    system = SpectrogramVAE(**vars(args))

    print(system)

    # train!
    trainer.fit(system)
