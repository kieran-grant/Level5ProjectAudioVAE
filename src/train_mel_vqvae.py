from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.callbacks.spectrogram_callback import LogSpectrogramCallback
from src.models.vqvae.mel_spec_vqvae import MelSpecVQVAE

DAFX_TO_USE = [
    'mda MultiBand',
    # 'clean',
    'mda Delay',
    'mda Overdrive',
    # 'mda Ambience',
    'mda RingMod',
    # 'mda Leslie',
    # 'mda Combo',
    'mda Thru-Zero Flanger',
    # 'mda Loudness',
    # 'mda Limiter',
    # 'mda Dynamics',
]

SEED = 1234
MAX_EPOCHS = 30

if __name__ == "__main__":
    wandb.require("service")
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision('medium')

    # arg parse for config
    parser = ArgumentParser()

    # Add available trainer args and system args
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MelSpecVQVAE.add_model_specific_args(parser)

    # Parse
    args = parser.parse_args()

    # callbacks
    wandb_logger = WandbLogger(name=f'vtck_{len(DAFX_TO_USE)}fx_xxxsmall', project='l5proj_MelSpecVQVAE')

    val_checkpoint = ModelCheckpoint(
        monitor="val_loss/loss",
        filename="{epoch}-{step}",
        mode="min"
    )

    # Change settings for training
    args.input_dirs = ['vctk_24000']

    args.dafx_file = "/home/kieran/Level5ProjectAudioVAE/src/dafx/mda.vst3"
    args.dafx_names = DAFX_TO_USE
    args.audio_dir = "/home/kieran/Level5ProjectAudioVAE/src/audio"

    args.num_hiddens = 4
    args.num_residual_hiddens = 2
    args.num_residual_layers = 8
    args.embedding_dim = 4
    args.num_embeddings = 4096

    # Set up trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        reload_dataloaders_every_n_epochs=1,
        logger=wandb_logger,
        callbacks=[
            LogSpectrogramCallback(),
            val_checkpoint,
            # early_stopping
        ],
        num_sanity_val_steps=0,
        max_epochs=MAX_EPOCHS,
        accelerator='gpu',
        gradient_clip_val=3.
    )

    # create the System
    system = MelSpecVQVAE(**vars(args))

    print(system)

    # train!
    trainer.fit(system)
