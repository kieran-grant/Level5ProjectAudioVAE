from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.models.linear_vae import LinearVAE

DAFX_TO_USE = [
    # 'mda MultiBand',
    'clean',
    'mda Overdrive',
    # 'mda Ambience',
    'mda Delay',
    # 'mda Leslie',
    # 'mda Combo',
    # 'mda Thru-Zero Flanger',
    # 'mda Loudness',
    # 'mda Limiter'
]

SEED = 1234

if __name__ == "__main__":
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision('medium')

    # callbacks
    wandb_logger = WandbLogger(name='vctk_linear_test', project='l5proj_linear_vae')
    # wandb_logger = None

    checkpoint_callback = ModelCheckpoint(monitor="val_loss/loss", mode="min")
    early_stopping = EarlyStopping(
        monitor="val_loss/loss",
        mode="min",
        # should cycle through all effects at least twice before early stopping
        patience=20)

    # arg parse for config
    parser = ArgumentParser()

    # Add available trainer args and system args
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LinearVAE.add_model_specific_args(parser)

    # Parse
    args = parser.parse_args()

    # Change settings for training
    args.input_dirs = ['vctk_24000']

    args.train_examples_per_epoch = 5_000
    args.val_examples_per_epoch = 500

    args.dafx_file = "/home/kieran/Level5ProjectAudioVAE/src/dafx/mda.vst3"
    args.dafx_names = DAFX_TO_USE
    args.audio_dir = "/home/kieran/Level5ProjectAudioVAE/src/audio"

    args.effect_audio = False
    args.dummy_setting = True

    args.vae_beta = 0
    args.lr = 1e-4

    args.hidden_layer_dims = [1024, 1024, 1024]
    args.latent_space_dim = 1024

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
        max_epochs=100,
        accelerator='gpu',
    )

    # create the System
    system = LinearVAE(**vars(args))

    print(system)

    # train!
    trainer.fit(system)
