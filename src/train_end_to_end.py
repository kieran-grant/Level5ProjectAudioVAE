from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.models.end_to_end import EndToEndSystem

SEED = 1234

if __name__ == "__main__":
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision('medium')

    # callbacks
    # wandb_logger = WandbLogger(name='vctk_2dafx_no_clean_dummy', project='l5proj_end2end')
    wandb_logger = None

    checkpoint_callback = ModelCheckpoint(monitor="val_loss/loss", mode="min")
    early_stopping = EarlyStopping(
        monitor="val_loss/loss",
        mode="min",
        patience=40)

    # arg parse for config
    parser = ArgumentParser()

    # Add available trainer args and system args
    parser = pl.Trainer.add_argparse_args(parser)
    parser = EndToEndSystem.add_model_specific_args(parser)

    # Parse
    args = parser.parse_args()

    # Change settings for training
    args.input_dirs = ['vctk_24000']

    args.dafx_file = "/home/kieran/Level5ProjectAudioVAE/src/dafx/mda.vst3"
    args.dafx_name = "mda Overdrive"
    args.audio_dir = "/home/kieran/Level5ProjectAudioVAE/src/audio"

    args.audio_encoder_ckpt = "/home/kieran/Level5ProjectAudioVAE/src/l5proj_style_vae/ync68xdq/checkpoints/epoch=193-step=121250.ckpt"

    args.effect_input = False
    args.effect_output = True
    args.dummy_setting = True
    args.return_phase = False

    args.latent_dim = 2048

    args.lr = 3e-4
    args.max_epochs = 20

    # Set up trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        # reload_dataloaders_every_n_epochs=1,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            early_stopping
        ],
        num_sanity_val_steps=0,
        accelerator='cpu',
    )

    # create the System
    system = EndToEndSystem(**vars(args))

    print(system)

    # train!
    trainer.fit(system)
