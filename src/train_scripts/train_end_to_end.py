from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.callbacks.audio import LogAudioCallback
from src.models.end_to_end import EndToEndSystem

SEED = 1234
MAX_EPOCHS = 20
DAFX = "mda Overdrive"
DUMMY_SETTINGS = True

if __name__ == "__main__":
    wandb.require("service")
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision('medium')

    # callbacks
    log_name = f'vtck_{DAFX.split()[-1].lower()}_dummy' if DUMMY_SETTINGS else f'vtck_{DAFX.split()[-1].lower()}'
    wandb_logger = WandbLogger(name=log_name, project='l5proj_end2end')
    # wandb_logger = None

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
    args.dafx_name = DAFX
    args.audio_dir = "/home/kieran/Level5ProjectAudioVAE/src/audio"

    args.audio_encoder_ckpt = \
        "/home/kieran/Level5ProjectAudioVAE/src/l5proj_spectrogram_vae/hdx3y4ly/checkpoints/epoch=169-step=35530.ckpt"

    args.dummy_setting = DUMMY_SETTINGS
    args.max_epochs = MAX_EPOCHS

    # Checkpoint on the first reconstruction loss
    args.train_monitor = f"train_loss/{args.recon_losses[-1]}"
    args.val_monitor = f"val_loss/{args.recon_losses[-1]}"

    dataset_str = args.input_dirs[0]

    train_checkpoint = ModelCheckpoint(
        monitor=args.train_monitor,
        filename="{epoch}-{step}-train-" + f"{dataset_str}",
    )
    val_checkpoint = ModelCheckpoint(
        monitor=args.val_monitor,
        filename="{epoch}-{step}-val-" + f"{dataset_str}",
    )

    # Set up trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        callbacks=[
            LogAudioCallback(),
            train_checkpoint,
            val_checkpoint,
            early_stopping
        ],
        num_sanity_val_steps=0,
        accelerator='cpu',
        gradient_clip_val=3.,
    )

    # create the System
    system = EndToEndSystem(**vars(args))

    # print(torchsummary.summary(system, input_size=(1, args.train_length), batch_size=16, device='cpu'))
    print(system)

    # train!
    trainer.fit(system)
