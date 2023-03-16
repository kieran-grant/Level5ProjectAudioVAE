from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.callbacks.audio import LogAudioCallback

from pytorch_lightning.loggers import WandbLogger

from src.models.end_to_end import EndToEndSystem

SEED = 1234

if __name__ == "__main__":
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision('medium')

    # callbacks
    wandb_logger = WandbLogger(name='vtck_delay_random_params_out_only', project='l5proj_end2end')
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

    args.dafx_file = "/src/dafx/mda.vst3"
    args.dafx_name = "mda Delay"
    args.audio_dir = "/src/audio"

    args.audio_encoder_ckpt = \
        "/home/kieran/Level5ProjectAudioVAE/src/l5proj_style_vae/3kdv9ddi/checkpoints/epoch=820-step=513125.ckpt"

    args.effect_input = False
    args.effect_output = True
    args.dummy_setting = True

    args.train_examples_per_epoch = 5_000
    args.val_examples_per_epoch = 500
    args.max_epochs = 30

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
    )

    # create the System
    system = EndToEndSystem(**vars(args))

    print(system)

    # train!
    trainer.fit(system)
