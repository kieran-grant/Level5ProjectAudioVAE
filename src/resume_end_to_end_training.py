from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.callbacks.audio import LogAudioCallback
from src.models.end_to_end import EndToEndSystem

SEED = 1234
PATH_TO_CHECKPOINT = "/home/kieran/Level5ProjectAudioVAE/src/l5proj_end2end/zlevhvbp/checkpoints/epoch=27-step=17500-val-vctk_24000.ckpt"

if __name__ == "__main__":
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision('medium')

    # callbacks
    wandb_logger = WandbLogger(name='vctk_overdrive_out_effect_only', project='l5proj_end2end')
    # wandb_logger = None

    # early_stopping = EarlyStopping(
    #     monitor="val_loss/loss",
    #     mode="min",
    #     patience=40)

    # arg parse for config
    parser = ArgumentParser()

    # Add available trainer args and system args
    parser = pl.Trainer.add_argparse_args(parser)

    # Parse
    args = parser.parse_args()

    system = EndToEndSystem.load_from_checkpoint(PATH_TO_CHECKPOINT)

    system.hparams.lr = 5e-5

    # Checkpoint on the first reconstruction loss
    args.train_monitor = f"train_loss/{system.hparams.recon_losses[-1]}"
    args.val_monitor = f"val_loss/{system.hparams.recon_losses[-1]}"

    dataset_str = system.hparams.input_dirs[0]

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
            # early_stopping
        ],
        num_sanity_val_steps=0,
        max_epochs=60,
        accelerator='cpu',
    )

    print(system)

    # train!
    trainer.fit(system, ckpt_path=PATH_TO_CHECKPOINT)
