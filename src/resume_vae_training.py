from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.models.style_transfer_vae import StyleTransferVAE

# DAFX_TO_USE = [
#     # 'mda MultiBand',
#     # 'clean',
#     'mda Overdrive',
#     # # 'mda Ambience',
#     'mda Delay',
#     # 'mda Leslie',
#     # 'mda Combo',
#     # 'mda Thru-Zero Flanger',
#     # 'mda Loudness',
#     # 'mda Limiter'
# ]

SEED = 1234
PATH_TO_CHECKPOINT = "/home/kieran/Level5ProjectAudioVAE/src/l5proj_style_vae/ed5w27io/checkpoints/epoch=788-step=493125.ckpt"

if __name__ == "__main__":
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision('medium')

    # callbacks
    wandb_logger = WandbLogger(name='vctk_2dafx_no_clean', project='l5proj_style_vae')
    # wandb_logger = None

    checkpoint_callback = ModelCheckpoint(monitor="val_loss/loss", mode="min")
    # early_stopping = EarlyStopping(
    #     monitor="val_loss/loss",
    #     mode="min",
    #     patience=100)

    # arg parse for config
    parser = ArgumentParser()

    # Add available trainer args and system args
    parser = pl.Trainer.add_argparse_args(parser)
    # parser = StyleTransferVAE.add_model_specific_args(parser)

    # Parse
    args = parser.parse_args()

    system = StyleTransferVAE.load_from_checkpoint(PATH_TO_CHECKPOINT)

    system.hparams.vae_beta = 5e-3
    system.hparams.lr = 1e-4

    # Change settings for training
    # args.input_dirs = ['vctk_24000']

    # args.dafx_file = "/home/kieran/Level5ProjectAudioVAE/src/dafx/mda.vst3"
    # args.dafx_names = DAFX_TO_USE
    # args.audio_dir = "/home/kieran/Level5ProjectAudioVAE/src/audio"
    #
    # args.effect_input = False
    # args.effect_output = True
    # args.dummy_setting = True
    # args.return_phase = False
    #
    # args.num_channels = 2
    # args.latent_dim = 4096
    #
    # args.vae_beta = 1e-3
    # args.lr = 5e-5

    # Set up trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        reload_dataloaders_every_n_epochs=1,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            # early_stopping
        ],
        num_sanity_val_steps=0,
        max_epochs=850,
        accelerator='gpu',
    )

    # create the System
    print(system)

    # train!
    trainer.fit(system, ckpt_path=PATH_TO_CHECKPOINT)