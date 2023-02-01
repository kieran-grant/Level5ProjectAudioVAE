from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.models.vae import SpectrogramVAE

DAFX_TO_USE = [
    'mda MultiBand',
    'mda Overdrive',
    'mda Ambience',
    'mda Delay',
    # 'mda Leslie',
    # 'mda Combo',
    'mda Thru-Zero Flanger',
    # 'mda Loudness',
    'mda Limiter'
]

if __name__ == "main":
    pl.seed_everything(0)

    # callbacks
    wandb_logger = WandbLogger(name='spectrogram_vae_training', project='l5proj_spectrogram_vae')
    checkpoint_callback = ModelCheckpoint(monitor="val_loss/loss", mode="min")
    early_stopping = EarlyStopping(
        monitor="val_loss/loss",
        mode="min",
        # should cycle through all effects at least twice before early stopping
        patience=len(DAFX_TO_USE) * 2)

    # arg parse for config
    parser = ArgumentParser()

    # Add available trainer args and system args
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SpectrogramVAE.add_model_specific_args(parser)

    # Parse
    args = parser.parse_args()

    # Change dummy test vars for debugging
    args.accelerator = 'gpu'
    args.device = 'cuda'

    args.max_epochs = 200
    args.train_examples_per_epoch = 250
    args.val_examples_per_epoch = 20
    args.batch_size = 16
    args.num_workers = 2
    args.vae_beta = 1
    args.lr = 1e-4

    args.dafx_names = DAFX_TO_USE
    args.log_every_n_steps = 1

    args.effect_input = False
    args.dummy_setting = True

    trainer = pl.Trainer.from_argparse_args(
        args,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=1,
        val_check_interval=1.,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        num_sanity_val_steps=0
    )

    # create the System
    system = SpectrogramVAE(**vars(args))

    print(system)

    # train!
    trainer.fit(system)