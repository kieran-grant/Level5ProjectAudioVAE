from argparse import ArgumentParser
import pytorch_lightning as pl
from src.vqvae_tester.vqvae import VQVAE
import torch


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQVAE.add_model_specific_args(parser)

    args = parser.parse_args(args=[])

    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=4,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        gradient_clip_val=5.
    )

    system = VQVAE(**vars(args))

    print(system)

    trainer.fit(system)
