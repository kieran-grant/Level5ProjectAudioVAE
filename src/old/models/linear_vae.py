from argparse import ArgumentParser
from typing import Tuple
from pedalboard.pedalboard import load_plugin

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.dataset.audio_dataset import AudioDataset
from src.wrappers.dafx_wrapper import DAFXWrapper
from src.wrappers.null_dafx_wrapper import NullDAFXWrapper


class LinearVAE(pl.LightningModule):
    # =========== MAGIC METHODS =============
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Load instances for each type of DAFX
        self.dafx_list = self._get_dafx_from_names()
        # Create entry for current dafx name for logging
        self.current_dafx = None

        self.input_shape = self.hparams.input_shape
        self.hidden_layer_dims = self.hparams.hidden_layer_dims
        self.latent_space_dim = self.hparams.latent_space_dim

        self._num_hidden_layers = len(self.hparams.hidden_layer_dims)

        self._build()

    # =========== PRIVATE METHODS =============
    def _build(self):
        self._add_encoder_linear_layers()
        self._add_distribution_params()
        self._add_decoder_linear_layers()

    def _add_distribution_params(self):
        self.mu = nn.Linear(self.hparams.hidden_layer_dims[-1], self.hparams.latent_space_dim)
        self.sigma = nn.Linear(self.hparams.hidden_layer_dims[-1], self.hparams.latent_space_dim)

    def _add_encoder_linear_layers(self):
        encoder_dims = self.hparams.hidden_layer_dims
        layers = [nn.Linear(self.hparams.input_shape, encoder_dims[0]), nn.LeakyReLU()]

        # all other hidden layers
        for i in range(self._num_hidden_layers - 1):
            layers.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))
            layers.append(nn.LeakyReLU())

        self.encoder = nn.Sequential(*layers)

    def _add_decoder_linear_layers(self):
        decoder_dims = self.hparams.hidden_layer_dims[::-1]
        layers = [nn.Linear(self.hparams.latent_space_dim, decoder_dims[0]), nn.LeakyReLU()]

        # all other hidden layers
        for i in range(self._num_hidden_layers - 1):
            layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(decoder_dims[-1], self.hparams.input_shape))

        self.decoder = nn.Sequential(*layers)

    @staticmethod
    def _calculate_kl_loss(mean, log_variance):
        # calculate KL divergence
        kld_batch = -0.5 * torch.sum(1 + log_variance - torch.square(mean) - torch.exp(log_variance), dim=1)
        kld = torch.mean(kld_batch)

        return kld

    def _calculate_reconstruction_loss(self, x, x_hat):
        if self.hparams.recon_loss.lower() == "mse":
            return F.mse_loss(x, x_hat, reduction="mean")
        elif self.hparams.recon_loss.lower() == "l1":
            return F.l1_loss(x, x_hat, reduction="mean")
        elif self.hparams.recon_loss.lower() == "bce":
            return F.binary_cross_entropy(x, x_hat, reduction="mean")
        else:
            raise NotImplementedError

    def calculate_loss(self, mean, log_variance, predictions, targets):
        r_loss = self._calculate_reconstruction_loss(targets, predictions)
        kl_loss = self._calculate_kl_loss(mean, log_variance)
        return r_loss, kl_loss

    def _get_dafx_from_names(self):
        dafx_instances = []

        for dafx_name in self.hparams.dafx_names:
            if dafx_name.lower() == "clean":
                dafx_instances.append(NullDAFXWrapper())
            else:
                dafx = load_plugin(self.hparams.dafx_file, plugin_name=dafx_name)
                dafx_instances.append(DAFXWrapper(dafx, sample_rate=self.hparams.sample_rate))

        return dafx_instances

    def _get_dafx_for_current_epoch(self, current_epoch: int):
        # Use mod arithmetic to cycle through dafx
        idx = current_epoch % len(self.dafx_list)

        self.current_dafx = self.hparams.dafx_names[idx]

        print(f"\nEpoch {current_epoch} using DAFX: {self.current_dafx}")

        return self.dafx_list[idx]

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.sigma(x)

        return mu, log_var

    def decode(self, z):
        x = self.decoder(z)

        return x

    @staticmethod
    def reparameterise(mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        out = self.decode(z)

        return out, mu, log_var, z

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def common_paired_step(
            self,
            batch: Tuple,
            batch_idx: int,
            train: bool = False,
    ):
        # Get spectrograms
        x = batch
        x = torch.flatten(x, start_dim=1)

        # Get reconstruction as well as mu, var
        x_hat, x_mu, x_log_var, _ = self(x)

        # Calculate recon losses for clean/effected signals
        r_loss, kl_loss = self.calculate_loss(x_mu, x_log_var, x_hat, x)

        # Total loss is additive
        loss = r_loss + (self.hparams.vae_beta * kl_loss)

        # log the losses
        self.log(("train" if train else "val") + "_loss/loss", loss)
        self.log(("train" if train else "val") + "_loss/reconstruction_loss", r_loss)
        self.log(("train" if train else "val") + "_loss/kl_divergence", kl_loss)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_paired_step(
            batch,
            batch_idx,
            train=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_paired_step(
            batch,
            batch_idx,
            train=False,
        )

        return loss

    def train_dataloader(self):
        # Return dataloader based on epoch??
        dafx = self._get_dafx_for_current_epoch(self.current_epoch)

        train_dataset = AudioDataset(
            dafx=dafx,
            audio_dir=self.hparams.audio_dir,
            subset="train",
            train_frac=self.hparams.train_frac,
            half=self.hparams.half,
            length=self.hparams.train_length,
            input_dirs=self.hparams.input_dirs,
            buffer_size_gb=self.hparams.buffer_size_gb,
            buffer_reload_rate=self.hparams.buffer_reload_rate,
            num_examples_per_epoch=self.hparams.train_examples_per_epoch,
            effect_audio=self.hparams.effect_audio,
            random_effect_threshold=self.hparams.random_effect_threshold,
            augmentations={
                "pitch": {"sr": self.hparams.sample_rate},
                "tempo": {"sr": self.hparams.sample_rate},
            },
            ext=self.hparams.ext,
            dummy_setting=self.hparams.dummy_setting
        )

        return torch.utils.data.DataLoader(
            train_dataset,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            timeout=60,
        )

    def val_dataloader(self):
        dafx = self._get_dafx_for_current_epoch(self.current_epoch)

        val_dataset = AudioDataset(
            dafx=dafx,
            audio_dir=self.hparams.audio_dir,
            subset="val",
            train_frac=self.hparams.train_frac,
            half=self.hparams.half,
            length=self.hparams.train_length,
            input_dirs=self.hparams.input_dirs,
            buffer_size_gb=self.hparams.buffer_size_gb,
            buffer_reload_rate=self.hparams.buffer_reload_rate,
            num_examples_per_epoch=self.hparams.train_examples_per_epoch,
            effect_audio=self.hparams.effect_audio,
            random_effect_threshold=self.hparams.random_effect_threshold,
            augmentations={},
            ext=self.hparams.ext,
            dummy_setting=self.hparams.dummy_setting
        )

        return torch.utils.data.DataLoader(
            val_dataset,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            timeout=60,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # -------- Training -----------
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--recon_loss", type=str, default="mse")
        parser.add_argument("--vae_beta", type=float, default=1.)

        # --------- DAFX ------------
        parser.add_argument("--dafx_file", type=str, default="src/dafx/mda.vst3")
        parser.add_argument("--dafx_names", nargs="*")
        parser.add_argument("--dafx_param_names", nargs="*", default=None)

        # --------- VAE -------------
        parser.add_argument("--hidden_layer_dims", nargs="+", default=[1024, 1024, 512, 512])
        parser.add_argument("--latent_space_dim", type=int, default=512)
        parser.add_argument("--input_shape", type=int, default=133_185)

        # ------- Dataset  -----------
        parser.add_argument("--audio_dir", type=str, default="src/audio")
        parser.add_argument("--ext", type=str, default="wav")
        parser.add_argument("--input_dirs", nargs="+", default=['musdb18_24000', 'vctk_24000'])
        parser.add_argument("--buffer_reload_rate", type=int, default=1000)
        parser.add_argument("--buffer_size_gb", type=float, default=1.0)
        parser.add_argument("--sample_rate", type=int, default=24_000)
        parser.add_argument("--dsp_sample_rate", type=int, default=24_000)
        parser.add_argument("--shuffle", type=bool, default=True)
        parser.add_argument("--random_effect_threshold", type=float, default=0.75)
        parser.add_argument("--train_length", type=int, default=131_072)
        parser.add_argument("--train_frac", type=float, default=0.9)
        parser.add_argument("--effect_audio", type=bool, default=True)
        parser.add_argument("--half", type=bool, default=False)
        parser.add_argument("--train_examples_per_epoch", type=int, default=10_000)
        parser.add_argument("--val_length", type=int, default=131_072)
        parser.add_argument("--val_examples_per_epoch", type=int, default=100)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--dummy_setting", type=bool, default=False)

        return parser
