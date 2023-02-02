from argparse import ArgumentParser
from math import prod
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistVAE(pl.LightningModule):
    # =========== MAGIC METHODS =============
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Load instances for each type of DAFX
        self.hidden_dim_enc = prod(self.hparams.hidden_dim)
        self.hidden_dim_dec = self.hparams.hidden_dim

        self._build_model()

    # =========== PRIVATE METHODS =============
    def _build_model(self):
        self._build_encoder()
        self._build_decoder()

    def _build_encoder(self):
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.hparams.num_channels,
                      out_channels=8,
                      kernel_size=self.hparams.conv_kernel,
                      padding=self.hparams.conv_padding,
                      stride=self.hparams.conv_stride
                      ),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )

        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=self.hparams.conv_kernel,
                      padding=self.hparams.conv_padding,
                      stride=self.hparams.conv_stride
                      ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=self.hparams.conv_kernel,
                      padding=self.hparams.conv_padding,
                      stride=self.hparams.conv_stride
                      ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        self.enc_linear = nn.Sequential(
            nn.Linear(self.hidden_dim_enc, self.hparams.linear_layer_dim),
            nn.ReLU()
        )

        self.mu = nn.Linear(self.hparams.linear_layer_dim, self.hparams.latent_dim)
        self.log_var = nn.Linear(self.hparams.linear_layer_dim, self.hparams.latent_dim)

    def _build_decoder(self):
        self.dec_hidden1 = self.dec_hidden = nn.Sequential(
            nn.Linear(in_features=self.hparams.latent_dim, out_features=self.hparams.linear_layer_dim),
            nn.ReLU())


        self.dec_hidden2 = nn.Sequential(
            nn.Linear(in_features=self.hparams.linear_layer_dim, out_features=self.hidden_dim_enc),
            nn.ReLU())

        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=16,
                               kernel_size=self.hparams.conv_kernel,
                               padding=self.hparams.conv_padding,
                               stride=self.hparams.conv_stride
                               ),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16,
                               out_channels=8,
                               kernel_size=self.hparams.conv_kernel,
                               padding=self.hparams.conv_padding,
                               stride=self.hparams.conv_stride
                               ),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )

        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8,
                               out_channels=self.hparams.num_channels,
                               kernel_size=self.hparams.conv_kernel,
                               padding=self.hparams.conv_padding,
                               stride=self.hparams.conv_stride
                               ),
        )

    @staticmethod
    def _compute_kl_loss(mean, log_variance):
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
        kl_loss = self._compute_kl_loss(mean, log_variance)
        return r_loss, kl_loss

    def encode(self, x):
        x = self.enc_conv1(x)
        x = self.enc_conv2(x)
        x = self.enc_conv3(x)

        x = x.view(-1, self.hidden_dim_enc)

        x = self.enc_linear(x)
        mu = self.mu(x)
        log_var = self.log_var(x)

        return mu, log_var

    def decode(self, z):
        x = self.dec_hidden1(z)
        x = self.dec_hidden2(x)

        x = x.view(-1, *self.hidden_dim_dec)

        x = self.dec_conv1(x)
        x = self.dec_conv2(x)
        x = self.dec_conv3(x)

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

        return out, mu, log_var

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def common_paired_step(
            self,
            batch: Tuple,
            batch_idx: int,
            train: bool = False,
    ):
        # Get spectrograms
        x, _ = batch

        # Get reconstruction as well as mu, var
        x_hat, x_mu, x_log_var = self(x)

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # -------- Training -----------
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--recon_loss", type=str, default="mse")
        parser.add_argument("--vae_beta", type=float, default=1.)

        # --------- VAE -------------
        parser.add_argument("--num_channels", type=int, default=1)
        parser.add_argument("--hidden_dim", nargs="*", default=(32, 28, 28))
        parser.add_argument("--linear_layer_dim", type=int, default=1024)
        parser.add_argument("--latent_dim", type=int, default=512)
        parser.add_argument("--conv_kernel", type=int, default=3)
        parser.add_argument("--conv_padding", type=int, default=1)
        parser.add_argument("--conv_stride", type=int, default=1)

        return parser
