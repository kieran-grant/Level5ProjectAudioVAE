from math import prod
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class SpectrogramVAE(pl.LightningModule):
    def __init__(self,
                 num_channels: int = 1,
                 hidden_dim: Tuple = (32, 32, 20),
                 latent_dim: int = 256,
                 learning_rate: float = 1e-4
                 ):
        super().__init__()

        self.num_channels = num_channels
        self.hidden_dim_enc = prod(hidden_dim)
        self.hidden_dim_dec = hidden_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

        self._build_model()

    def _build_model(self):
        self._build_encoder()
        self._build_decoder()

    def _build_encoder(self):
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(self.num_channels, 8, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )

        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        self.mu = nn.Linear(self.hidden_dim_enc, self.latent_dim)
        self.log_var = nn.Linear(self.hidden_dim_enc, self.latent_dim)

    def _build_decoder(self):
        self.dec_hidden = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=self.hidden_dim_enc),
            nn.ReLU())

        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )

        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose2d(8, self.num_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    @staticmethod
    def _calculate_loss(y, y_hat, mu, log_var):
        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = F.binary_cross_entropy(y, y_hat, reduction='sum') + kl_divergence

        return loss

    def encode(self, x):
        x = self.enc_conv1(x)
        x = self.enc_conv2(x)
        x = self.enc_conv3(x)

        x = x.view(-1, self.hidden_dim_enc)

        mu = self.mu(x)
        log_var = self.log_var(x)

        return mu, log_var

    def decode(self, z):
        x = self.dec_hidden(z)

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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, train_batch, batch_idx):
        imgs = train_batch

        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        out, mu, log_var = self(imgs)

        loss = self._calculate_loss(out, imgs, mu, log_var)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, val_idx):
        imgs = val_batch

        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        out, mu, log_var = self(imgs)

        loss = self._calculate_loss(out, imgs, mu, log_var)

        self.log('val_loss', loss)

        return loss
