from argparse import ArgumentParser
from math import prod
from typing import Tuple
from pedalboard.pedalboard import load_plugin

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.dataset.audio_dataset import AudioDataset
from src.wrappers.dafx_wrapper import DAFXWrapper


class SpectrogramVAE(pl.LightningModule):
    # =========== MAGIC METHODS =============
    def __init__(self,
                 num_channels: int = 1,
                 hidden_dim: Tuple = (32, 32, 20),
                 latent_dim: int = 256,
                 learning_rate: float = 1e-4
                 ):
        super().__init__()

        # Load instances for each type of DAFX
        self.dafx_list = self._get_dafx_from_names()
        # Create entry for current dafx name for logging
        self.current_dafx = None

        self.num_channels = num_channels
        self.hidden_dim_enc = prod(hidden_dim)
        self.hidden_dim_dec = hidden_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

        self._build_model()

    # =========== PRIVATE METHODS =============
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

    def _get_dafx_from_names(self):
        dafx_instances = []

        for dafx_name in self.hparams.dafx_names:
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
            effect_input=self.hparams.effect_input,
            effect_output=self.hparams.effect_output,
            random_effect_threshold=self.hparams.random_effect_threshold,
            augmentations={
                "pitch": {"sr": self.hparams.sample_rate},
                "tempo": {"sr": self.hparams.sample_rate},
            },
            ext=self.hparams.ext,
            dummy_setting=self.hparams.dummy_setting
        )

        g = torch.Generator()
        g.manual_seed(0)

        return torch.utils.data.DataLoader(
            train_dataset,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            generator=g,
            # pin_memory=True,
            # persistent_workers=True,
            timeout=6000,
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
            effect_input=self.hparams.effect_input,
            effect_output=self.hparams.effect_output,
            random_effect_threshold=self.hparams.random_effect_threshold,
            augmentations={},
            ext=self.hparams.ext,

        )

        g = torch.Generator()
        g.manual_seed(0)

        return torch.utils.data.DataLoader(
            val_dataset,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            # worker_init_fn=utils.seed_worker,
            generator=g,
            # pin_memory=True,
            # persistent_workers=True,
            timeout=60,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # -------- Training -----------
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--lr", type=float, default=1e-4)

        # --------- DAFX ------------
        parser.add_argument("--dafx_file", type=str, default="src/dafx/mda.vst3")
        parser.add_argument("--dafx_names", nargs="*")
        parser.add_argument("--dafx_param_names", nargs="*", default=None)

        # --------- VAE -------------
        parser.add_argument("--vae_beta", type=float, default=100.)
        parser.add_argument("--num_channels", type=int, default=1)
        parser.add_argument("--hidden_dim", nargs="*", default=(32, 9, 257))
        parser.add_argument("--latent_dim", type=int, default=256)

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
        parser.add_argument("--effect_input", type=bool, default=True)
        parser.add_argument("--effect_output", type=bool, default=True)
        parser.add_argument("--half", type=bool, default=False)
        parser.add_argument("--train_examples_per_epoch", type=int, default=10_000)
        parser.add_argument("--val_length", type=int, default=131_072)
        parser.add_argument("--val_examples_per_epoch", type=int, default=100)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--dummy_setting", type=bool, default=False)

        return parser