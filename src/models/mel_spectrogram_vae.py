from argparse import ArgumentParser
from math import prod

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pedalboard.pedalboard import load_plugin

from src.dataset.audio_dataset import AudioDataset
from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.schedulers.beta_annealing import BetaAnnealing
from src.schedulers.cyclic_annealing import CyclicAnnealing
from src.utils import audio_to_mel_spectrogram
from src.wrappers.dafx_wrapper import DAFXWrapper
from src.wrappers.null_dafx_wrapper import NullDAFXWrapper


# noinspection DuplicatedCode
class MelSpectrogramVAE(pl.LightningModule):
    # =========== MAGIC METHODS =============
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.annealing_type == 'cyclic':
            self.beta_annealing = CyclicAnnealing(
                self.hparams.min_beta,
                self.hparams.max_beta,
                self.hparams.beta_start_epoch,
                self.hparams.beta_end_epoch,
                self.hparams.beta_cycle_length
            )
        else:
            self.beta_annealing = BetaAnnealing(
                self.hparams.min_beta,
                self.hparams.max_beta,
                self.hparams.beta_start_epoch,
                self.hparams.beta_end_epoch
            )

        self._build_model()

    # =========== PRIVATE METHODS =============
    def _build_model(self):
        self._build_vae_parameters()
        self._build_dafx()
        self._build_encoder()
        self._build_decoder()

    def _build_vae_parameters(self):
        self.hidden_dim_enc = prod(self.hparams.hidden_dim)
        self.hidden_dim_dec = self.hparams.hidden_dim

    def _build_dafx(self):
        # Load instances for each type of DAFX
        self.dafx_list = self._get_dafx_from_names()
        # Create entry for current dafx name for logging
        self.current_dafx = None

    def _build_encoder(self):
        self.encoder = Encoder(1, self.hparams.num_channels,
                               self.hparams.num_residual_layers,
                               self.hparams.num_residual_channels)

        self.pre_linear_conv = nn.Conv2d(in_channels=self.hparams.num_channels,
                                          out_channels=self.hparams.num_pre_linear_channels,
                                          kernel_size=1,
                                          stride=1)

        self.mu = nn.Linear(self.hidden_dim_enc, self.hparams.latent_dim)
        self.log_var = nn.Linear(self.hidden_dim_enc, self.hparams.latent_dim)

    def _build_decoder(self):
        self.decoder_linear = nn.Sequential(
            nn.Linear(in_features=self.hparams.latent_dim, out_features=self.hidden_dim_enc),
            nn.ReLU())

        self.decoder = Decoder(self.hparams.num_pre_linear_channels,
                               self.hparams.num_channels,
                               self.hparams.num_residual_layers,
                               self.hparams.num_residual_channels)

    @staticmethod
    def _calculate_kl_loss(mean, log_variance):
        return torch.mean(-0.5 * torch.sum(1 + log_variance - mean ** 2 - log_variance.exp(), dim=1), dim=0)

    @staticmethod
    def _calculate_reconstruction_loss(x, x_hat):
        return F.mse_loss(x, x_hat, reduction="mean")

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
        x = self.pre_linear_conv(x)

        x = x.reshape(-1, self.hidden_dim_enc)

        mu = self.mu(x)
        log_var = self.log_var(x)

        return mu, log_var

    def decode(self, z):
        x = self.decoder_linear(z)

        x = x.view(-1, *self.hidden_dim_dec)

        x = self.decoder(x)

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
            batch: torch.Tensor,
            batch_idx: int,
            train: bool = False,
    ):
        # Get audio
        x = batch

        # Get spectrograms
        X = audio_to_mel_spectrogram(signal=x,
                                     sample_rate=self.hparams.sample_rate,
                                     n_mels=self.hparams.n_mels,
                                     n_fft=self.hparams.n_fft,
                                     win_length=self.hparams.win_length,
                                     f_max=self.hparams.f_max,
                                     f_min=self.hparams.f_min)

        # Get reconstruction as well as mu, var
        X_hat, X_mu, X_log_var, _ = self(X)

        # Calculate recon losses for clean/effected signals
        r_loss, kl_loss = self.calculate_loss(X_mu, X_log_var, X_hat, X)

        # Total loss is additive
        loss = r_loss + (self.beta_annealing.beta * kl_loss)

        # log the losses
        self.log(("train" if train else "val") + "_loss/loss", loss)
        self.log(("train" if train else "val") + "_loss/reconstruction_loss", r_loss)
        self.log(("train" if train else "val") + "_loss/kl_divergence", kl_loss)

        # log current beta value
        self.log("beta", self.beta_annealing.beta)

        data_dict = {
            "x": X.cpu(),
            "x_hat": X_hat.cpu(),
            "dafx": self.current_dafx.split()[-1]
        }

        return loss, data_dict

    def training_step(self, batch, batch_idx):
        loss, _ = self.common_paired_step(
            batch,
            batch_idx,
            train=True,
        )

        return loss

    def training_epoch_end(self, training_step_outputs):
        if self.hparams.beta_annealing:
            self.beta_annealing.step(self.current_epoch)

    def validation_step(self, batch, batch_idx):
        loss, data_dict = self.common_paired_step(
            batch,
            batch_idx,
            train=False,
        )

        return data_dict

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
            length=self.hparams.val_length,
            input_dirs=self.hparams.input_dirs,
            buffer_size_gb=self.hparams.buffer_size_gb,
            buffer_reload_rate=self.hparams.buffer_reload_rate,
            num_examples_per_epoch=self.hparams.val_examples_per_epoch,
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
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--lr", type=float, default=5e-4)

        # -------- Beta Annealing ---------
        parser.add_argument("--beta_annealing", type=bool, default=True)
        parser.add_argument("--annealing_type", type=str, default='cyclic')
        parser.add_argument("--min_beta", type=float, default=0.)
        parser.add_argument("--max_beta", type=float, default=4.)
        parser.add_argument("--beta_start_epoch", type=int, default=0)
        parser.add_argument("--beta_end_epoch", type=int, default=100)
        parser.add_argument("--beta_cycle_length", type=int, default=11)

        # --------- DAFX ------------
        parser.add_argument("--dafx_file", type=str, default="src/dafx/mda.vst3")
        parser.add_argument("--dafx_names", nargs="*")
        parser.add_argument("--dafx_param_names", nargs="*", default=None)

        # -------- Spectrogram ----------
        parser.add_argument("--n_mels", type=int, default=256)
        parser.add_argument("--n_fft", type=int, default=4096)
        parser.add_argument("--win_length", type=int, default=1024)
        parser.add_argument("--f_max", type=int, default=12_000)
        parser.add_argument("--f_min", type=int, default=20)

        # --------- VAE -------------
        parser.add_argument("--hidden_dim", nargs="*", default=(4, 28, 28))
        parser.add_argument("--latent_dim", type=int, default=256)
        parser.add_argument("--num_channels", type=int, default=16)
        parser.add_argument("--num_residual_channels", type=int, default=8)
        parser.add_argument("--num_residual_layers", type=int, default=2)
        parser.add_argument("--num_pre_linear_channels", type=int, default=4)

        # ------- Dataset  -----------
        parser.add_argument("--audio_dir", type=str, default="src/audio")
        parser.add_argument("--ext", type=str, default="wav")
        parser.add_argument("--input_dirs", nargs="+", default=['musdb18_24000', 'vctk_24000'])
        parser.add_argument("--buffer_reload_rate", type=int, default=1000)
        parser.add_argument("--buffer_size_gb", type=float, default=1.0)
        parser.add_argument("--sample_rate", type=int, default=24_000)
        parser.add_argument("--dsp_sample_rate", type=int, default=24_000)
        parser.add_argument("--shuffle", type=bool, default=True)
        parser.add_argument("--random_effect_threshold", type=float, default=0.)
        parser.add_argument("--train_length", type=int, default=130_560)
        parser.add_argument("--train_frac", type=float, default=0.9)
        parser.add_argument("--effect_audio", type=bool, default=True)
        parser.add_argument("--half", type=bool, default=False)
        parser.add_argument("--train_examples_per_epoch", type=int, default=5_000)
        parser.add_argument("--val_length", type=int, default=130_560)
        parser.add_argument("--val_examples_per_epoch", type=int, default=500)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--dummy_setting", type=bool, default=False)

        return parser
