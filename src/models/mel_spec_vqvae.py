from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pedalboard.pedalboard import load_plugin

from src.dataset.audio_dataset import AudioDataset
from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.quantizer import VectorQuantizerEMA
from src.utils import audio_to_mel_spectrogram
from src.wrappers.dafx_wrapper import DAFXWrapper
from src.wrappers.null_dafx_wrapper import NullDAFXWrapper


class MelSpecVQVAE(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self._build_model()
        self._build_dafx()

    def _build_model(self):
        self._encoder = Encoder(1, self.hparams.num_hiddens,
                                self.hparams.num_residual_layers,
                                self.hparams.num_residual_hiddens)

        self._pre_vq_conv = nn.Conv2d(in_channels=self.hparams.num_hiddens,
                                      out_channels=self.hparams.embedding_dim,
                                      kernel_size=1,
                                      stride=1)

        self._vq_vae = VectorQuantizerEMA(self.hparams.num_embeddings,
                                          self.hparams.embedding_dim,
                                          self.hparams.commitment_cost,
                                          self.hparams.decay)

        self._decoder = Decoder(self.hparams.embedding_dim,
                                self.hparams.num_hiddens,
                                self.hparams.num_residual_layers,
                                self.hparams.num_residual_hiddens)

    def _build_dafx(self):
        # Load instances for each type of DAFX
        self.dafx_list = self._get_dafx_from_names()
        # Create entry for current dafx name for logging
        self.current_dafx = None

    def _get_dafx_for_current_epoch(self, current_epoch: int):
        # Use mod arithmetic to cycle through dafx
        idx = current_epoch % len(self.dafx_list)

        self.current_dafx = self.hparams.dafx_names[idx]

        print(f"\nEpoch {current_epoch} using DAFX: {self.current_dafx}")

        return self.dafx_list[idx]

    def _get_dafx_from_names(self):
        dafx_instances = []

        for dafx_name in self.hparams.dafx_names:
            if dafx_name.lower() == "clean":
                dafx_instances.append(NullDAFXWrapper())
            else:
                dafx = load_plugin(self.hparams.dafx_file, plugin_name=dafx_name)
                dafx_instances.append(DAFXWrapper(dafx, sample_rate=self.hparams.sample_rate))

        return dafx_instances

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity

    def common_step(
            self,
            batch: torch.Tensor,
            batch_idx: int,
            train: bool = False,
    ):
        x = batch

        # Get spectrograms
        X = audio_to_mel_spectrogram(signal=x,
                                     sample_rate=self.hparams.sample_rate,
                                     n_mels=self.hparams.n_mels,
                                     n_fft=self.hparams.n_fft,
                                     win_length=self.hparams.win_length,
                                     f_max=self.hparams.f_max,
                                     f_min=self.hparams.f_min)

        vq_loss, X_hat, perplexity = self(X)
        recon_loss = F.mse_loss(X_hat, X)

        loss = recon_loss + vq_loss

        self.log(("train" if train else "val") + "_loss/loss", loss)
        self.log(("train" if train else "val") + "_loss/recon_loss", recon_loss)
        self.log(("train" if train else "val") + "_loss/vq_loss", vq_loss)
        self.log(("train" if train else "val") + "_loss/perplexity", perplexity)

        data_dict = {
            "x": X.cpu(),
            "x_hat": X_hat.cpu(),
            "dafx": self.current_dafx.split()[-1]
        }

        return loss, data_dict

    def training_step(self, batch, batch_idx):
        loss, _ = self.common_step(
            batch,
            batch_idx,
            train=True)

        return loss

    def validation_step(self, batch, batch_idx):
        _, data_dict = self.common_step(
            batch,
            batch_idx,
            train=False)

        return data_dict

    def train_dataloader(self):
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
        parser.add_argument("--num_hiddens", type=int, default=128)
        parser.add_argument("--num_residual_hiddens", type=int, default=32)
        parser.add_argument("--num_residual_layers", type=int, default=2)
        parser.add_argument("--embedding_dim", type=int, default=128)
        parser.add_argument("--num_embeddings", type=int, default=4096)
        parser.add_argument("--commitment_cost", type=float, default=.25)
        parser.add_argument("--decay", type=float, default=.99)

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
