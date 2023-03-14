from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from src.models.vqvae.decoder import Decoder
from src.models.vqvae.encoder import Encoder
from src.models.vqvae.quantizer import VectorQuantizerEMA


class VQVAE(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self._build_model()

    def _build_model(self):
        self._encoder = Encoder(3, self.hparams.num_hiddens,
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity

    def common_step(self, batch, batch_idx, train=False):
        data, _ = batch
        vq_loss, data_recon, perplexity = self(data)
        recon_error = F.mse_loss(data_recon, data)

        loss = recon_error + vq_loss

        self.log(("train" if train else "val") + "_loss/loss", loss)
        self.log(("train" if train else "val") + "_loss/recon_loss", recon_error)
        self.log(("train" if train else "val") + "_loss/vq_loss", vq_loss)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, False)

        return loss

    def train_dataloader(self):
        training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                         ]))

        return torch.utils.data.DataLoader(training_data,
                                           num_workers=self.hparams.num_workers,
                                           batch_size=self.hparams.batch_size,
                                           shuffle=True,
                                           pin_memory=True)

    def val_dataloader(self):
        validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                           ]))

        return torch.utils.data.DataLoader(validation_data,
                                           num_workers=self.hparams.num_workers,
                                           batch_size=32,
                                           shuffle=False,
                                           pin_memory=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # -------- Training -----------
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--lr", type=float, default=1e-3)

        # --------- VAE -------------
        parser.add_argument("--num_hiddens", type=int, default=128)
        parser.add_argument("--num_residual_hiddens", type=int, default=32)
        parser.add_argument("--num_residual_layers", type=int, default=2)
        parser.add_argument("--embedding_dim", type=int, default=64)
        parser.add_argument("--num_embeddings", type=int, default=512)
        parser.add_argument("--commitment_cost", type=float, default=.25)
        parser.add_argument("--decay", type=float, default=.99)

        # ------- Dataset  -----------
        parser.add_argument("--num_workers", type=int, default=4)

        return parser
