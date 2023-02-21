from argparse import ArgumentParser
from itertools import chain
from typing import Tuple

import auraloss
import torch
import torchaudio
import torch.nn as nn
import pytorch_lightning as pl

from pedalboard.pedalboard import load_plugin

from src import utils
from src.spsa.dafx_layer import DAFXLayer
from src.dataset.paired_audio_dataset import PairedAudioDataset
from src.wrappers.dafx_wrapper import DAFXWrapper
from src.wrappers.null_dafx_wrapper import NullDAFXWrapper
from src.models.style_transfer_vae import StyleTransferVAE


class EndToEndSystem(pl.LightningModule):
    # ====== magic methods ========
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self._build_dafx()
        self._build_audio_encoder()
        self._build_controller_network()

        self._configure_losses()

    # ===== private methods =======
    def _build_audio_encoder(self):
        if self.hparams.audio_encoder_ckpt is not None:
            # load encoder weights from a pre-trained system
            system = StyleTransferVAE.load_from_checkpoint(self.hparams.audio_encoder_ckpt)
            self.hparams.controller_input_dim = system.hparams.latent_dim
        else:
            raise NotImplementedError("End-to-end system expects pre-trained audio VAE")

        if self.hparams.audio_encoder_freeze:
            for param in system.parameters():
                param.requires_grad = False

        self.audio_encoder = system

    def _build_controller_network(self):
        layers = [nn.Linear(self.hparams.controller_input_dim,
                            self.hparams.controller_hidden_dims[0]),
                  nn.LeakyReLU()]

        # all other hidden layers
        for i in range(len(self.hparams.controller_hidden_dims) - 1):
            layers.append(nn.Linear(self.hparams.controller_hidden_dims[i], self.hparams.controller_hidden_dims[i + 1]))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(self.hparams.controller_hidden_dims[-1], self.dafx.get_num_params()))

        self.controller = nn.Sequential(*layers)

        self.dafx_layer = DAFXLayer(self.dafx, self.hparams.spsa_epsilon)

    def _configure_losses(self):
        if len(self.hparams.recon_losses) != len(self.hparams.recon_loss_weights):
            raise ValueError("Must supply same number of weights as losses.")

        self.recon_losses = torch.nn.ModuleDict()
        for recon_loss in self.hparams.recon_losses:
            if recon_loss == "mrstft":
                self.recon_losses[recon_loss] = auraloss.freq.MultiResolutionSTFTLoss(
                    fft_sizes=[32, 128, 512, 2048, 8192, 32768],
                    hop_sizes=[16, 64, 256, 1024, 4096, 16384],
                    win_lengths=[32, 128, 512, 2048, 8192, 32768],
                    w_sc=0.0,
                    w_phs=0.0,
                    w_lin_mag=1.0,
                    w_log_mag=1.0,
                )
            elif recon_loss == "mrstft-md":
                self.recon_losses[recon_loss] = auraloss.freq.MultiResolutionSTFTLoss(
                    fft_sizes=[128, 512, 2048, 8192],
                    hop_sizes=[32, 128, 512, 2048],  # 1 / 4
                    win_lengths=[128, 512, 2048, 8192],
                    w_sc=0.0,
                    w_phs=0.0,
                    w_lin_mag=1.0,
                    w_log_mag=1.0,
                )
            elif recon_loss == "mrstft-sm":
                self.recon_losses[recon_loss] = auraloss.freq.MultiResolutionSTFTLoss(
                    fft_sizes=[512, 2048, 8192],
                    hop_sizes=[256, 1024, 4096],  # 1 / 4
                    win_lengths=[512, 2048, 8192],
                    w_sc=0.0,
                    w_phs=0.0,
                    w_lin_mag=1.0,
                    w_log_mag=1.0,
                )
            elif recon_loss == "melfft":
                self.recon_losses[recon_loss] = auraloss.freq.MelSTFTLoss(
                    self.hparams.sample_rate,
                    fft_size=self.hparams.train_length,
                    hop_size=self.hparams.train_length // 2,
                    win_length=self.hparams.train_length,
                    n_mels=128,
                    w_sc=0.0,
                    device="cuda" if self.hparams.gpus > 0 else "cpu",
                )
            elif recon_loss == "melstft":
                self.recon_losses[recon_loss] = auraloss.freq.MelSTFTLoss(
                    self.hparams.sample_rate,
                    device="cuda" if self.hparams.gpus > 0 else "cpu",
                )
            elif recon_loss == "l1":
                self.recon_losses[recon_loss] = torch.nn.L1Loss()
            elif recon_loss == "sisdr":
                self.recon_losses[recon_loss] = auraloss.time.SISDRLoss()
            else:
                raise ValueError(
                    f"Invalid reconstruction loss: {self.hparams.recon_losses}"
                )

    def _build_dafx(self):
        self.dafx = self._get_dafx_from_name(self.hparams.dafx_name,
                                             dafx_file=self.hparams.dafx_file,
                                             sample_rate=self.hparams.sample_rate)

    @staticmethod
    def _get_dafx_from_name(name, dafx_file, sample_rate=24_000):
        if name.lower() == "clean":
            wrapper = NullDAFXWrapper()
        else:
            dafx = load_plugin(dafx_file, plugin_name=name)
            wrapper = DAFXWrapper(dafx, sample_rate=sample_rate)

        return wrapper

    # ======= public methods =========
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor = None,
                analysis_length: int = 0,
                sample_rate: int = 24_000,
                ):

        if sample_rate != self.hparams.sample_rate:
            x_enc = torchaudio.transforms.Resample(
                sample_rate, self.hparams.sample_rate
            ).to(x.device)(x)

            y_enc = torchaudio.transforms.Resample(
                sample_rate, self.hparams.sample_rate
            ).to(x.device)(y)

        else:
            x_enc = x
            y_enc = y

        if analysis_length > 0:
            x_enc = x_enc[..., :analysis_length]
            y_enc = y_enc[..., :analysis_length]

        # Get spectrograms
        x_s = self.audio_encoder.audio_to_spectrogram(signal=x_enc,
                                                      n_fft=self.hparams.n_fft,
                                                      hop_length=self.hparams.hop_length,
                                                      return_phase=self.hparams.return_phase)

        y_s = self.audio_encoder.audio_to_spectrogram(signal=y_enc,
                                                      n_fft=self.hparams.n_fft,
                                                      hop_length=self.hparams.hop_length,
                                                      return_phase=self.hparams.return_phase)

        # Create layered spectrogram
        X = torch.concat([x_s, y_s], dim=1)

        # Get joint embedding from layered spectrogram
        _, _, _, z = self.audio_encoder(X)

        # Map embedding to parameter prediction
        p_logits = self.controller(z)

        # Activation -> (0,1)
        p = torch.sigmoid(p_logits)

        # Process audio conditioned on parameters
        y_hat = self.dafx_layer([x, p]).unsqueeze(1)

        return y_hat, p, z

    def common_paired_step(self,
                           batch: Tuple,
                           batch_idx: int,
                           optimizer_idx: int = 0,
                           train: bool = False
                           ):
        x, y = batch

        x, y_ref, y = utils.get_training_reference(x, y)

        y_hat, p, z = self(x, y=y_ref)

        loss = 0

        # compute reconstruction loss terms
        for loss_idx, (loss_name, recon_loss_fn) in enumerate(
                self.recon_losses.items()
        ):
            temp_loss = recon_loss_fn(y_hat, y)  # reconstruction loss
            loss += float(self.hparams.recon_loss_weights[loss_idx]) * temp_loss

            self.log(
                ("train" if train else "val") + f"_loss/{loss_name}",
                temp_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        # log the overall aggregate loss
        self.log(
            ("train" if train else "val") + "_loss/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        # store audio data
        data_dict = {
            "x": x.cpu(),
            "y": y.cpu(),
            "y_ref": y_ref.cpu(),
            "p": p.cpu(),
            "z": z.cpu(),
            "y_hat": y_hat.cpu(),
        }

        return loss, data_dict

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss, _ = self.common_paired_step(
            batch,
            batch_idx,
            optimizer_idx,
            train=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        loss, data_dict = self.common_paired_step(
            batch,
            batch_idx,
            optimizer_idx,
            train=False,
        )

        return data_dict

    def optimizer_step(
            self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            optimizer_closure,
            on_tpu=False,
            using_lbfgs=False,
    ):
        if optimizer_idx == 0:
            optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        # we need additional optimizer for the discriminator
        optimizers = []
        g_optimizer = torch.optim.Adam(
            chain(
                self.controller.parameters(),
                self.dafx_layer.parameters()
            ),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
        )
        optimizers.append(g_optimizer)

        g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            g_optimizer,
            patience=self.hparams.lr_patience,
            verbose=True,
        )
        ms1 = int(self.hparams.max_epochs * 0.8)
        ms2 = int(self.hparams.max_epochs * 0.95)
        print(
            "Learning rate schedule:",
            f"0 {self.hparams.lr:0.2e} -> ",
            f"{ms1} {self.hparams.lr * 0.1:0.2e} -> ",
            f"{ms2} {self.hparams.lr * 0.01:0.2e}",
        )
        g_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            g_optimizer,
            milestones=[ms1, ms2],
            gamma=0.1,
        )

        lr_schedulers = {
            "scheduler": g_scheduler,
        }

        return optimizers, lr_schedulers

    def train_dataloader(self):
        wrapper = self._get_dafx_from_name(self.hparams.dafx_name,
                                         dafx_file=self.hparams.dafx_file,
                                         sample_rate=self.hparams.sample_rate)

        train_dataset = PairedAudioDataset(
            dafx=wrapper,
            audio_dir=self.hparams.audio_dir,
            subset="train",
            train_frac=self.hparams.train_frac,
            half=self.hparams.half,
            length=self.hparams.train_length * 2,  # Need double length for training
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

        return torch.utils.data.DataLoader(
            train_dataset,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            timeout=6000,
        )

    def val_dataloader(self):
        wrapper = self._get_dafx_from_name(self.hparams.dafx_name,
                                           dafx_file=self.hparams.dafx_file,
                                           sample_rate=self.hparams.sample_rate)

        val_dataset = PairedAudioDataset(
            dafx=wrapper,
            audio_dir=self.hparams.audio_dir,
            subset="val",
            train_frac=self.hparams.train_frac,
            half=self.hparams.half,
            length=self.hparams.val_length * 2, # Need double length for training
            input_dirs=self.hparams.input_dirs,
            buffer_size_gb=self.hparams.buffer_size_gb,
            buffer_reload_rate=self.hparams.buffer_reload_rate,
            num_examples_per_epoch=self.hparams.val_examples_per_epoch,
            effect_input=self.hparams.effect_input,
            effect_output=self.hparams.effect_output,
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
        # --- Training  ---
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--lr_patience", type=int, default=20)
        parser.add_argument("--recon_losses", nargs="+", default=["mrstft", "l1"])
        parser.add_argument("--recon_loss_weights", nargs="+", default=[1.0, 100.0])

        # --------- DAFX ------------
        parser.add_argument("--dafx_file", type=str, default="src/dafx/mda.vst3")
        parser.add_argument("--dafx_name", type=str, default="clean")
        parser.add_argument("--dafx_param_names", nargs="*", default=None)

        # --- Controller  ---
        parser.add_argument("--controller_input_dim", type=int, default=2048)
        parser.add_argument("--controller_hidden_dims", nargs="+", default=[512])

        # --- Encoder ---
        parser.add_argument("--audio_encoder_ckpt", type=str, default=None)
        parser.add_argument("--audio_encoder_freeze", type=bool, default=True)

        # -------- Spectrogram ----------
        parser.add_argument("--n_fft", type=int, default=4096)
        parser.add_argument("--hop_length", type=int, default=2048)
        parser.add_argument("--window_size", type=int, default=4096)
        parser.add_argument("--return_phase", type=bool, default=False)

        # ---  SPSA  ---
        parser.add_argument("--plugin_config_file", type=str, default=None)
        parser.add_argument("--spsa_epsilon", type=float, default=0.005)
        parser.add_argument("--spsa_schedule", action="store_true")

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
        parser.add_argument("--train_examples_per_epoch", type=int, default=5_000)
        parser.add_argument("--val_length", type=int, default=131_072)
        parser.add_argument("--val_examples_per_epoch", type=int, default=500)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--dummy_setting", type=bool, default=False)

        return parser
