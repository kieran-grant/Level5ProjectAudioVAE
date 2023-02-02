from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchvision import datasets, transforms

from mnist_test.mnist_vae import MnistVAE

if __name__ == "__main__":
    bs = 32

    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, num_workers=4)

    torch.set_float32_matmul_precision('medium')

    # arg parse for config
    parser = ArgumentParser()

    # Add available trainer args and system args
    parser = MnistVAE.add_model_specific_args(parser)

    # Parse
    args = parser.parse_args()

    args.vae_beta = 0

    model = MnistVAE(**vars(args))

    checkpoint_callback = ModelCheckpoint(monitor="val_loss/loss", mode="min")
    wandb_logger = WandbLogger(name='mnist_test', project='l5proj_spectrogram_vae')

    trainer = pl.Trainer(accelerator="gpu",
                         max_epochs=50,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback])

    trainer.fit(model, train_loader, test_loader)
