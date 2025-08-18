from typing import *
import os
import csv
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as sched
import torch.optim as optim
from torch.utils.data import DataLoader
from data_factory.loader import Dataset
from models.tcnvae import VAE
from utils.loss import loss
"""
Codes for training of TCN-VAE. This code follows the paper
Darban et al., 2025, GenIAS: Generator for Instantiating Anomalies in Time Series.

Paper link: https://arxiv.org/pdf/2502.08262

The default values of arguments follow the paper.
"""


def str2bool(v: str) -> bool:
    """
    Converts string to bool.

    Parameters:
        v: String instance. Either 'True' or 'False.'

    Returns:
        Boolean instance. Either True or False.
    """
    return v.lower() in ('true')


def train(
    dataset: str,
    batch_size: int = 100,
    depth: int = 8,
    window_size: int = 200,
    latent_dim: int = 100,
    epochs: int = 1000,
    retrain: bool = False,
    ckpt_epoch: int = 5,
    init_lr: float = 1e-3,
    checkpoint_step: int = 5,
) -> None:
    """
    Training code.

    Parameters:
        dataset:         Name of dataset.
        batch_size:      Batch size. Default is 100.
        hidden_size:     List of the in_channels of each TNC layers of encoder.
        depth:           Depth of Encoder and Decoder of VAE.
        window_size:     Window size of the sliding window. Default is 200.
        latent_dim:      Dimension of latent space. Default is 100.
                         If the dataset is univariate, set it to 50.
        epochs:          Number of epochs. Default is 1000.
        retrain:         Flag for retraining. Default is False.
        ckpt_epoch:      If retrain, from this epoch, retraining begins. Default is 5.
        init_lr:         Initial learnig rate. Default is 1e-4.
        sched_stepsize:  Learning rate is updated once every this epochs. Default is 5.
        checkpoint_step: Model is saved once every this epochs. Default is 5.
    """
    train_data = Dataset(
        dataset=dataset,
        window_size=window_size,
        mode='train',
        convert_nan='overwrite',
    )
    data_dim = train_data.data_shape[-1]

    device = torch.device('cuda:0')

    model = VAE(
        window_size=window_size,
        data_dim=data_dim,
        latent_dim=latent_dim,
        depth=depth,
    ).to(device)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = optim.Adam(params=model.parameters(), lr=init_lr)

    scheduler = sched.StepLR(optimizer=optimizer, step_size=10, gamma=0.99)

    ckpt_dir = f'checkpoints/{dataset}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    if retrain:
        ckpt = torch.load(os.path.join(ckpt_dir, f'epoch_{ckpt_epoch}.pt'))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])

    log_dir = f'log/{dataset}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    exp_num = len(os.listdir(log_dir))
    log_path = os.path.join(log_dir, f'log_{exp_num}.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'recon_loss', 'pert_loss', 'zero_pert_loss', 'kld_loss', 'total_loss'])

    start_epoch = 0

    if retrain:
        start_epoch = ckpt_epoch
        print(f'Training loop restart from epoch {ckpt_epoch}...')
    else:
        print('Training loop start...')

    for epoch in range(start_epoch, epochs):
        recon_loss = 0.0
        pert_loss = 0.0
        zero_pert_loss = 0.0
        kld_loss = 0.0
        train_loss = 0.0

        for data in tqdm(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            mu, logvar, x_hat, x_tilde = model(data)
            recon, pert, zero_pert, kld, total_loss = loss(x=data, x_hat=x_hat, x_tilde=x_tilde, mu=mu, logvar=logvar)
            total_loss.backward()
            recon_loss += recon
            pert_loss += pert
            zero_pert_loss += zero_pert
            kld_loss += kld
            train_loss += total_loss.item()
            optimizer.step()

        scheduler.step()

        recon_loss /= len(train_loader)
        pert_loss /= len(train_loader)
        zero_pert_loss /= len(train_loader)
        kld_loss /= len(train_loader)
        train_loss /= len(train_loader)

        print(f'Epoch {epoch+1} Finished.')
        print(f'Reconstruction loss: {recon_loss:.4f}')
        print(f'Perturbation loss: {pert_loss:.4f}')
        print(f'Zero perturbation loss: {zero_pert_loss:.4f}')
        print(f'KL-Divergence loss: {kld_loss:.4f}')
        print(f'Total loss: {train_loss:.4f}')

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch+1,
                    np.round(recon_loss, 4),
                    np.round(pert_loss, 4),
                    np.round(zero_pert_loss, 4),
                    np.round(kld_loss, 4),
                    np.round(train_loss, 4),
                ]
            )
    
        if epoch == 0 or (epoch + 1) % checkpoint_step == 0:
            torch.save(
                obj={
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                },
                f=os.path.join(ckpt_dir, f'epoch_{epoch + 1}.pt')
            )
            
    print('Training Finished')

    return None


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, help='Name of the dataset.')
    args.add_argument('--hidden-list', nargs='+', type=int, help='List of the in_channels of each TNC layers of encoder.')
    args.add_argument('--batch-size', type=int, default=100, help='Batch size.')
    args.add_argument('--depth', type=int, default=8, help='Depth of Encoder and Decoder of VAE.')
    args.add_argument('--window-size', type=int, default=200, help='Window size.')
    args.add_argument('--latent-dim', type=int, default=100, help='Dimension of latent space.')
    args.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    args.add_argument('--retrain', type=str2bool, default=False, help='Flag for retraining.')
    args.add_argument('--ckpt-epoch', type=int, default=5, help='From what epoch retraining starts.')
    args.add_argument('--init-lr', type=float, default=1e-4, help='Initial learning rate.')
    args.add_argument('--checkpoint-step', type=int, default=5, help='Model is saved once every this epochs.')
    config = args.parse_args()

    train(
        dataset=config.dataset,
        batch_size=config.batch_size,
        depth=config.depth,
        window_size=config.window_size,
        latent_dim=config.latent_dim,
        epochs=config.epochs,
        retrain=config.retrain,
        ckpt_epoch=config.ckpt_epoch,
        init_lr=config.init_lr,
        checkpoint_step=config.checkpoint_step,
    )