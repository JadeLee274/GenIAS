from typing import *
import os
import csv
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_factory.loader import Dataset
from models.tcnvae import VAE
from utils.loss import loss


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
    batch_size: int,
    hidden_list: List[int],
    tcn_depth: int = 3,
    window_size: int = 200,
    latent_dim: int = 100,
    epochs: int = 1000,
    retrain: bool = False,
    ckpt_epoch: int = 5,
    init_lr: float = 1e-4,
    sched_stepsize: int = 5,
    checkpoint_step: int = 5,
) -> None:
    """
    Training code.

    Parameters:
        dataset:         Name of dataset.
        batch_size:      Batch size.
        hidden_size:     List of the in_channels of each TNC layers of encoder.
        tcn_depth:       Depth of TCN.
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

    model = VAE(
        batch_size=batch_size,
        window_size=window_size,
        data_dim=data_dim,
        latent_dim=latent_dim,
        hidden_list=hidden_list,
        tcn_depth=tcn_depth,
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = optim.Adam(params=model.parameters(), lr=init_lr)

    scheduler = optim.lr_scheduler.StepLR(
        optmizer=optimizer,
        step_size=sched_stepsize,
        gamma=0.95,
    )

    ckpt_dir = f'/checkpoints/{dataset}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    if retrain:
        ckpt = torch.load(os.path.join(ckpt_dir, f'epoch_{ckpt_epoch}.pt'))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        scheduler.load_state_dict(ckpt['sched'])

    log_dir = f'/log/{dataset}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'recon_loss', 'pert_loss', 'zero_pert_loss', 'kld_loss', 'total_loss'])

    writer = csv.writer()

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
        kld_loss /= len(kld_loss)
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
                    np.round(total_loss, 4),
                ]
            )
    
        if epoch == 0 or (epoch + 1) % checkpoint_step == 0:
            torch.save(
                obj={
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'sched': scheduler.state_dict(),
                },
                f=os.path.join(ckpt_dir, f'epoch_{epoch + 1}.pt')
            )
    print('Training Finished')

    return None


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, help='Name of the dataset.')
    args.add_argument('--batch-size', type=int, help='Batch size.')
    args.add_argument('--hidden-list', nargs='+', type=int, help='List of the in_channels of each TNC layers of encoder.')
    args.add_argument('--tcn-depth', type=int, default=3, help='Depth of TCN layer.')
    args.add_argument('--window-size', type=int, default=200, help='Window size.')
    args.add_argument('--latent-dim', type=int, default=100, help='Dimension of latent space.')
    args.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    args.add_argument('--retrain', type=str2bool, default=False, help='Flag for retraining.')
    args.add_argument('--ckpt-epoch', type=int, help='From what epoch retraining starts')
    args.add_argument('--init-lr', type=float, default=1e-4, help='Initial learning rate.')
    args.add_argument('--sched-stepsize', type=int, default=5, help='Learning rate is updated once every this epoch.')
    args.add_argument('--checkpoint-step', type=int, default=5, help='Model is saved once every this epochs.')
    config = args.parse_args()

    train(
        dataset=config.dataset,
        batch_size=config.batch_size,
        hidden_list=config.hidden_list,
        tcn_depth=config.tcn_depth,
        window_size=config.window_size,
        latent_dim=config.latent_dim,
        epochs=config.epochs,
        retrain=config.retrain,
        ckpt_epoch=config.ckpt_epoch,
        init_lr=config.init_lr,
        sched_stepsize=config.sched_stepsize,
        checkpoint_step=config.checkpoint_step,
    )