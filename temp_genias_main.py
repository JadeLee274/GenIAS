import csv
from tqdm import tqdm
import argparse
import torch.optim.lr_scheduler as sched
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.common_import import *
from data_factory.temp_loader import GenIASDataset
from genias.tcnvae import VAE
from deepsvdd.deepsvdd import SVDD
from utils.loss import vae_loss, svdd_loss
"""
Codes for training of TCN-VAE. This code follows the paper
GenIAS: Generator for Instantiating Anomalies in Time Series,
Darban et al., 2025.

Paper link: https://arxiv.org/pdf/2502.08262

The default values of arguments follow the paper.
"""


def str2bool(v: str) -> bool:
    """
    Converts string (Either True or False) to bool.

    Parameters:
        v: String instance. Either 'True' or 'False.'

    Returns:
        Boolean instance. Either True or False.
    """
    return v.lower() in ('true')


def train_vae(
    data: str,
    subdata: str,
    batch_size: int = 100,
    depth: int = 10,
    window_size: int = 200,
    latent_dim: int = 100,
    gpu_num: int = 0,
    epochs: int = 1000,
    init_lr: float = 1e-4,
    checkpoint_step: int = 100,
) -> None:
    train_data = GenIASDataset(
        data_name=data,
        subdata=subdata,
        window_size=window_size,
        mode='train',
        convert_nan='overwrite',
    )
    data_dim = train_data.data_shape[-1]

    device = torch.device(f'cuda:{gpu_num}')

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

    ckpt_dir = f'temp_checkpoints/{data}/{subdata}'
    os.makedirs(ckpt_dir, exist_ok=True)

    log_dir = f'temp_log/{data}'
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f'{subdata}_log.csv')
    
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                'epoch',
                'recon_loss',
                'pert_loss',
                'zero_pert_loss',
                'kld_loss',
                'total_loss'
            ]
        )

    start_epoch = 0

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
            recon, pert, zero_pert, kld, total_loss = vae_loss(
                x=data,
                x_hat=x_hat,
                x_tilde=x_tilde,
                mu=mu,
                logvar=logvar
            )
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

    return


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        '--data',
        type=str,
        help='Name of the dataset.'
    )
    args.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size. Default 100.'
    )
    args.add_argument(
        '--depth',
        type=int,
        default=10,
        help='Depth of encoder and decoder of VAE. Default 8.'
    )
    args.add_argument(
        '--gpu-num',
        type=int,
        default=0,
        help='What GPU will be used for training. Default 0.'
    )
    config = args.parse_args()

    train_list = sorted(
        os.listdir(f'/data/seungmin/{config.data}_SEPARATED/train')
    )
    train_list = [f.replace('_train.npy', '') for f in train_list]
    
    for subdata in train_list:
        train_vae(data=config.data, subdata=subdata)
