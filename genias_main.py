"""
Codes for training of TCN-VAE. This code follows the paper
GenIAS: Generator for Instantiating Anomalies in Time Series,
Darban et al., 2025.

Paper link: https://arxiv.org/pdf/2502.08262

The default values of arguments follow the paper.
"""
import argparse, logging
import torch.optim.lr_scheduler as sched
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.common_import import *
from utils.set_logging import set_logging_filehandler
from data_factory.loader import GenIASDataset
from genias.tcnvae import VAE
from utils.loss import vae_loss


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
    dataset: str,
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
        dataset=dataset,
        subdata=subdata,
        window_size=window_size,
    )
    data_dim = train_data.data_dim

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

    ckpt_dir = os.path.join('checkpoints/vae', dataset)

    if dataset in ['MSL', 'SMAP', 'SMD', 'Yahoo-A1', 'KPI']:
        ckpt_dir = os.path.join(ckpt_dir, subdata)
        
    os.makedirs(ckpt_dir, exist_ok=True)

    logging.info(f'Training loop on {dataset} {subdata} dataset start...\n')

    for epoch in range(epochs):
        recon_loss = 0.0
        pert_loss = 0.0
        zero_pert_loss = 0.0
        kld_loss = 0.0
        train_loss = 0.0

        for data in train_loader:
            data = data.to(device).float()
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

        logging.info(f'Epoch {epoch+1} loss:')
        logging.info(f'- Reconstruction loss: {recon_loss:.4f}')
        logging.info(f'- Perturbation loss: {pert_loss:.4f}')
        logging.info(f'- Zero perturbation loss: {zero_pert_loss:.4f}')
        logging.info(f'- KL-Divergence loss: {kld_loss:.4f}')
        logging.info(f'- Total loss: {train_loss:.4f}\n')

        if epoch == 0 or (epoch + 1) % checkpoint_step == 0:
            torch.save(
                obj={
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                },
                f=os.path.join(ckpt_dir, f'epoch_{epoch + 1}.pt')
            )
            
    logging.info('Training Finished')

    return


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        '--dataset',
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
        help='Depth of encoder and decoder of VAE. Default 10.'
    )
    args.add_argument(
        '--gpu-num',
        type=int,
        default=0,
        help='What GPU will be used for training. Default 0.'
    )
    config = args.parse_args()

    if config.dataset in ['MSL', 'SMAP', 'SMD', 'Yahoo-A1', 'KPI']:
        data_dir = f'data/{config.dataset}/train'
        train_list = sorted(os.listdir(f'{data_dir}/train'))
        train_list = [f.replace('.npy', '') for f in train_list]

        for subdata in train_list:
            set_logging_filehandler(
                log_file_path=f'log/vae/{config.dataset}.log'
            )
            train_vae(
                dataset=config.dataset,
                subdata=subdata,
                gpu_num=config.gpu_num
            )
    
    else:
        set_logging_filehandler(log_file_path=f'log/vae/{config.dataset}.log')
        train_vae(
            dataset=config.dataset,
            gpu_num=config.gpu_num
        )
