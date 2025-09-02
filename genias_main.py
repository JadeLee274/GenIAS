import csv
from tqdm import tqdm
import argparse
import torch.optim.lr_scheduler as sched
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.common_import import *
from data_factory.loader import GenIASDataset
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


def update_radius(
    dist: Tensor,
    nu: float,
) -> float:
    """
    OPtimally solves the sphere radius for deep SVDD with the (1 - nu)-quantile
    of distances.

    Parameters:
        dist: Distances
        nu:   Quantile parameter.
    
    Retruns:
        Updated radius with the (1 - nu)-quantile of distance.
    """
    return np.quantile(
        a=np.sqrt(dist.clone().data.cpu()),
        q=1-nu
    )


def train_vae(
    dataset: str,
    batch_size: int = 100,
    depth: int = 8,
    window_size: int = 200,
    latent_dim: int = 100,
    gpu_num: int = 0,
    epochs: int = 1000,
    retrain: bool = False,
    retrain_start_epoch: Optional[int] = None,
    init_lr: float = 1e-4,
    checkpoint_step: int = 5,
) -> None:
    """
    Training code for VAE.

    Parameters:
        dataset:             Name of dataset.
        batch_size:          Batch size. Default is 100.
        hidden_size:         List of the in_channels of each TCN of encoder.
        depth:               Depth of Encoder and Decoder of VAE.
        window_size:         Window size of the sliding window. Default 200.
        latent_dim:          Dimension of latent space. Default 100.
                             If the dataset is univariate, set it to 50.
        gpu_num:             What GPU will be used for training.
        epochs:              Number of epochs. Default 1000.
        retrain:             Flag for retraining. Default False.
        retrain_start_epoch: If retrain, retraining begins from this epoch. 
        init_lr:             Initial learnig rate. Default 1e-4.
        checkpoint_step:     Model is saved once every this epochs. Default 5.
    """
    train_data = GenIASDataset(
        dataset=dataset,
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

    ckpt_dir = f'checkpoints/vae/{dataset}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    if retrain:
        ckpt = torch.load(
            os.path.join(ckpt_dir, f'epoch_{retrain_start_epoch}.pt')
        )
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])

    log_dir = f'log/vae/{dataset}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    exp_num = len(os.listdir(log_dir))
    log_path = os.path.join(log_dir, f'log_{exp_num}.csv')
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

    if retrain:
        start_epoch = retrain_start_epoch
        print(f'Training loop restart from epoch {retrain_start_epoch}...')
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

    return None


def train_svdd(
    dataset: str,
    batch_size: int = 100,
    objective: str = 'soft-boundary',
    depth: int = 4,
    window_size: int = 200,
    reperesentation_dim: int = 128,
    gpu_num: int = 0,
    epochs: int = 150,
    nu: float = 0.1,
    radius_update_epoch: int = 10,
    retrain: str2bool = False,
    retrain_start_epoch: Optional[int] = None,
    checkpoint_step: int = 5,
    init_lr: float = 1e-4,
    weight_decay: float = 1e-6,
) -> None:
    """
    Training code for deep SVDD. For details, go to the link below and click
    src -> optim -> deepSVDD_trainer.py.

    Github link: https://github.com/lukasruff/Deep-SVDD-PyTorch/tree/master

    Parameters:
        dataset:             Name of dataset.
        batch_size:          Batch size. Default 100.
        objective:           Objective of SVDD. 
                             Must be either 'soft-boundary' or 'one-class'. 
                             Default 'soft-boundary'.
        depth:               Depth of Deep SVDD model. Default 4.
        window_size:         Window size of sliding window. Default 200.
        representation_dim:  Dimension of representation space. Default 128.
        gpu_num:             What GPU will be used for training. Default 0.
        epochs:              Number of epochs. Default 100.
        nu:                  Tradeoff between penalties and the sphere volume.
                             Default 0.1.
        radius_update_epoch: Radius is updated from this epoch. Default 10.
        milestone:           Radius of sphere is updated once in this epoch.
                             Default 10.
        retrain:             Flag for retraining. Default False.
        retrain_start_epoch: If retrain, retraining begins from this epoch.
        checkpoint_step:     Model is saved once every this epochs. Default 5.
        init_lr:             Initial learning rate. Default 1e-4.
    """
    train_data = GenIASDataset(
        dataset=dataset,
        window_size=window_size,
        mode='train',
        convert_nan='overwrite',
    )

    data_dim = train_data.data_shape[-1]
    
    device = torch.device(f'cuda:{gpu_num}')

    model = SVDD(
        data_dim=data_dim,
        window_size=window_size,
        depth=depth,
        representation_dim=reperesentation_dim,
    )

    model = model.to(device)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=init_lr,
        weight_decay=weight_decay,
    )

    scheduler = sched.StepLR(optimizer=optimizer, step_size=5, gamma=0.95)

    ckpt_dir = f'checkpoints/svdd/{dataset}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    
    if retrain:
        ckpt = torch.load(
            os.path.join(ckpt_dir, f'epoch_{retrain_start_epoch}.pt')
        )
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])

    log_dir = f'log/svdd/{dataset}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    exp_num = len(os.listdir(log_dir))
    log_path = os.path.join(log_dir, f'log_{exp_num}.csv')

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                'epoch',
                'train_loss',
            ]
        )
    
    start_epoch = 0
    
    # Initialize radius by zero.
    radius = torch.tensor(0.0, device=device)

    # Initialize sphere center as the mean of initial forward pass on data.
    center = torch.zeros(reperesentation_dim, device=device)
    n_samples = 0
    model.eval()
    with torch.no_grad():
        for data in train_loader:
            data = data.to(device)
            outputs = model(data)
            n_samples += outputs.shape[0]
            center += torch.sum(outputs, dim=[0, 1])
    center /= n_samples

    print('Initialized SVDD radius center...')

    if retrain:
        start_epoch = retrain_start_epoch
        print(f'Training loop restart from epoch {start_epoch}...')
    else:
        print('Training loop start...')

    model.train()
    for epoch in range(start_epoch, epochs):
        print(f'Epoch {epoch + 1} start')
        train_loss = 0.0
        for data in tqdm(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            dist, loss = svdd_loss(
                objective=objective,
                radius=radius,
                center=center,
                nu=nu,
                x=model(data),
            )
            loss.backward()
            optimizer.step()
            if objective == 'soft-boundary' and epoch >= radius_update_epoch:
                radius.data = torch.tensor(
                    data=update_radius(dist=dist, nu=nu),
                    device=device,
                )
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f'Epoch {epoch + 1} train loss: {train_loss:.4e}')

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch+1,
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

        scheduler.step()
    
    print('Training finished.')

    return None


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument(
        '--train-model',
        type=str,
        help="Which model to train. Either 'VAE' or 'SVDD'."
    )
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
        '--vae-depth',
        type=int,
        default=10,
        help='Depth of encoder and decoder of VAE. Default 8.'
    )
    args.add_argument(
        '--svdd-depth',
        type=int,
        default=8,
        help='Depth of SVDD. Default 4.'
    )
    args.add_argument(
        '--window-size',
        type=int,
        default=200,
        help='Window size. Default 200.'
    )
    args.add_argument(
        '--latent-dim',
        type=int,
        default=100,
        help='Dimension of latent space of VAE. Default 100.'
    )
    args.add_argument(
        '--representation-dim',
        type=int,
        default=128,
        help='Dimension of representation space of SVDD. Default 128.'
    )
    args.add_argument(
        '--svdd-objective',
        type=str,
        default='soft-boundary',
        help='Objective of SVDD. Either one-class or soft-boundary. \
              Default soft-boundary.'
    )
    args.add_argument(
        '--gpu-num',
        type=int,
        default=0,
        help='What GPU will be used for training. Default 0.'
    )
    args.add_argument(
        '--vae-epochs',
        type=int,
        default=1000,
        help='Number of epochs for training VAE. Default 1000.'
    )
    args.add_argument(
        '--svdd-epochs',
        type=int,
        default=150,
        help='Number of epochs for training SVDD. Default 1000.'
    )
    args.add_argument(
        '--retrain',
        type=str2bool,
        default=False,
        help='Flag for retraining. Default False.'
    )
    args.add_argument(
        '--retrain-start-epoch',
        type=int,
        default=5,
        help='From what epoch retraining starts. Default 5.'
    )
    args.add_argument(
        '--init-lr',
        type=float,
        default=1e-4,
        help='Initial learning rate. Default 1e-4.'
    )
    args.add_argument(
        '--checkpoint-step',
        type=int,
        default=5,
        help='Model is saved once every this epochs. Default 5.'
    )

    config = args.parse_args()

    assert config.train_model in ('VAE', 'SVDD'), \
    'train-model must be either VAE or SVDD'
    
    if config.train_model == 'VAE':
        train_vae(
            dataset=config.dataset,
            batch_size=config.batch_size,
            depth=config.vae_depth,
            window_size=config.window_size,
            latent_dim=config.latent_dim,
            gpu_num=config.gpu_num,
            epochs=config.vae_epochs,
            retrain=config.retrain,
            ckpt_epoch=config.retrain_start_epoch,
            init_lr=config.init_lr,
            checkpoint_step=config.checkpoint_step,
        )

    elif config.train_model == 'SVDD':
        train_svdd(
            dataset=config.dataset,
            batch_size=config.batch_size,
            objective=config.svdd_objective,
            depth=config.svdd_depth,
            window_size=config.window_size,
            reperesentation_dim=config.representation_dim,
            gpu_num=config.gpu_num,
            epochs=config.svdd_epochs,
            retrain=config.retrain,
            retrain_start_epoch=config.retrain_start_epoch,
            checkpoint_step=config.checkpoint_step,
            init_lr=config.init_lr,
        )
