from typing import *
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
"""
Codes for loss function of TCN-VAE. This code follows the paper:
Paper link: https://arxiv.org/pdf/2502.08262
"""


mse_loss = nn.MSELoss()


def recon_loss(x: Tensor, x_hat: Tensor) -> Tensor:
    """
    Reconstruction loss.

    Parameters:
        x:     Original data.
        x_hat: Reconstructed data.

    Returns:
        MSE loss between x and x_hat.
    """
    return mse_loss(x, x_hat)


def pert_loss(
    x: Tensor,
    x_hat: Tensor,
    x_tilde: Tensor,
    delta_min: float = 0.1,
    delta_max: float = 0.2,
) -> Tensor:
    """
    Perturbation loss of VAE.

    Parameters:
        x:         Original data, i.e., anchor.
        x_hat:     Reconstrucded data, without anomalies injected.
        x_tilde:   Reconstructed data, with anomalies injected.

        delta_min: Minmum threshold of recon(x, x_hat) - recon(x, x_tilde)
                   Following the paper, default is 0.1.

        delta_max: Maxmum threshold of recon(x, x_tilde)
                   Following the paper, default is 0.2.

    Returns:
        Perturbation loss.

    Here, the defaults of delta_min and delta_max is set to 0.1 and 0.2,
    following section 4.8.3 of the paper.
    """
    return F.relu(recon_loss(x, x_hat) - recon_loss(x, x_tilde) + delta_min) \
    + F.relu(recon_loss(x, x_tilde) - delta_max)


def zero_pert_loss(x: Tensor, x_tilde: Tensor) -> Tensor:
    """
    Zero perturbatino loss of vAE.
    Follows the equation (4) of the paper.

    Parameters:
        x:       Input of VAE.
        x_tilde: Perturbed output of VAE.

    Returns:
        Zero perturbation loss.
    """
    return torch.mean((recon_loss(x, x_tilde) + 1) ** -1)


def kld_loss(
    mu: Tensor,
    logvar: Tensor,
    prior_var: float,
) -> Tensor:
    """
    Modified KL-Divergence loss of VAE.
    Follows the equation (5) of the paper.
    The existence of logvar_prior is justified in the theorem that is proved at
    the appendix of the paper.

    Paramters:
        mu:           Mean of latent space, from the encodr.
        logvar:       Log variance of the latent space, from the encoder.
        logvar_prior: Prior log variance of the latent space.


    Returns:
        KL-Divergence loss.
    """
    return -0.5 * torch.sum(
        input=(1 + logvar - mu ** 2 
               - torch.exp(logvar)/prior_var + 2 * prior_var
        ),
        dim=1,
    )


def loss(
    x: Tensor,
    x_hat: Tensor,
    x_tilde: Tensor,
    mu: Tensor,
    logvar: Tensor,
    prior_var: float = 0.5,
    recon_weight: float = 1.0,
    pert_weight: float = 0.1,
    zero_pert_weight: float = 0.01,
    kld_weight: float = 0.1,
) -> Tensor:
    """
    Total loss function of VAE.

    Parameters:
        x:                Input of VAE.
        x_hat:            Reconstructed output, from the VAE.
        x_tilde:          Perturbed output, from the VAE.
        mu:               Mean of latent space, from the encoder.
        logvar:           Log variance of latent space, from the encoder.

        prior_var:        Prior variance prior of the latent space.
                          Following the paper, default is 0.5.

        recon_weight:     Weight of the reconstruction loss.
                          Following the paper, default is 1.0.
                          
        pert_weight:      Weight of the perturbation loss.
                          Following the paper, default is 0.1.
                          
        zero_pert_weight: Weight of the zero perturbation loss.
                          Following the paper, default is 0.01.
                          It the dataset if univariate dataset, set it to 0.0.

        kld_weight:       Weight of the kld loss.
                          Following the paper, default is 0.1.

    Returns:
        Total loss of VAE.

    Here, the defaults of recon_weight, pert_weight, zero_pert_weight, kld_weight
    are given by 1.0., 0.1, 0.01, 0.1, following section 4.4 of the paper.
    """
    return recon_weight * recon_loss(x, x_hat) \
    + pert_weight * pert_loss(x, x_hat, x_tilde) \
    + zero_pert_weight * zero_pert_loss(x, x_tilde) \
    + kld_weight * kld_loss(mu, logvar, prior_var=prior_var)
