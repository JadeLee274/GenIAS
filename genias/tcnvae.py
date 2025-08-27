from typing import *
import warnings
import torch.nn as nn
from .tcn import TemporalConvNet
from utils.common_import import *
warnings.filterwarnings('ignore')
"""
Codes for TCN-VAE. This code follows the paper
Darban et al., 2025, GenIAS: Generator for Instantiating Anomalies in Time Series.

Paper link: https://arxiv.org/pdf/2502.08262

Default values of arguments follow the paper.
"""


class Encoder(nn.Module):
    def __init__(
        self,
        window_size: int,
        num_features: int,
        depth: int,
        latent_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = TemporalConvNet(
            in_channels=window_size,
            hidden_channels=[window_size] * depth,
            dropout=dropout,
        )
        self.fc_mu = nn.Linear(
            in_features=num_features,
            out_features=latent_dim,
        )
        self.fc_logvar = nn.Linear(
            in_features=num_features,
            out_features=latent_dim,
        )
        nn.init.xavier_normal_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_normal_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.encoder(x) # (B, W, F) -> (B, 1, F)
        mu = self.fc_mu(x) # (B, 1, F) -> (B, 1, L)
        logvar = self.fc_logvar(x) # (B, 1, F) -> (B, 1, L)
        return mu, logvar # (B, 1, L), (B, 1, L)


class Decoder(nn.Module):
    """
    Decoder layer of VAE. Here, TCN is not used.
    Given the input of the encoder with size (B, W, F), where

    B = Batch size,
    W = Window size,
    F = Number of features of each data point,

    the parameter of decoder is given as follows. 

    Parameters:
        latent_dim:   Dimension of latent space encoded by the encoder.  
        window_size:  W
        num_features: F
        hidden_list:  List of the in_channels of each TNC layers.
                      This should be the reverse of hidden_list of Encoder.
        dropout:      In what probabiliy that the dropout layer of eacn TCN layer
                      will be activated.
    """
    def __init__(
        self,
        latent_dim: int,
        window_size: int,
        num_features: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        activation = nn.ReLU()
        hidden_list = [window_size] * depth
        dec_layer = [
            nn.Linear(
                in_features=latent_dim,
                out_features=num_features,
            ),
            activation,
        ]

        for i in range(depth - 1):
            dec_layer.append(
                nn.ConvTranspose1d(
                    in_channels=hidden_list[i],
                    out_channels=hidden_list[i + 1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            dec_layer.append(activation)
            dec_layer.append(nn.Dropout(p=dropout))
        
        dec_layer.append(
            nn.ConvTranspose1d(
                in_channels=hidden_list[depth - 1],
                out_channels=window_size,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        dec_layer.append(activation)
        self.dec_layer = nn.Sequential(*dec_layer)

        self._init_weights()

    def forward(self, z: Tensor) -> Tensor:
        out = self.dec_layer(z)
        return out

    def _init_weights(self) -> None:
        for m in self.parameters():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)


class VAE(nn.Module):
    """
    TCN-based VAE. Uses Encoder and Decoder classes for encoder and decoder
    layers. Given the input with shape (B, W, F), where 

    B = Batch size,
    W = Window length,
    F = Number of features of each data point,

    the parameters are given as follows.

    Parameters:
        window_size:   W
        data_dim:      F
        latent_dim:    Dimension of latent space.
        hidden_list:   List of the in_channels of each TNC layers of encoder.
                       It decides the depth of encoder and decoder.
                       The hidden_list of decoder is the reversed version of it.
        tcn_depth:     Depth of the TCN layer.
        perturb_const: Perturbation constant for the perturbation in the latent space.
    """
    def __init__(
        self,
        window_size: int,
        data_dim: int,
        latent_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            window_size=window_size,
            num_features=data_dim,
            depth=depth,
            latent_dim=latent_dim,
            dropout=0.1,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            window_size=window_size,
            num_features=data_dim,
            depth=depth,
            dropout=0.1,
        )
        self.psi = nn.Parameter(
            data=torch.ones(1, latent_dim),
            requires_grad=True,
        )
    
    def reparam_and_perturb(
        self,
        mu: Tensor,
        logvar: Tensor,
        psi: nn.Parameter,
    ) -> Tuple[Tensor, Tensor]:
        eps = torch.randn_like(mu)
        sigma = torch.exp(0.5 * logvar)
        z_recon = mu + eps * sigma
        z_pert = mu + psi * eps * sigma
        return z_recon, z_pert
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        mu, logvar = self.encoder(x)
        z_recon, z_pert = self.reparam_and_perturb(mu, logvar, psi=self.psi)
        x_hat = self.decoder(z_recon)
        x_tilde = self.decoder(z_pert)
        return mu, logvar, x_hat, x_tilde
