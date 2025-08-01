from typing import *
import torch
from torch import Tensor
import torch.nn as nn
from .tcn import TemporalConvNet


class Encoder(nn.Module):
    """
    TCN-based Encoder layer of VAE.
    Given the input with shape (B, W, F), where
    
    B = Batch size,
    W = Window size,
    F = Number of features of each data point,

    the parameters of encoder is given as follows.

    Parameters:
        in_channels:    W
        input_features: F
        tcn_depth:      Depth of the TCN layer.
        hidden_list:    List of the in_channels of each TNC layers.
                        It decides the number of TCN layers.
        dropout:        In what probabiliy that the dropout layer of eacn TCN layer
                        will be activated.
        latent_dim:     The dimension of latent space.
    """
    def __init__(
        self,
        in_channels: int,
        input_features: int,
        tcn_depth: int,
        hidden_list: List[int],
        dropout: float,
        latent_dim: int,
    ) -> None:
        super().__init__()
        enc_depth = len(hidden_list)

        enc_layer = []
        activation = nn.ReLU()

        for i in range(enc_depth - 1):
            channels_in = in_channels if i == 0 else hidden_list[i - 1]
            channels_out = hidden_list[i]
            enc_layer.append(
                TemporalConvNet(
                    in_channels=channels_in,
                    hidden_channels=[channels_out] * tcn_depth,
                )
            )
            enc_layer.append(activation)
            enc_layer.append(nn.Dropout(p=dropout))
        
        enc_layer.append(
            TemporalConvNet(
                in_channels=hidden_list[enc_depth - 2],
                hidden_channels=[hidden_list[enc_depth - 1]] * tcn_depth,
            )
        )
        enc_layer.append(activation)
        self.enc_layer = nn.Sequential(*enc_layer)

        self.fc_mu = nn.Linear(
            in_features=input_features,
            out_features=latent_dim,
        )
        self.fc_logvar = nn.Linear(
            in_features=input_features,
            out_features=latent_dim,
        )

        self._init_linear_weights()
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.enc_layer(x)
        x = x.squeeze(1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def _init_linear_weights(self) -> None:
        for m in self.parameters():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        out_features: int,
        hidden_list: List[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        dec_depth = len(hidden_list)

        dec_layer = [
            nn.Linear(
                in_features=latent_dim,
                out_features=out_features,
            )
        ]
        activation = nn.ReLU()

        for i in range(dec_depth - 1):
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
                in_channels=hidden_list[dec_depth - 1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        dec_layer.append(activation)
        self.dec_layer = nn.Sequential(*dec_layer)

        self._init_weights()

    def forward(self, z: Tensor) -> Tensor:
        out = z.unsqueeze(1)
        out = self.dec_layer(out)
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
    def __init__(
        self,
        window_size: int,
        data_dim: int,
        latent_dim: int,
        hidden_list: List[int],
        tcn_depth: int,
        perturb_const: float,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            in_channels=window_size,
            input_features=data_dim,
            tcn_depth=tcn_depth,
            hidden_list=hidden_list,
            dropout=0.2,
            latent_dim=latent_dim,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            out_channels=window_size,
            out_features=data_dim,
            hidden_list=list(reversed(hidden_list)),
            dropout=0.2,
        )
        self.psi = perturb_const
    
    def reparam_and_perturb(
        self,
        mu: Tensor,
        logvar: Tensor,
        psi: float,
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
        return x_hat, x_tilde
