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
        in_channels:    F
        tcn_depth:      Depth of the TCN layer.
        hidden_list:    List of the in_channels of each TNC layers.
                        It decides the number of TCN layers.
        dropout:        In what probabiliy that the dropout layer of eacn TCN layer
                        will be activated.
        latent_dim:     The dimension of latent space.
    """
    def __init__(
        self,
        window_size: int,
        num_features: int,
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
            in_channels = num_features if i == 0 else hidden_list[i - 1]
            out_channels = hidden_list[i]
            enc_layer.append(
                TemporalConvNet(
                    in_channels=in_channels,
                    hidden_channels=[out_channels] * tcn_depth,
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
            in_features=window_size,
            out_features=latent_dim,
        )
        self.fc_logvar = nn.Linear(
            in_features=window_size,
            out_features=latent_dim,
        )

        self._init_linear_weights()
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.transpose(1, 2)      # B, W, F -> B, F, W
        x = self.enc_layer(x)      # B, F, W -> B, 1, W
        x = x.squeeze(1)           # B, 1, W -> B, W
        mu = self.fc_mu(x)         # B, W -> B, L
        logvar = self.fc_logvar(x) # B, W -> B, L
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
        window_size: int,
        num_features: int,
        hidden_list: List[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        dec_depth = len(hidden_list)

        dec_layer = [
            nn.Linear(
                in_features=latent_dim,
                out_features=window_size,
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
                out_channels=num_features,
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
        out = out.transpose(1, 2)
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
            window_size=window_size,
            num_features=data_dim,
            tcn_depth=tcn_depth,
            hidden_list=hidden_list,
            dropout=0.2,
            latent_dim=latent_dim,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            window_size=window_size,
            num_features=data_dim,
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
