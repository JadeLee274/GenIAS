import warnings
from torch.nn.utils import weight_norm
from genias.utils.common_import import *
warnings.filterwarnings('ignore')


class Chomp1d(nn.Module):
    """
    Discard the features of input.

    Parameters:
        chomp_size: How many features that you want to discard.
    """
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Discard the features of input

        Parameters:
            x: Tensor with size (B, W, F), where
               B = Batch size
               W = Window length
               F = Number of Features

        Returns:
            The clipped x with size (B, W, F - chomp_size).
            It is ensured the memory continuity.
        """
        return x[:, :, :-self.chomp_size].contiguous()
    

class TemporalBlock(nn.Module):
    """
    Block for the Temporal Convolutional Netrowk.

    Parameters:
        n_inputs:    The number of features of the input tensor.
        n_outputs:   The number of features of the output tensor.
        kernel_size: The size of the kernel of convolution.
        stride:      The step size of the kernel of convolution.
        dilation:    The dilation of the kernel of convolution.
        padding:     How many to pad to the input data.
        dropout:     The probability of dropout layers.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.activation = nn.ReLU()
        self.conv1 = weight_norm(
            module=nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(chomp_size=padding)
        self.dropout1 = nn.Dropout(p=dropout)

        self.conv2 = weight_norm(
            module=nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(chomp_size=padding)
        self.dropout2 = nn.Dropout(p=dropout)

        self.conv3 = weight_norm(
            module=nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp3 = Chomp1d(chomp_size=padding)
        self.dropout3 = nn.Dropout(p=dropout)


        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.dropout1,
            self.activation,
            self.conv2,
            self.chomp2,
            self.dropout2,
            self.activation,
            self.conv3,
            self.chomp3,
            self.dropout3,
            self.activation,
        )
        self.downsample = None

        if in_channels != out_channels:
            self.downsample = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            )
        
        self._init_weights()

    def _init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)

        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

        return None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the block.

        Parameters:
            x: Input tensor with size (B, W, F)
        """
        out = self.net(x)
        residual = self.downsample(x) if self.downsample else x
        return self.activation(out + residual)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network layer.

    Parameters:
        in_channels:     The number of channels of input tensor.
        hidden_channels: The list of channels of the hidden layers.
        kernel_size:     The size of the kernel of the convolutional layer.
        dropout:         The probability of dropout.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        n_levels = len(hidden_channels)

        for i in range(n_levels):
            dilation_size = 2 ** i
            in_channels = in_channels if i == 0 else hidden_channels[i - 1]
            out_channels = hidden_channels[i]
            layers += [
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


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
        x = self.encoder(x) # (B, W, F) -> (B, W, F)
        mu = self.fc_mu(x) # (B, W, F) -> (B, W, L)
        logvar = self.fc_logvar(x) # (B, W, F) -> (B, W, L)
        return mu, logvar # (B, W, L), (B, W, L)


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
        dropout:      In what probabiliy that the dropout layer of eacn TCN 
                      layer will be activated.
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
                       The hidden_list of decoder is its reversed version.
        tcn_depth:     Depth of the TCN layer.
        perturb_const: Perturbation constant for the perturbation 
                       in the latent space.
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
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encoder(x)
        z_recon, z_pert = self.reparam_and_perturb(mu, logvar, psi=self.psi)
        x_hat = self.decoder(z_recon)
        x_tilde = self.decoder(z_pert)
        return mu, logvar, x_hat, x_tilde
