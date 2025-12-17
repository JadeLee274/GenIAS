import warnings
from torch.nn.utils import weight_norm
from exp.utils.common_import import *
from exp.models.tcn import *
    

class TCNPerturbator(nn.Module):
    def __init__(
        self,
        window_size: int,
        data_dim: int,
        latent_perturbator_factor: float,
    ) -> None:
        super().__init__()
        # Data informations
        self.window_size = window_size
        self.data_dim = data_dim

        # Parameters
        self.encoder = TemporalConvNet(
            in_channels=data_dim,
            hidden_channels=[data_dim, data_dim*2, data_dim*4, data_dim*8],
            # hidden_channels=[data_dim, data_dim*2, data_dim*4],
        )
        self.latent_sigma_pert = nn.Parameter(
            latent_perturbator_factor * torch.ones(window_size, data_dim*4)
        )
        # self.latent_sigma_pert = nn.Parameter(
        #     latent_perturbator_factor * torch.ones(window_size, data_dim*2),
        # )
        self.decoder = TemporalConvNet(
            in_channels=data_dim*4,
            hidden_channels=[data_dim*4, data_dim*2, data_dim*2],
            # in_channels=data_dim*2,
            # hidden_channels=[data_dim*2, data_dim*2, data_dim*2]
        )
        
        return
    
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        encoder_out = self.encoder.forward(x)
        mu = encoder_out[..., :self.data_dim*4]
        logvar = encoder_out[..., self.data_dim*4:]
        # mu = encoder_out[..., :self.data_dim*2]
        # logvar = encoder_out[..., self.data_dim*2:]
        return mu, logvar

    def reparametrize(
        self,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        eps = torch.randn_like(mu).to(mu.device)
        sigma = torch.exp(0.5 * logvar)
        z_recon = mu + eps * sigma
        z_pert = mu + self.latent_sigma_pert * eps * sigma
        return z_recon, z_pert
    
    def decode(self, z: Tensor) -> Tensor:
        decoder_out = self.decoder.forward(z)
        mult_factor: Tensor = decoder_out[..., :self.data_dim]
        add_factor: Tensor = decoder_out[..., self.data_dim:]
        return mult_factor, add_factor
    
    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)

        z_recon, z_pert = self.reparametrize(mu=mu, logvar=logvar)

        recon_mult, recon_add = self.decode(z_recon)
        pert_mult, pert_add = self.decode(z_pert)

        x_recon = x * recon_mult + recon_add
        x_pert = x * pert_mult + pert_add

        return x_recon, x_pert, mu, logvar

    def init_perturbator(
        self,
        time: str,
        data: str,
        subdata: Optional[str],
        epoch: int = 50,
    ) -> None:
        ckpt_dir = os.path.join('exp', 'checkpoints', data)

        if subdata is not None:
            ckpt_dir = os.path.join(ckpt_dir, subdata)
        
        ckpt_dir = os.path.join(
            ckpt_dir, 'perturbator', time, f'epoch_{epoch}.pt'
        )
        ckpt = torch.load(ckpt_dir)
        self.load_state_dict(ckpt['perturbator'])
        self.seed = ckpt['seed']
        
        return


class TCNDiscriminator(nn.Module):
    def __init__(
        self,
        window_size: int,
        data_dim: int,
    ) -> None:
        super().__init__()
        self.tcn = TemporalConvNet(
            in_channels=window_size,
            hidden_channels=[window_size, window_size//2, window_size//4, 1],
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features=data_dim, out_features=data_dim//2),
            nn.ReLU(),
            nn.Linear(in_features=data_dim//2, out_features=data_dim//4),
            nn.ReLU(),
            nn.Linear(in_features=data_dim//4, out_features=1),
            nn.Sigmoid(),
        )
        return
    
    def _init_weights(self) -> None:
        for param in self.parameters():
            if isinstance(param, nn.Linear):
                nn.init.xavier_normal_(param.weight)
                nn.init.zeros_(param.bias)
        return
    
    def forward(self, x: Tensor) -> Tensor:
        tcn_out: Tensor = self.tcn.forward(x.transpose(-2, -1))
        mlp_out: Tensor = self.mlp.forward(tcn_out.transpose(-2, -1))
        label = mlp_out.squeeze(-1)
        return label
    
    def init_discriminator(
        self,
        time: str,
        data: str,
        subdata: Optional[str],
        epoch: int = 100,
    ) -> None:
        ckpt_dir = os.path.join('exp', 'checkpoints', data)

        if subdata is not None:
            ckpt_dir = os.path.join(ckpt_dir, subdata)
        
        ckpt_dir = os.path.join(
            ckpt_dir, 'perturbator', time, f'epoch_{epoch}.pt'
        )
        ckpt = torch.load(ckpt_dir)
        self.load_state_dict(ckpt['discriminator'])
        
        return
    

class TCNPerturbatorTemp(nn.Module):
    def __init__(
            self,
            window_size: int,
            data_dim: int,
            latent_perturbator_factor: float,
    ) -> None:
        super().__init__()
        # Data information
        self.window_size = window_size
        self.data_dim = data_dim

        # Parameters
        self.encoder = TemporalConvNet(
            in_channels=data_dim,
            hidden_channels=[data_dim, data_dim*2]
        )
        latent_dim = 50 if data_dim == 1 else 100
        self.mu = nn.Linear(
            in_features=data_dim*2,
            out_features=latent_dim,
        )
        self.logvar = nn.Linear(
            in_features=data_dim*2,
            out_features=latent_dim,
        )
        self.latent_sigma_pert = nn.Parameter(
            latent_perturbator_factor * torch.ones(window_size, latent_dim)
        )
        self.fc = nn.Linear(
            in_features=latent_dim,
            out_features=data_dim*2,
        )
        self.decoder = TemporalConvNet(
            in_channels=data_dim*2,
            hidden_channels=[data_dim*2, data_dim*2],
        )

        return
    
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        enc_out = self.encoder.forward(x)
        mu = self.mu.forward(enc_out)
        logvar = self.logvar.forward(enc_out)
        return mu, logvar
    
    def reparametrize(
        self,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        eps = torch.randn_like(mu).to(mu.device)
        sigma = torch.exp(0.5 * logvar)
        z_recon = mu + eps * sigma
        z_pert = mu + self.latent_sigma_pert * eps * sigma
        return z_recon, z_pert

    def decode(self, z: Tensor) -> Tensor:
        z_fc = self.fc.forward(z)
        decoder_out = self.decoder.forward(z_fc)
        mult_factor: Tensor = decoder_out[..., :self.data_dim]
        add_factor: Tensor = decoder_out[..., self.data_dim:]
        return mult_factor, add_factor
    
    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z_recon, z_pert = self.reparametrize(mu=mu, logvar=logvar)
        
        recon_mult, recon_add = self.decode(z=z_recon)
        pert_mult, pert_add = self.decode(z=z_pert)

        x_recon = x * recon_mult + recon_add
        x_pert = x * pert_mult + pert_add

        return x_recon, x_pert, mu, logvar
    
    def init_perturbator(
        self,
        time: str,
        data: str,
        subdata: Optional[str],
        epoch: int = 50,
    ) -> None:
        ckpt_dir = os.path.join('exp', 'checkpoints', data)

        if subdata is not None:
            ckpt_dir = os.path.join(ckpt_dir, subdata)
        
        ckpt_dir = os.path.join(
            ckpt_dir, 'perturbator', time, f'epoch_{epoch}.pt'
        )
        ckpt = torch.load(ckpt_dir)
        self.load_state_dict(ckpt['perturbator'])
        
        return
