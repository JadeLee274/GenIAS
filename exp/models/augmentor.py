import warnings
from torch.nn.utils import weight_norm
from exp.utils.common_import import *
from cuts_plus.models.cuts_plus import CUTS_Plus_Net
from exp.models.tcn import *


class PositiveAugmentor(nn.Module):
    def __init__(
        self,
        noise_injection_step: int,
        prediction_step: int,
        noise_level: float,
        n_nodes: int,
    ) -> None:
        super().__init__()
        self.noise_injection_step = noise_injection_step
        self.prediction_step = prediction_step
        self.noise_level = noise_level
        self.causal_discoverer = CUTS_Plus_Net(n_nodes=n_nodes, data_dim=1)

        return
    
    def init_causal_discoverer(
        self,
        time: str,
        data: str,
        subdata: Optional[str]
    ) -> None:
        cuts_plus_dir = os.path.join(
            os.getcwd(), 'cuts_plus', 'checkpoints', data
        )

        if subdata is not None:
            cuts_plus_dir = os.path.join(cuts_plus_dir, subdata)
            
        cuts_plus_dir = os.path.join(cuts_plus_dir, time, 'best_model.pt')

        cuts_plus_ckpt = torch.load(cuts_plus_dir)
        self.causal_discoverer.load_state_dict(cuts_plus_ckpt['model'])
        
        return

    def forward(self, x: Tensor, causality_matrix: Matrix) -> Tensor:
        x_positive = x.clone()
        causality_matrix = (causality_matrix > 0.5).int()
        
        start_node = random.choice(range(len(causality_matrix)))
        self.start_node = start_node
        
        effects = [
            i for i, val in enumerate(causality_matrix[start_node]) if val == 1
        ]
        self.effects = effects

        x_positive[:, self.noise_injection_step, start_node] \
        += torch.randn(x.size(0)).to(x.device) * self.noise_level

        with torch.no_grad():
            graph = (self.causal_discoverer.causality_mtx > 0.5).float()
            graph = graph[None].expand(x.size(0), -1, -1)
            graph_output = self.causal_discoverer.forward(
                x=x_positive[:, :self.noise_injection_step],
                fwd_graph=graph,
            ).transpose(1, 2)
        
        x_positive[:, -self.prediction_step, effects] \
        = graph_output[:, -self.prediction_step, effects]

        return x_positive
    

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
        )
        self.latent_sigma_pert = nn.Parameter(
            latent_perturbator_factor * torch.ones(window_size, data_dim*4)
        )
        self.decoder = TemporalConvNet(
            in_channels=data_dim*4,
            hidden_channels=[data_dim*4, data_dim*2, data_dim*2],
        )
        
        return
    
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        encoder_out = self.encoder.forward(x)
        mu = encoder_out[..., :self.data_dim*4]
        logvar = encoder_out[..., self.data_dim*4:]
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
        epoch: int = 100,
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


class TCNPerturbator_(nn.Module):
    def __init__(
        self,
        window_size: int,
        data_dim: int,
        latent_dim: int = 100,
        sigma_pert_factor: float = 2.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.encoder = TemporalConvNet(
            in_channels=data_dim,
            hidden_channels=[data_dim, 2*data_dim, 2*latent_dim],
        )
        self.decoder = TemporalConvNet(
            in_channels=latent_dim,
            hidden_channels=[latent_dim, 2*data_dim, 2*data_dim],
        )
        self.latent_sigma_pert = nn.Parameter(
            data=sigma_pert_factor * torch.ones(window_size, latent_dim),
            requires_grad=True,
        )
        return
    
    def encode(self, x: Tensor) -> Tensor:
        encoder_out = self.encoder.forward(x)
        mu = encoder_out[..., :self.latent_dim]
        logvar = encoder_out[..., self.latent_dim:]
        return mu, logvar
    
    def reparametrize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        eps = torch.randn_like(mu).to(mu.device)
        sigma = torch.exp(0.5 * logvar)
        z = mu + eps * sigma
        z_pert = mu + self.latent_sigma_pert * eps * sigma
        return z, z_pert
    
    def decode(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        decoder_out = self.decoder.forward(x=z)
        mult_factor: Tensor = decoder_out[..., :self.data_dim]
        add_factor: Tensor = decoder_out[..., self.data_dim:]
        return mult_factor, add_factor
    
    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x=x)
        z, z_pert = self.reparametrize(mu=mu, logvar=logvar)
        recon_mult, recon_add = self.decode(z=z)
        pert_mult, pert_add = self.decode(z=z_pert)
        x_recon = recon_mult * x + recon_add
        x_pert = pert_mult * x + pert_add
        return x_recon, x_pert, mu, logvar
    
    def init_perturbator(
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
        self.load_state_dict(ckpt['perturbator'])
        
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
            nn.Sigmoid()
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
    

class TCNDiscriminator_(nn.Module):
    def __init__(self, data_dim: int) -> None:
        super().__init__()
        self.tcn = TemporalConvNet(
            in_channels=data_dim,
            hidden_channels=[data_dim, 2*data_dim, 2*data_dim],
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features=2*data_dim, out_features=data_dim//2),
            nn.ReLU(),
            nn.Linear(in_features=data_dim//2, out_features=1),
        )
        return
    
    def _init_weights(self) -> None:
        for param in self.parameters():
            if isinstance(param, nn.Linear):
                nn.init.xavier_normal_(param.weight)
                nn.init.zeros_(param.bias)
        return
    
    def forward(self, x: Tensor) -> Tensor:
        tcn_out = self.tcn.forward(x)
        tcn_out_pool = tcn_out.mean(dim=1)
        label = self.mlp.forward(tcn_out_pool)
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


class BidirectionalPerturbator(nn.Module):
    def __init__(
        self,
        window_size: int,
        data_dim: int,
        sigma_pert_factor: float,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.data_dim = data_dim

        # self.temporal_encoder = TemporalConvNet(
        #     in_channels=data_dim,
        #     hidden_channels=[data_dim, 2*data_dim, 4*data_dim, 8*data_dim],
        # )
        # self.temporal_decoder = TemporalConvNet(
        #     in_channels=4*data_dim,
        #     hidden_channels=[4*data_dim, 2*data_dim, 2*data_dim],
        # )
        self.feature_encoder = nn.Sequential(
            weight_norm(nn.Linear(in_features=data_dim, out_features=data_dim)),
            nn.LeakyReLU(),
            weight_norm(nn.Linear(in_features=data_dim, out_features=data_dim)),
            nn.LeakyReLU(),
            weight_norm(nn.Linear(in_features=data_dim, out_features=data_dim)),
        )
        self.feature_mu = nn.Linear(
            in_features=data_dim,
            out_features=data_dim,
        )
        self.feature_logvar = nn.Linear(
            in_features=data_dim,
            out_features=data_dim,
        )
        self.feature_decoder = nn.Sequential(
            weight_norm(nn.Linear(in_features=data_dim, out_features=data_dim)),
            nn.LeakyReLU(),
            weight_norm(nn.Linear(in_features=data_dim, out_features=data_dim)),
            nn.LeakyReLU(),
            weight_norm(nn.Linear(in_features=data_dim, out_features=2*data_dim)),
        )
        # self.temporal_sigma_pert = nn.Parameter(
        #     data=sigma_pert_factor * torch.ones(window_size, 4*data_dim),
        #     requires_grad=True,
        # )
        self.feature_sigma_pert = nn.Parameter(
            data=sigma_pert_factor * torch.ones(window_size, data_dim),
            requires_grad=True,
        )
        return
    
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # temporal_encoer_out = self.temporal_encoder.forward(x)
        # temporal_mu = temporal_encoer_out[..., :4*self.data_dim]
        # temporal_logvar = temporal_encoer_out[..., 4*self.data_dim:]

        feature_encoder_out = self.feature_encoder.forward(x)
        feature_mu = self.feature_mu.forward(feature_encoder_out)
        feature_logvar = self.feature_logvar.forward(feature_encoder_out)

        # return temporal_mu, temporal_logvar, feature_mu, feature_logvar
        return feature_mu, feature_logvar
    
    # def temporal_reparametrize(
    #     self,
    #     temporal_mu: Tensor,
    #     temporal_logvar: Tensor,
    # ) -> Tuple[Tensor, Tensor]:
    #     temporal_eps = torch.randn_like(temporal_mu).to(temporal_mu.device)
    #     temporal_sigma = torch.exp(0.5 * temporal_logvar)

    #     temporal_z_recon = temporal_mu + temporal_eps * temporal_sigma
    #     temporal_z_pert \
    #     = temporal_mu + self.temporal_sigma_pert * temporal_eps * temporal_sigma
        
    #     return temporal_z_recon, temporal_z_pert
    
    def feature_reparametrize(
        self,
        feature_mu: Tensor,
        feature_logvar: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        feature_eps = torch.randn_like(feature_mu).to(feature_mu.device)
        feature_sigma = torch.exp(0.5 * feature_logvar)

        feature_z_recon = feature_mu + feature_eps * feature_sigma
        feature_z_pert \
        = feature_mu + self.feature_sigma_pert * feature_eps * feature_sigma

        return feature_z_recon, feature_z_pert
    
    def decode(
        self,
        # temporal_z: Tensor,
        feature_z: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # temporal_decoder_out = self.temporal_decoder.forward(temporal_z)
        feature_decoder_out = self.feature_decoder.forward(feature_z)

        # temporal_mult: Tensor = temporal_decoder_out[..., :self.data_dim]
        # temporal_add: Tensor = temporal_decoder_out[..., self.data_dim:]

        feature_mult: Tensor = feature_decoder_out[..., :self.data_dim]
        feature_add: Tensor = feature_decoder_out[..., self.data_dim:]
        
        # return temporal_mult, temporal_add, feature_mult, feature_add
        return feature_mult, feature_add
    
    def forward(
        self,
        x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Encode
        # temporal_mu, temporal_logvar, feature_mu, feature_logvar = self.encode(x)
        feature_mu, feature_logvar = self.encode(x)


        # Reparametrize
        # temporal_z_recon, temporal_z_pert, = self.temporal_reparametrize(
        #     temporal_mu=temporal_mu,
        #     temporal_logvar=temporal_logvar,
        # )
        feature_z_recon, feature_z_pert = self.feature_reparametrize(
            feature_mu=feature_mu,
            feature_logvar=feature_logvar,
        )
        
        # Decode
        # temporal_mult_recon, temporal_add_recon, \
        feature_mult_recon, feature_add_recon = self.decode(
            # temporal_z=temporal_z_recon,
            feature_z=feature_z_recon,
        )
        # temporal_mult_pert, temporal_add_pert, \
        feature_mult_pert, feature_add_pert = self.decode(
            # temporal_z=temporal_z_pert,
            feature_z=feature_z_pert,
        )

        # Reconstruction factors
        # recon_mult_factor = (temporal_mult_recon + feature_mult_recon) / 2
        # recon_add_factor = (temporal_add_recon + feature_add_recon) / 2
        recon_mult_factor = feature_mult_recon
        recon_add_factor = feature_add_recon
        # Perturbation factors
        # pert_mult_factor = (temporal_mult_pert + feature_mult_pert) / 2
        # pert_add_factor = (temporal_add_pert + feature_add_pert) / 2
        pert_mult_factor = feature_mult_pert
        pert_add_factor = feature_add_pert

        # Reconstruction and perturbation
        x_recon = recon_mult_factor * x + recon_add_factor
        x_pert = pert_mult_factor * x + pert_add_factor

        # return x_recon, x_pert, \
        #        temporal_mu, temporal_logvar, \
        #        feature_mu, feature_logvar
        return x_recon, x_pert, \
               feature_mu, feature_logvar


class Discriminator(nn.Module):
    def __init__(
        self,
        data_dim:int,
        conv_hidden_dim: int,
        mlp_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            weight_norm(
                module=nn.Conv1d(
                    in_channels=data_dim,
                    out_channels=conv_hidden_dim,
                    kernel_size=3,
                    padding=1,
                )
            ),
            nn.ReLU(),
            weight_norm(
                module=nn.Conv1d(
                    in_channels=conv_hidden_dim,
                    out_channels=conv_hidden_dim,
                    kernel_size=3,
                    padding=1,
                )
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            weight_norm(
                module=nn.Conv1d(
                    in_channels=conv_hidden_dim,
                    out_channels=data_dim,
                    kernel_size=1,
                )
            ),
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features=data_dim, out_features=mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=mlp_hidden_dim, out_features=1),
            nn.Sigmoid(),
        )
        self._init_weights()
        return
    
    def _init_weights(self) -> None:
        for param in self.parameters():
            if isinstance(param, nn.Linear):
                nn.init.xavier_normal_(param.weight)
                nn.init.zeros_(param.bias)
            elif isinstance(param, nn.Conv1d):
                nn.init.kaiming_normal_(param.weight)
                nn.init.zeros_(param.bias)
        return
    
    def forward(self, x: Tensor) -> Tensor:
        conv_in = x.transpose(-2, -1)
        conv_out: Tensor = self.conv.forward(conv_in)
        mlp_in = conv_out.transpose(-2, -1)
        mlp_out: Tensor = self.mlp.forward(mlp_in)
        label = mlp_out.squeeze(-1) 
        return label


class Augmentor(nn.Module):
    def __init__(
        self,
        cuts_plus_time: str,
        perturbator_time: str,
        perturbator_mode: str,
        data: str,
        subdata: Optional[str],
        window_size: int,
        data_dim: int,
    ) -> None:
        super().__init__()
        assert perturbator_mode in ['linear', 'tcn', 'biirectional'], \
        "'linear', 'tcn', 'bidirectional'"

        self.pred_step = 1
        self.noise_injection_step = window_size - self.pred_step
        
        self.causal_discoverer = CUTS_Plus_Net(n_nodes=data_dim)
        self.causal_discoverer.init_cuts_plus(
            time=cuts_plus_time,
            data=data,
            subdata=subdata,
        )
        self.causal_discoverer.eval()

        # if perturbator_mode == 'linear':
        #     self.perturbator = LinearPerturbator(
        #         window_size=window_size,
        #         data_dim=data_dim,
        #         latent_perturbator_factor=2.0
        #     )
        if perturbator_mode == 'tcn':
            self.perturbator = TCNPerturbator(
                window_size=window_size,
                data_dim=data_dim,
                latent_perturbator_factor=2.0
            )
        elif perturbator_mode == 'bidirectional':
            self.perturbator = BidirectionalPerturbator(
                window_size=window_size,
                data_dim=data_dim,
                sigma_pert_factor=2.0,
            )

        self.perturbator.init_perturbator(
            time=perturbator_time,
            data=data,
            subdata=subdata,
        )
        self.perturbator.eval()

        self.discriminator = TCNDiscriminator(
            window_size=window_size,
            data_dim=data_dim,
        )
        self.discriminator.init_discriminator(
            time=perturbator_time,
            data=data,
            subdata=subdata,
        )
        self.discriminator.eval()

        return
    
    def make_positive_pair(
        self,
        anchor: Tensor,
        noise_level: float = 0.1,
    ) -> Tensor:
        positive_pair = anchor.clone()
        causality_matrix = self.causal_discoverer.causality_mtx
        node_selector = (causality_matrix > 0.5).int()

        start_node = random.choice(range(len(node_selector)))
        self.start_node = start_node

        effects = [
            i for i, val in enumerate(node_selector[start_node]) if val == 1
        ]
        self.effects = effects

        noise = torch.randn(anchor.size(0)).to(anchor.device) * noise_level

        positive_pair[:, self.noise_injection_step, start_node] += noise

        with torch.no_grad():
            fwd_graph = (causality_matrix > 0.5).float()
            fwd_graph = fwd_graph[None].expand(anchor.size(0), -1, -1)
            causal_discoverer_out = self.causal_discoverer.forward(
                x=positive_pair[:, :self.noise_injection_step],
                fwd_graph=fwd_graph,
            )
            causal_discoverer_out = causal_discoverer_out.transpose(1, 2)

        positive_pair[:, -self.pred_step, effects] \
        = causal_discoverer_out[:, -self.pred_step, effects]

        return positive_pair
    
    def make_negative_pair(
        self,
        anchor: Tensor,
        noise_level: float = 0.1,
    ) -> Tensor:
        _, negative_pair, _, _, _, _, _, _ \
        = self.perturbator.forward(x=anchor)
        
        negative_pair = self.make_positive_pair(
            x=negative_pair,
            noise_level=noise_level,
        )

        return negative_pair
    
    def forward(self, anchor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        positive_pair = self.make_positive_pair(anchor=anchor)
        negative_pair = self.make_negative_pair(anchor=anchor)
        return anchor, positive_pair, negative_pair
    