from cuts_plus.utils.imports import *
from cuts_plus.models.encoder import LSTMEncoder, GRUEncoder
from cuts_plus.models.cuts_plus import CUTS_Plus_Net
from cuts_plus.models.augmentor import PositiveAugmentor, NegativeAugmentor


class CAROTS(nn.Module):
    def __init__(
        self,
        data_dim: int,
        encoder: str,
        encoder_hidden_size: int = 512,
        encoder_num_layers: int = 1,
        encoder_dropout: float = 0.0,
        encoder_batch_first: bool = True,
        projector_input_dim: int = 512,
        projector_hidden_dim: int = 1024,
        projector_output_dim: int = 512,
        positive_augmentor_input_step: int = 9,
        positive_augmentor_pred_step: int = 1,
        positive_augmentor_noise_level: float = 0.1,
        negative_augmentor_cutoff_probability: float = 0.1,
    ) -> None:
        super().__init__()
        if encoder == 'lstm':
            self.encoder = LSTMEncoder(
                num_vars=data_dim,
                hidden_size=encoder_hidden_size,
                num_layers=encoder_num_layers,
                dropout=encoder_dropout,
                batch_first=encoder_batch_first
            )
        elif encoder == 'gru':
            self.encoder = GRUEncoder(
                num_vars=data_dim,
                hidden_size=encoder_hidden_size,
                num_layers=encoder_num_layers,
                dropout=encoder_dropout,
                batch_first=encoder_batch_first
            )
        
        self.projector = nn.Sequential(
            nn.Linear(
                in_features=projector_input_dim,
                out_features=projector_hidden_dim,
            ),
            nn.BatchNorm1d(num_features=projector_hidden_dim),
            nn.GELU(),
            nn.Linear(
                in_features=projector_hidden_dim,
                out_features=projector_output_dim,
            ),
        )

        self.causal_discoverer = CUTS_Plus_Net(
            n_nodes=data_dim,
            data_dim=data_dim,
        )

        self.positive_augmentor = PositiveAugmentor(
            n_nodes=data_dim,
            data_dim=data_dim,
            input_step=positive_augmentor_input_step,
            pred_step=positive_augmentor_pred_step,
            noise_level=positive_augmentor_noise_level,
        )

        self.negative_augmentor = NegativeAugmentor(
            cutoff_probability=negative_augmentor_cutoff_probability,
        )

        return

    def forward(self, x: Tensor) -> Tensor:
        x_all = x.clone()
        
        positive_samples = self.positive_augmentor.forward(
            x=x_all,
            causality_mtx=self.causal_discoverer.causality_mtx,
        )
        x_all = torch.concat([x_all, positive_samples], dim=0)
        
        negative_samples = self.negative_augmentor.forward(
            x=x_all,
            causality_mtx=self.causal_discoverer.causality_mtx,
        )
        x_all = torch.concat([x_all, negative_samples], dim=0)

        enc_out = self.encoder.forward(x_all)
        
        # Project the encoded output
        out = self.projector.forward(enc_out)

        return out
