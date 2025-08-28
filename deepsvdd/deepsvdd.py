from utils.common_import import *


class SVDD(nn.Module):
    """
    Deep SVDD (Suppport Vector Data Discription) model.
    Based on Deep One-Class Classification, L. Ruff et al., 2018, ICML.

    Paper link: https://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf
    
    Github link: https://github.com/lukasruff/Deep-SVDD-PyTorch
    
    This model is for ARP and EDI metric.

    Parameters:
        data_dim:           Dimension of dataset.
        window_size:        Length of the input window.
        depth:              Depth of Deep SVDD model.
        representation_dim: Dimension of representation space. Default 128.
    """
    def __init__(
        self,
        data_dim: int,
        window_size: int,
        depth: int,
        representation_dim: int = 128,
    ) -> None:
        super().__init__()
        network = []
        hidden_dims = [
            int(data_dim + i * (representation_dim - data_dim) / (depth)) \
                for i in range(depth + 1)
            ]
        for i in range(1, depth):
            network.append(
                nn.Linear(
                    in_features=hidden_dims[i - 1],
                    out_features=hidden_dims[i]
                )
            )
            network.append(nn.BatchNorm1d(window_size))
            network.append(nn.ReLU())

        network.append(
            nn.Linear(
                in_features=hidden_dims[depth - 1],
                out_features=hidden_dims[depth],
            )
        )

        self.network = nn.Sequential(*network)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.parameters():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
