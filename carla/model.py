from utils.common_import import *
from .resnet import ResNet
RESNET_PATH = '../checkpoints/resnet'


class ContrastiveModel(nn.Module):
    """
    Contrastive 
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        head: str = 'mlp',
        representation_dim: int = 128,
    ) -> None:
        super().__init__()
        self.resnet = ResNet(
            in_channels=in_channels,
            mid_channels=mid_channels,
        )
        self.backbone_dim = 2 * mid_channels
        self.head = head

        assert head in ['linear', 'mlp'], \
        "head must be either 'linear' or 'mlp'"

        if head == 'linear':
            self.contrastive_head = nn.Linear(
                in_features=self.backbone_dim,
                out_features=representation_dim,
            )
        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                nn.Linear(
                    in_features=self.backbone_dim,
                    out_features=self.backbone_dim,
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=self.backbone_dim,
                    out_features=representation_dim,
                )
            )
    
    def forward(self, x: Tensor):
        z = self.resnet(x)
        z = self.contrastive_head(z)
        z = F.normalize(input=z, dim=1)
        return z
    

class ClusteringModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        num_clusters: int,
        num_heads: int,
        dataset: str,
    ) -> None:
        super().__init__()
        self.resnet = ResNet(
            in_channels=in_channels,
            mid_channels=mid_channels,
        )
        self.dataset = dataset
        self._initiate_resnet()
        self.backbone_dim = 2 * mid_channels

        assert isinstance(num_heads, int), "num_heads must be an integer"
        assert num_heads > 0, "num_heads must be positive"
        self.num_heads = num_heads

        self.cluster_head = nn.ModuleList(
            [
                nn.Linear(
                    in_features=self.backbone_dim,
                    out_features=num_clusters,
                ) for _ in range(self.num_heads)
            ]
        )
    
    def forward(
        self,
        x: Tensor,
        forward_pass: str = 'default',
    ) -> Union[List[Tensor], Tensor, Dict[str, Union[Tensor, List[Tensor]]]]:
        if forward_pass == 'default':
            representation = self.resnet(x)
            out = [head(representation) for head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.resnet(x)

        elif forward_pass == 'head':
            out = [head(x) for head in self.cluster_head]

        elif forward_pass == 'return_all':
            representation = self.resnet(x)
            out = {
                'representations': representation,
                'output': [head(x) for head in self.cluster_head],
            }

        else:
            raise ValueError(f'Invalid forward pass type {forward_pass}')
        
        return out

    def _initiate_resnet(self) -> None:
        resnet_path = os.path.join(RESNET_PATH, self.dataset, 'epoch_100.pt')
        ckpt = torch.load(resnet_path)
        self.resnet.load_state_dict(ckpt['model'])
        return None
