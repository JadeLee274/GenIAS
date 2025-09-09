from utils.common_import import *
from .resnet import ResNet
RESNET_PATH = '../checkpoints/resnet'


class PretextModel(nn.Module):
    """
    Model for Pretext stage of CARLA.
    
    Bring the positive pair closer to the anchor, and keeps the negative pair
    away from the anchor.

    Parameters:
        in_channels:        Dimension of the data.
        mid_channels:       out_channels of the hidden convolutional layers.
                            Default 4.
        head:               Type of head. Default 'mlp'.
        representation_dim: Dimension of representation space. Default 128.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 4,
        head: str = 'mlp',
        representation_dim: int = 128,
    ) -> None:
        super().__init__()
        self.resnet = ResNet(
            in_channels=in_channels,
            mid_channels=mid_channels,
        )
        self.feature_dim = 2 * mid_channels
        self.head = head

        assert head in ['linear', 'mlp'], \
        "head must be either 'linear' or 'mlp'"

        if head == 'linear':
            self.contrastive_head = nn.Linear(
                in_features=self.feature_dim,
                out_features=representation_dim,
            )
        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                nn.Linear(
                    in_features=self.feature_dim,
                    out_features=self.feature_dim,
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=self.feature_dim,
                    out_features=representation_dim,
                )
            )

        return
    
    def forward(self, x: Tensor) -> Tensor:
        z = self.resnet.forward(x)
        z = self.contrastive_head(z)
        z = F.normalize(input=z, dim=1)
        return z
    
    def _init_weights(self) -> None:
        for param in self.parameters():
            if isinstance(param, nn.Linear):
                nn.init.xavier_normal_(param.weight)
                nn.init.zeros_(param.bias)
            elif isinstance(param, nn.Conv1d):
                nn.init.kaiming_normal_(param.weight)
                nn.init.zeros_(param.bias)

        return
    

class ClassificationModel(nn.Module):
    """
    Model for Self-supervised classification stage of CARLA.

    Parameters:
        in_channels:  Dimension of data.
        mid_channels: Mid-channel of the resnet structure.
        dataset:      Dataset name.
        num_classes:  Number of classes to which the input will be classified.
                      Default 10.


    This model consists of ResNet (which is pre-trained at the pretext stage)
    and the cluster head, sending the anchor and nearest neighbors to 
    (num_classes)-dimensional space.

    With the classificationloss at utils.carlalss.py, the model maximizes the
    similarity between the representations of window and nearest neighborhood,
    while minimizing the similarity between those of window and furthest
    neighborhood.

    By doing so, the inputs of this model is classified to one of the 
    (num_classes) classes. If trained well, the model sends the majority of
    normal data to a particular class, which is called C_m - th class.

    At the inference stage, if the input of test set is mapped to C_m - th
    class, it is inferred as normal; abnormal otherwise.
    """
    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 4,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.resnet = ResNet(
            in_channels=in_channels,
            mid_channels=mid_channels,
        )
        self.feature_dim = 2 * mid_channels

        self.classification_head = nn.Linear(
            in_features=self.feature_dim,
            out_features=num_classes,
            )
        
        nn.init.xavier_normal_(self.classification_head.weight)
        nn.init.zeros_(self.classification_head.bias)

        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        x: Tensor,
        forward_pass: str = 'default',
    ) -> Union[List[Tensor], Tensor, Dict[str, Union[Tensor, List[Tensor]]]]:
        if forward_pass == 'default':
            feature = self.resnet(x)
            out = self.classification_head(feature)
            out = self.softmax(out)

        elif forward_pass == 'backbone':
            feature = self.resnet(x)
            out = self.softmax(feature)

        elif forward_pass == 'head':
            out = self.classification_head(x)
            out = self.softmax(out)

        elif forward_pass == 'return_all':
            feature = self.resnet(x)
            out = {
                'feature': feature,
                'output': self.softmax(self.classification_head(x)),
            }

        else:
            raise ValueError(f'Invalid forward pass type {forward_pass}')
        
        return out
