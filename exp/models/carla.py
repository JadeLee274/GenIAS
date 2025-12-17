import warnings
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from exp.utils.common_import import *
warnings.filterwarnings('ignore')


def conv1d_same_padding(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> Tensor:
    kernel = weight.size(2)
    dilation = dilation[0]
    stride = stride[0]

    # This is to assert that the output dimension matches the input dimension.
    input_dim = input.size(2)
    output_dim = input.size(2)  
    
    padding = (
        ((output_dim - 1) * stride) - input_dim + (dilation * (kernel - 1)) + 1
    )

    if padding % 2 != 0:
        input = F.pad(input=input, pad=[0, 1])

    return F.conv1d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding//2,
        dilation=dilation,
        groups=groups,
    )


class Conv1dSamePadding(nn.Conv1d):
    def forward(self, x: Tensor) -> Tensor:
        return conv1d_same_padding(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
        )


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm: str,
    ) -> None:
        super().__init__()
        assert norm in ['batch', 'instance', 'weight'], \
        "'batch', 'instance', 'weight'"

        # Convolution kernel is applied to time axis.      
        if norm == 'batch':
            self.layers = nn.Sequential(
                Conv1dSamePadding(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU(),
            )
        elif norm == 'instance':
            self.layers = nn.Sequential(
                Conv1dSamePadding(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                nn.InstanceNorm1d(num_features=out_channels),
                nn.ReLU(),
            )
        elif norm == 'weight':
            self.layers = nn.Sequential(
                weight_norm(
                    Conv1dSamePadding(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                    )
                ),
                nn.ReLU(),
            )
        
        return
    
    def forward(self, x: Tensor) -> Tensor:
        return self.layers.forward(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str,
        kernel_size_list: List[int] = [8, 5, 3]
    ) -> None:
        super().__init__()
        block_depth = len(kernel_size_list)
        channels = [in_channels] + block_depth * [out_channels]

        layers = []

        for i in range(block_depth):
            layers.append(
                ConvBlock(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    kernel_size=kernel_size_list[i],
                    stride=1,
                    norm=norm,
                )
            )
        
        self.layers = nn.Sequential(*layers)

        self.match_channels = False

        if in_channels != out_channels:
            self.match_channels = True
            if norm == 'batch':
                self.residual_layer = nn.Sequential(      
                    Conv1dSamePadding(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                    ),
                    nn.BatchNorm1d(num_features=out_channels),
                )
            elif norm == 'instance':
                self.residual_layer = nn.Sequential(      
                    Conv1dSamePadding(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                    ),
                    nn.InstanceNorm1d(num_features=out_channels),
                )
            elif norm == 'weight':
                self.residual_layer = weight_norm(
                    Conv1dSamePadding(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                    )
                )
    
    def forward(self, x: Tensor) -> Tensor:
        if self.match_channels:
            return self.layers(x) + self.residual_layer(x)
        return self.layers(x)
    

class ResNet(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 4) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            ResidualBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
            ),
            ResidualBlock(
                in_channels=mid_channels,
                out_channels=2*mid_channels,
            ),
            ResidualBlock(
                in_channels=2*mid_channels,
                out_channels=2*mid_channels,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        z: Tensor = self.layers.forward(x)
        z = z.mean(dim=-1)
        return z


class PretextModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 4,
        representation_dim: int = 128,
    ) -> None:
        super().__init__()
        self.resnet = ResNet(
            in_channels=in_channels,
            mid_channels=mid_channels,
        )
        self.feature_dim = 2 * mid_channels
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
        
        nn.init.xavier_uniform_(self.classification_head.weight)
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
