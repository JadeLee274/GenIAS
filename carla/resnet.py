from utils.common_import import *


def conv1d_same_padding(
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
) -> F.conv1d:
    """
    Function for the 'same' padding in TensorFlow. It is the convolution layer
    that ensures the input dimension and the output dimension are the same.

    Parameters:
        input:    Input tensor of convolution layers.
        weight:   Weight of the convolutional kernel.
        bias:     Bias of the convolutional kernel.
        stride:   Stride.
        dilation: Dilation.
        groups:   Groups.
    """
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
    ) -> None:
        super().__init__()
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
    
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_list: List[int] = [8, 5, 3]
    ) -> None:
        super().__init__()
        channels = [in_channels] + 3 * [out_channels]
        block_depth = len(kernel_size_list)

        layers = []

        for i in range(block_depth):
            layers.append(
                ConvBlock(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    kernel_size=kernel_size_list[i],
                    stride=1,
                )
            )
        
        self.layers = nn.Sequential(*layers)

        self.match_channels = False

        if in_channels != out_channels:
            self.match_channels = True
            self.residual_layer = nn.Sequential(
                *[
                    Conv1dSamePadding(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                    ),
                    nn.BatchNorm1d(num_features=out_channels),
                ]
            )
    
    def forward(self, x: Tensor) -> Tensor:
        if self.match_channels:
            return self.layers(x) + self.residual_layer(x)
        return self.layers(x)
    

class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int =  4,
    ) -> None:
        super().__init__()

        self.input_args = {
            'in_channels': in_channels
        }

        self.layers = nn.Sequential(
            *[
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=mid_channels
                ),
                ResidualBlock(
                    in_channels=mid_channels,
                    out_channels=2*mid_channels
                ),
                ResidualBlock(
                    in_channels=2*mid_channels,
                    out_channels=2*mid_channels,
                )
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.layers(x)
        z = z.mean(dim=-1)
        return z
