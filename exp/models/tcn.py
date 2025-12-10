import warnings
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from exp.utils.common_import import *
warnings.filterwarnings('ignore')


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: Tensor) -> Tensor:
        return x[:, :, :-self.chomp_size].contiguous()
    

class TemporalBlock(nn.Module):
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

        return

    def _init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv1.bias.data.zero_()
        self.conv2.weight.data.normal_(0, 0.01)
        self.conv2.bias.data.zero_()
        self.conv3.weight.data.normal_(0, 0.01)
        self.conv3.bias.data.zero_()

        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)
            self.downsample.bias.data.zero_()

        return
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        residual = self.downsample(x) if self.downsample else x
        return self.activation(out + residual)


class TemporalConvNet(nn.Module):
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

        return

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.network.forward(x.transpose(-2, -1))
        out = out.transpose(-2, -1)
        return out
 

class DeconvTemporalBlock(nn.Module):
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

        self.deconv = weight_norm(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                output_padding=0,
            )
        )
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.conv1 = weight_norm(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                output_padding=0,
            )
        )
        self.dropout1 = nn.Dropout(p=dropout)

        self.conv2 = weight_norm(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                output_padding=0,
            )
        )
        self.dropout2 = nn.Dropout(p=dropout)

        self.net = nn.Sequential(
            self.deconv,
            self.dropout1,
            self.activation,
            self.conv1,
            self.dropout2,
            self.activation,
            self.conv2,
            self.dropout2,
            self.activation,
        )

        self.downsample = None
        
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )
        
        self._init_weights()
        
        return

    def _init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

        return
    
    def match_length(self, x: Tensor, net_out: Tensor) -> Tensor:
        x_len = x.size(-1)
        net_out_len = net_out.size(-1)

        if net_out_len > x_len:
            net_out = net_out[..., :x_len]
        elif net_out_len < x_len:
            pad = x_len - net_out_len
            net_out = F.pad(input=net_out, pad=(0, pad))
        
        return net_out
    
    def forward(self, x: Tensor) -> Tensor:
        net_out = self.net.forward(x)
        net_out = self.match_length(x=x, net_out=net_out)
        residual = self.downsample(x) if self.downsample is not None else x
        
        return self.activation(net_out + residual)


class TemporalDeconvNet(nn.Module):
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
                DeconvTemporalBlock(
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

        return
    
    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.network.forward(x.transpose(-2, -1))
        out = out.transpose(-2, -1)
        return out
