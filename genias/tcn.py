from typing import *
import torch.nn as nn
from torch.nn.utils import weight_norm
from utils.common_import import *
"""
This code follows the following paper:
An Empirical Evaluation of Generic Convolutional and Recurrent Networks 
for Sequence Modeling, Bai et al., 2018

Paper link:  https://arxiv.org/pdf/1803.01271
Github link: https://github.com/locuslab/TCN
"""


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
    