import torch as t
from torch import nn
from typing import Optional, Callable, Union

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int, groups: int, 
                    bias: bool, bn_eps: float = 1e-5, activation: Union[Callable[..., t.Tensor], str] = (lambda: nn.ReLU(inplace=True))) -> None:
        super().__init__()

        self.activation = activation is None
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        if self.activation:
            self.activ = activation

    def forward(self, x) -> t.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.activ(x)
        return x


def conv3x3_block(in_channels: int, out_channels: int, stride: int, padding: int, dilation: int = 1, groups: int = 1, bias: bool = False,
                    bn_eps: float = 1e-5, activation: Optional[Union[Callable[..., t.Tensor], str]] = (lambda: nn.ReLU(inplace=True))) -> nn.Module:

    return ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, groups=groups,
                            bias=bias, bn_eps=bn_eps, activation=activation)

def conv1x1_block(in_chennels: int, out_channels: int, stride: int = 1, padding: int = 1, groups: int = 1, 
                    bias: bool = False, bn_eps: float = 1e-5, activation: Optional[Union[Callable[..., t.Tensor], str]] = (lambda: nn.ReLU(inplace=True))) -> nn.Module:
    return ConvBlock(in_channels=in_chennels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding, groups=groups, bias=bias, bn_eps=bn_eps, activation=activation)