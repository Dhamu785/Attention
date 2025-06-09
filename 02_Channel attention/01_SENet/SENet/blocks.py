import torch as t
from torch import nn
from typing import Optional, Callable, Union

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int, groups: int, 
                    bias: bool, bn_eps: float = 1e-5, activation: Union[Callable[..., t.Tensor], str] = (lambda: nn.ReLU(inplace=True))) -> None:
        super().__init__()

        self.conv = nn.Conv2d()


def conv3x3_block(in_chennels: int, out_chennels: int, stride: int, padding: int, dilation: int = 1, groups: int = 1, bias: bool = False,
                    bn_eps: float = 1e-5, activation: Union[Callable[..., t.Tensor], str]=(lambda: nn.ReLU(inplace=True))):

        self.conv = nn.Conv2d(in_channels=in_chennels, out_channels=out_chennels)