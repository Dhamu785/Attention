import torch as t
import torch.nn as nn
from typing import Tuple, Union, Callable, Optional

def conv1x1(in_planes: int, out_planes: int, strides: int=1, groups: int=1, dilation: int=1) -> nn.Module:
    return nn.conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=(3,3), stride=strides,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, strides: int=1) -> nn.Module:
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=(1,1), stride=strides)

class BasicBlock(nn.Module):
    expansion: int=1

    def __init__(self, in_planes: int, planes: int, stride: int=1, downsample: Optional[Callable[..., nn.Module]]=None, norm_layer: Optional[Callable[..., nn.Module]]=None) -> None:
        ...