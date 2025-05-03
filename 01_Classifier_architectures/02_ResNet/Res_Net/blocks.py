import torch as t
import torch.nn as nn
from typing import Callable, Optional

__all__ = ['conv3x3', 'conv1x1', 'BasicBlock']

def conv3x3(in_planes: int, out_planes: int, strides: int=1, dilation: int=1) -> nn.Module:
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=(3,3), stride=strides,
                        padding=dilation, bias=False)

def conv1x1(in_planes: int, out_planes: int, strides: int=1) -> nn.Module:
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=(1,1), stride=strides, bias=False)

class BasicBlock(nn.Module):
    expansion: int=1

    def __init__(self, in_planes: int, planes: int, stride: int=1, downsample: Optional[Callable[..., nn.Module]]=None, norm_layer: Optional[Callable[..., nn.Module]]=None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(in_planes=in_planes, out_planes=planes, strides=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_planes=planes, out_planes=planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identy = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.relu(out)

        if self.downsample is not None:
            identy = self.downsample(identy)
        
        out += identy
        out = self.relu(out)

        return out