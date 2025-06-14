import torch as t
from torch import nn
from blocks import conv3x3_block, conv1x1_block
from typing import Optional, Union, Callable
from utils import get_activation

import math

class init_block(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        mid_channel = out_channel // 2

        self.conv1 = conv3x3_block(in_channels=in_channel, out_channels=mid_channel, stride=2)
        self.conv2 = conv3x3_block(in_channels=mid_channel, out_channels=mid_channel)
        self.conv3 = conv3x3_block(in_channels=mid_channel, out_channels=out_channel)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x) -> t.Tensor:
        x = self.conv1(self.conv2(self.conv3(self.pool(x))))
        return x

class bottleneck(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, stride: int, cardinality: int, bottleneck_width: int) -> None:
        super().__init__()
        mid_channel = out_channel // 2
        D = int(math.floor(mid_channel * (bottleneck_width / 64.0)))
        group_width = cardinality * D
        group_width2 = group_width // 2

        self.conv1 = conv1x1_block(in_channel=in_channel, out_channels=group_width2)
        self.conv2 = conv3x3_block(in_channels=group_width2, out_channels=group_width, stride=stride, groups=cardinality)
        self.conv3 = conv1x1_block(in_chennels=group_width, out_channels=out_channel)
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.conv1(self.conv2(self.conv3(x)))
        return x

class SEblock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, approx_sigmoid: bool = False, 
                    activation: Optional[Union[Callable[..., t.Tensor], str]] = (lambda: nn.ReLU(inplace=True))) -> None:
        super().__init__()
        mid_channels = channels // reduction

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = conv1x1_block(in_chennels=channels, out_channels=mid_channels, bias=True)
        self.activ = get_activation(activation)
        self.conv2 = conv1x1_block(in_chennels=mid_channels, out_channels=channels, bias=True)
        self.sigmoid = nn.Sigmoid() if approx_sigmoid else nn.Sigmoid()

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.sigmoid(self.conv2(self.activ(self.conv1(self.pool(x)))))

class SENetUnit(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, stride: int, bottleneck_width: int, identity_conv3x3: bool, cardinality: int) -> None:
        super().__init__()
        self.resize_identity = (in_channel != out_channel) or (stride != 1)

        self.body = bottleneck(in_channel=in_channel, out_channel=out_channel, stride=stride, cardinality=cardinality, bottleneck_width=bottleneck_width)
        
class SE_Net(nn.Module):
    def __init__(self, channels: list[list], init_block_channels: int, cardinality: int, bottleneck_width: int, in_channels: int = 3, in_size = (224, 224), num_classes=100) -> None:
        super().__init__()
        self.in_size = in_size
        self.num_class = num_classes

        self.features = nn.Sequential()
        self.features.add_module('init_block', init_block(in_channel=in_channels, out_channel=init_block_channels))
        in_channels = init_block_channels

        for i, channels_per_stage in enumerate(channels):
            stages = nn.Sequential()
            identy_conv3x3 = (i != 0)
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stages.add_module()