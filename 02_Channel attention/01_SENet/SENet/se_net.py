import torch as t
from torch import nn
import torch.nn.init as init

from SENet.blocks import conv3x3_block, conv1x1_block
from typing import Optional, Union, Callable
from SENet.utils import get_activation

import math
import numpy as np

from datetime import datetime
import os

class init_block(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        mid_channel = out_channel // 2

        self.conv1 = conv3x3_block(in_channels=in_channel, out_channels=mid_channel, stride=2)
        self.conv2 = conv3x3_block(in_channels=mid_channel, out_channels=mid_channel)
        self.conv3 = conv3x3_block(in_channels=mid_channel, out_channels=out_channel)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x) -> t.Tensor:
        return self.pool(self.conv3(self.conv2(self.conv1(x))))

class bottleneck(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, stride: int, cardinality: int, bottleneck_width: int) -> None:
        super().__init__()
        mid_channel = out_channel // 2
        D = int(math.floor(mid_channel * (bottleneck_width / 64.0)))
        group_width = cardinality * D
        group_width2 = group_width // 2

        self.conv1 = conv1x1_block(in_channels=in_channel, out_channels=group_width2)
        self.conv2 = conv3x3_block(in_channels=group_width2, out_channels=group_width, stride=stride, groups=cardinality)
        self.conv3 = conv1x1_block(in_channels=group_width, out_channels=out_channel)
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.conv3(self.conv2(self.conv1(x)))

class SEblock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, approx_sigmoid: bool = False, 
                    activation: Optional[Union[Callable[..., t.Tensor], str]] = (lambda: nn.ReLU(inplace=True))) -> None:
        super().__init__()
        mid_channels = channels // reduction

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=1, stride=1, groups=1, bias=False)
        self.activ = get_activation(activation)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=channels, kernel_size=1, stride=1, groups=1, bias=False)
        self.sigmoid = nn.Sigmoid() if approx_sigmoid else nn.Sigmoid()
        self.sav = 0

    def forward(self, x: t.Tensor) -> t.Tensor:
        X = x * self.sigmoid(self.conv2(self.activ(self.conv1(self.pool(x)))))
        if not self.training and len(x) == 6:
            timestamp = datetime.now().strftime("%H%M%S")
            self.sav+=1
            np.save(f'./seIO/Inp_{timestamp}_{self.sav}.npy', x.detach().cpu().numpy())
            np.save(f'./seIO/Outp_{timestamp}_{self.sav}.npy', X.detach().cpu().numpy())
        return X

class SENetUnit(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, stride: int, bottleneck_width: int, identity_conv3x3: bool, cardinality: int) -> None:
        super().__init__()
        self.resize_identity = (in_channel != out_channel) or (stride != 1)

        self.body = bottleneck(in_channel=in_channel, out_channel=out_channel, stride=stride, cardinality=cardinality, bottleneck_width=bottleneck_width)
        self.se = SEblock(channels=out_channel)

        if self.resize_identity:
            if identity_conv3x3:
                self.identity_conv = conv3x3_block(in_channels=in_channel, out_channels=out_channel, stride=stride)
            else:
                self.identity_conv = conv1x1_block(in_channels=in_channel, out_channels=out_channel, stride=stride)
        
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x: t.Tensor) -> t.Tensor:
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        return self.activ(identity + self.se(self.body(x)))


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
                stages.add_module(f'Unit-{j+1}', SENetUnit(in_channel=in_channels, out_channel=out_channels, stride=stride,
                                                            bottleneck_width=bottleneck_width, identity_conv3x3=identy_conv3x3,
                                                            cardinality=cardinality))
                in_channels = out_channels
            self.features.add_module(f'Stage-{i+1}', stages)
        self.features.add_module('final_pool', nn.AvgPool2d(kernel_size=7, padding=1))
            
        self.output = nn.Sequential()
        self.output.add_module('dropout', nn.Dropout(p=0.2))
        self.output.add_module('classifier head', nn.Linear(in_features=in_channels, out_features=num_classes))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.features(x)
        return self.output(x.view(x.size(0), -1))