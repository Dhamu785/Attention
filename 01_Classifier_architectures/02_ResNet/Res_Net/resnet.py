from .blocks import *

import torch as t
import torch.nn as nn

from typing import Callable, Optional, Union, List

class ResNet(nn.Module):
    def __init__(self, block: type[Union[BasicBlock]], layers: List[int],
                    num_class: int=1000, zero_init_residual: bool=False, 
                    norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_planes, kernel_size=(7,7),
                                stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(512, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: type[Union[BasicBlock]], planes: int, blocks: int, stride: int=1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                conv1x1(in_planes=self.in_planes, out_planes=planes, strides=stride),
                norm_layer(planes)
            )
        layers = list()
        layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride, downsample=downsample, norm_layer=norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_planes=self.in_planes, planes=planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def _forward(self, x: t.Tensor) -> t.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = t.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        return self._forward(x=x)