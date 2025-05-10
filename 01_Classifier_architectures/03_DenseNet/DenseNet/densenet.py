import torch as t
import torch.nn as nn
from torch import Tensor
import torch.utils.checkpoint as cp
import torch.nn.functional as F

from collections import OrderedDict

class _layers(nn.Module):
    def __init__(self, in_features: int, bn_size: int, growth_rate: int, drop_rate: float, memory_efficiency: bool) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=bn_size*growth_rate, kernel_size=1, bias=False)

        self.norm2 = nn.BatchNorm2d(num_features=bn_size*growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=bn_size*growth_rate, out_channels=growth_rate, kernel_size=3, bias=False, padding=1)

        self.drop_rate = drop_rate
        self.memory_efficiency = memory_efficiency

    def bottle_neck(self, inputs: list[Tensor]) -> Tensor:
        con_feat = t.cat(inputs, 1)
        btl_nk = self.conv1(self.relu1(self.norm1(con_feat)))
        return btl_nk
    
    def any_requires_grad(self, inputs: list[Tensor]) -> bool:
        for ins in inputs:
            if ins.requires_grad:
                return True
        return False
    
    def check_point(self, inputs: list[Tensor]) -> Tensor:
        def clouser(*feat_in):
            return self.bottle_neck(feat_in)
        return cp.checkpoint(clouser, *inputs, use_reentrant=False)
    
    def forward(self, inp_features: list[Tensor]) -> Tensor:
        if isinstance(inp_features, Tensor):
            previous_feature = [inp_features]
        else:
            previous_feature = inp_features

        if self.memory_efficiency and self.any_requires_grad(previous_feature):
            btl_nk_out = self.check_point(previous_feature)
        else:
            btl_nk_out = self.bottle_neck(previous_feature)

        out3x3 = self.conv2(self.relu2(self.norm2(btl_nk_out)))

        if self.drop_rate > 0:
            out3x3 = F.dropout(out3x3, self.drop_rate, training=self.training)

        return out3x3
    
class _block(nn.ModuleDict):
    def __init__(self, num_layers: int, input_feat: int, bn_size: int, growth_rate: int, memory_efficienct: bool, drop_rate: float) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _layers(input_feat+i*growth_rate, bn_size, growth_rate, drop_rate, memory_efficienct)
            self.add_module(f'denselayer-{i+1}', layer)

    def forward(self, input_features: Tensor) -> Tensor:
        features = [input_features]
        for name, layer in self.items():
            out_feat = layer(features)
            features.append(out_feat)
        return t.cat(features, 1)

class _transition(nn.Sequential):
    def __init__(self, input_features: int, out_features: int) -> Tensor:
        super().__init__()
        self.norm = nn.BatchNorm2d(input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_features, out_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

class denseNet(nn.Module):
    def __init__(self, block_config: tuple[int, int, int, int], num_classes: int, growth_rate: int = 32, bn_size: int = 4, drop_rate: float=0.2, 
                    initial_features: int = 64, memory_effiency: bool = True) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            OrderedDict([
                ('conv0', nn.Conv2d(3, initial_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(initial_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, padding=1, stride=2))
            ])
        )

        # Adding dense block to above layers
        num_features = initial_features
        for i, num_layers in enumerate(block_config):
            block = _block(num_layers=num_layers, input_feat=num_features, bn_size=bn_size, growth_rate=growth_rate, memory_efficienct=memory_effiency, drop_rate=drop_rate)
            self.layers.add_module(f"denseblock-{i+1}", block)

            num_features = initial_features + growth_rate * num_layers

            if i != len(block_config) - 1:
                transition_layer = _transition(num_features, num_features//2)
                self.layers.add_module(f'transition-{i+1}', transition_layer)
                num_features = num_features // 2

        # final batch norm
        self.layers.add_module('norm-final', nn.BatchNorm2d(num_features=num_features))

        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = t.flatten(x, 1)
        x = self.classifier(x)

        return x