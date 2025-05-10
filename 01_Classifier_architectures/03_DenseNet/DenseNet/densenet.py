import torch as t
import torch.nn as nn
from torch import Tensor
import torch.utils.checkpoint as cp
import torch.nn.functional as F

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
            layer = _layers(input_feat, bn_size, growth_rate, drop_rate, memory_efficienct)
            self.add_module(f'denselayer {i+1}', layer)

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