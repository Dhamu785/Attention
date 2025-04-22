from parts import basicConv2d, Inception, InceptionAux

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Optional, List, Callable, Any, Union, Tuple
import warnings

from collections import namedtuple

GoogleNetOutputs = namedtuple("GoogleNetOutputs", ["logits", "aux_logits1", "aux_logits2"])
GoogleNetOutputs.__annotations__ = {"logits": Tensor, "aux_logits1": Optional[Tensor], "aux_logits2": Optional[Tensor]}
_googlenetOutputs = GoogleNetOutputs
class Inception(nn.Module):
    __constants__ = ['aux_logit', 'transform_input']
    def __init__(self, num_classes: int, aux_logit: bool, transform_input: bool,
                    init_weight: Optional[bool] = None, blocks: List[Callable[..., nn.Module]] = None,
                    drop_out: float = 0.20, drop_out_aux: float = 0.70) -> None:
        super().__init__()
        if init_weight is None:
            warnings.warn("The score behaviours are different from the original GoogleNet", FutureWarning)
            init_weight = True
        if len(blocks) != 3:
            raise ValueError(f"Blocks length should be 3 instead of {len(blocks)}")
        if blocks is None:
            blocks = [basicConv2d, Inception, InceptionAux]

        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logit = aux_logit
        self.transform_input = transform_input
        
        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(62, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32, basicConv2d)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64, basicConv2d)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64, basicConv2d)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64, basicConv2d)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64, basicConv2d)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64, basicConv2d)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128, basicConv2d)
        self.maxpool4 = nn.Maxpool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128, basicConv2d)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128, basicConv2d)

        if aux_logit:
            self.aux1 = inception_aux_block(512, num_classes, drop_out_aux, basicConv2d)
            self.aux2 = inception_aux_block(528, num_classes, drop_out_aux, basicConv2d)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=drop_out)
        self.fc = nn.Linear(1024, num_classes)

        if init_weight:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    t.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _transform_input(self, x:Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = t.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = t.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = t.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = t.cat([x_ch0, x_ch1, x_ch2])
        return x
        
    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        aux1: Optional[Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2: Optional[Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = t.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x, aux1, aux2

    @t.jit.unused
    def eager_output(self, x: Tensor, aux1: Tensor, aux2: Tensor) -> GoogleNetOutputs:
        if self.training and self.aux_logit:
            return _googlenetOutputs(x, aux1, aux2)
        else:
            return x
    def forward(self, x: Tensor) -> GoogleNetOutputs:
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)
        aux_defined = self.training and self.aux_logit
        if t.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted GoogleNet will always return GoogleNetOuputs Tuple")
            return GoogleNetOutputs(x, aux1, aux2)
        else:
            return self.eager_output(x, aux1, aux2)