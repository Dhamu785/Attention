import torch as t
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, List, Tuple, Callable, Any, Union

class basicConv2d(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, **kwargs:Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x:Tensor) -> Tensor:
        x = F.relu(self.bn(self.conv(x)), inplace=False)
        return x
    
class Inception(nn.Module):
    # def __init__(self, in_channel:int, ch1x1:int, ch3x3red:int, ch3x3:int, ch5x5red:int, ch5x5:int, pool_proj:int, conv_blk: Union[Callable[..., nn.Module], None]) -> None:
    def __init__(self, in_channel:int, ch1x1:int, ch3x3red:int, ch3x3:int, ch5x5red:int, ch5x5:int, pool_proj:int, conv_blk: Optional[Callable[..., nn.Module]] = None) -> None:
        # super(Inception, self).__init__()
        super().__init__()
        if conv_blk is None:
            conv_blk = basicConv2d
        self.branch1 = conv_blk(in_channel, ch1x1, kernel_size=(1,1))
        self.branch2 = nn.Sequential(
            conv_blk(in_channel, ch3x3red, kernel_size=(1,1)),
            conv_blk(ch3x3red, ch3x3, kernel_size=(3,3), padding=1)
        )
        self.branch3 = nn.Sequential(
            conv_blk(in_channel, ch5x5red, kernel_size=(1,1)),
            conv_blk(ch5x5red, ch5x5, kernel_size=(5,5), padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=1, ceil_mode=True),
            conv_blk(in_channel, pool_proj, kernel_size=1)
        )

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return [branch1, branch2, branch3, branch4]
    
    def forward(self, x:Tensor) -> Tensor:
        outputs = self._forward(x)
        return t.cat(outputs, 1)
    

class InceptionAux(nn.Module):
    def __init__(self, in_channels:int, num_classes:int, dropout:float = 0.7, conv2d_blk: Optional[Callable[..., nn.Module]]=None) -> None:
        super().__init__()
        if conv2d_blk is None:
            conv2d_blk = basicConv2d
        self.conv = conv2d_blk(in_channels, 128, kernel_size=(1,1))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x:Tensor) -> Tensor:
        x = F.adaptive_max_pool2d(x, (4,4))
        x = self.conv(x)
        x = t.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        return x