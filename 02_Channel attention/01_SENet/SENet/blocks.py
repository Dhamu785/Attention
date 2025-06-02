import torch as t
from torch import nn
from typing import Optional, Callable, Union

class conv3x3_block(nn.Module):
    def __init__(self, in_chennels: int, out_chennels: int, kernel_size: int, stride: int, padding: int, dilation: int = 1, groups: int = 1, bias: bool = False,
                    bn_eps: float = 1e-5, activation: Union[Callable[..., t.Tensor], str]=(lambda: nn.ReLU(inplace=True))):
        ...