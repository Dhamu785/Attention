import torch as t
from torch import nn, Tensor
from typing import Optional, Callable
from torchvision.ops import Conv2dNormActivation

class inverted_residual(nn.Module):
    def __init__(self, inp: int, out: int, exp_rate: float, stride: int, normlayer: Optional[Callable[..., nn.Module]]=None) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1,2]:
            raise "Stride must be 1 or 2"
        if normlayer is None:
            normlayer = nn.BatchNorm2d

        hidden_dim = int(round(inp * exp_rate))
        self.use_res_connection = inp == out and self.stride == 1

        layers: list[nn.Module] = []

        if exp_rate != 1:
            layers.append(Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=normlayer, activation_layer=nn.ReLU6))
        layers.extend([
            Conv2dNormActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=normlayer, activation_layer=nn.ReLU6),
            nn.Conv2d(hidden_dim, out, 1, 1, 0, bias=False),
            normlayer(out)
            ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connection:
            return x + self.conv(x)
        else:
            return self.conv(x)
