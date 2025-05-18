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

def make_divisible(v: int, divisor: int, min_val: Optional[int] = None) -> int:
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor/2)//divisor*divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class mobilenetv2(nn.Module):
    def __init__(self, num_class: int, width_mul: float, inverted_residual_setting: Optional[list[list[int]]]=None,
                    round_nearest: int = 8, block: Optional[Callable[..., nn.Module]] = None, normlayer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()

        if block is None:
            block = inverted_residual
        if normlayer is None:
            normlayer = nn.BatchNorm2d
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise "Issue in inverted residual setting"
        input_channels =32
        last_channel = 32
