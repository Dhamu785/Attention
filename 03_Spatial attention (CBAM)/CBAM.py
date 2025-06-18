import torch as t
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Sequential([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.conv(x)