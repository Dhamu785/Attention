import torch as t
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.conv(x)
    
class CAM(nn.Module):
    def __init__(self, in_channels: int, reduction_factor: int) -> None:
        super().__init__()
        self.channels = in_channels
        self.reduction = reduction_factor
        self.avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.max = nn.AdaptiveMaxPool2d(output_size=1)
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.reduction, out_features=self.channels)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        b, c, _, _ = x.shape
        linear_max = self.linear(self.max(x).view(b,c)).view(b,c,1,1)
        linear_mean = self.linear(self.avg(x).view(b,c)).view(b,c,1,1)
        out = nn.functional.sigmoid(linear_max + linear_mean) * x
        return out
