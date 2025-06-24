import torch as t
import torch.nn as nn
from collections import OrderedDict

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Sequential(OrderedDict([
                    ('conv_layer', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)),
                    ('BatchNorm', nn.BatchNorm2d(out_channels)),
                    ('activation', nn.ReLU(inplace=True))
        ]))

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.conv(x)
        return x
    
class CAM(nn.Module):
    def __init__(self, in_channels: int, reduction_factor: int = 16) -> None:
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

class SAM(nn.Module):
    def __init__(self, bias: bool) -> None:
        super().__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3, bias=self.bias)

    def forward(self, x: t.Tensor) -> t.Tensor:
        maxx = t.max(x, 1)[0].unsqueeze(1)
        avg = t.mean(x, 1).unsqueeze(1)
        return nn.functional.sigmoid(self.conv(t.concat((maxx, avg), dim=1))) * x

class cbam_net(nn.Module):
    def __init__(self, in_channels: int, num_class: int) -> None:
        super().__init__()
        self.features = nn.Sequential()
        self.features.add_module('Conv_1', ConvBlock(in_channels=in_channels, out_channels=24, kernel_size=3, stride=2, padding=1)) # 24,256,256
        self.features.add_module('CAM_1', CAM(in_channels=24, reduction_factor=16))
        self.features.add_module('SAM_1', SAM(bias=True))
        self.features.add_module('Conv_2', ConvBlock(in_channels=24, out_channels=256, kernel_size=3, padding=1, stride=2)) # 256, 128, 128
        self.features.add_module('CAM_2', CAM(in_channels=256))
        self.features.add_module('SAM_2', SAM(bias=True))
        self.features.add_module('Conv_3', ConvBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2)) # 512, 64, 64
        self.features.add_module('CAM_3', CAM(in_channels=512))
        self.features.add_module('SAM_3', SAM(bias=True))
        self.features.add_module('Conv_4', ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2)) # 1024, 32, 32
        self.features.add_module('CAM_4', CAM(in_channels=1024))
        self.features.add_module('SAM_4', SAM(bias=True))
        self.features.add_module('Conv_5', ConvBlock(in_channels=1024, out_channels=2048, kernel_size=3, padding=1, stride=5)) # 2048, 7, 7
        self.features.add_module('CAM_5', CAM(in_channels=2048))
        self.features.add_module('SAM_5', SAM(bias=True))
        self.flatten = nn.Flatten(start_dim=1)

        self.output = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256, out_features=num_class)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = nn.functional.adaptive_avg_pool2d(self.features(x), (1,1))
        x = self.flatten(x)
        return self.output(x)
