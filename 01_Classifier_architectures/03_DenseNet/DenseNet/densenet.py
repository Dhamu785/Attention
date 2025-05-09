import torch as t
import torch.nn as nn

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