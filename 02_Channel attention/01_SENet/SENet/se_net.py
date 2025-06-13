import torch as t
from torch import nn
from blocks import conv3x3_block

class init_block(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        mid_channel = out_channel // 2

        self.conv1 = conv3x3_block(in_chennels=in_channel, out_chennels=mid_channel, stride=2)
        self.conv2 = conv3x3_block(in_channel=mid_channel, out_chennels=mid_channel)
        self.conv3 = conv3x3_block(in_channel=mid_channel, out_chennels=out_channel)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x) -> t.Tensor:
        x = self.conv1(self.conv2(self.conv3(self.pool(x))))
        return x
    
class SE_Net(nn.Module):
    def __init__(self, channels: list[list], init_block_channels: int, cardinality: int, bottleneck_width: int, in_channels: int = 3, in_size = (224, 224), num_classes=100) -> None:
        super().__init__()
        self.in_size = in_size
        self.num_class = num_classes

        self.features = nn.Sequential()
        self.features.add_module('init_block', init_block(in_channel=in_channels, out_channel=init_block_channels))
        in_channels = init_block_channels

        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()