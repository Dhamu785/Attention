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