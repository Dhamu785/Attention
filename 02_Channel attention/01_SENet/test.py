# %% libs
import torch as t
from SENet.se_net import SE_Net

# %%
a = t.randn((6, 3, 224,224))
a.shape

# %%
bottleneck_width = 4
layers = [1,1,1,1]
cardinality = 32
channels_per_layers = [256, 512, 1024, 2048]
init_block_channels = 128

channels = [[j]*i for i,j in zip(layers, channels_per_layers)]
print(channels)

# %%

model = SE_Net(channels=channels, init_block_channels=init_block_channels, cardinality=cardinality, bottleneck_width=bottleneck_width,
                in_channels=3)
res = model(a)
print(res.shape)
