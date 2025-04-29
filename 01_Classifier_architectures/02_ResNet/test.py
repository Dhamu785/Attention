# %% import libaries
from resnet import ResNet
from blocks import *
import torch as t

# %% call resnet
model = ResNet(BasicBlock, [2,2,2,2], 100, True)
# %%
test_data = t.randn((4,3,244,244))
res = model(test_data)
# %%
print(res)
# %%
res.shape
# %%