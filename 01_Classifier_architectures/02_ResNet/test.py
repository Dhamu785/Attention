# %% import libaries
from Res_Net import resnet, blocks
# %%
import torch as t
# %%
a = t.randn(2,3,512,512)
# %%
model = resnet.ResNet(blocks.BasicBlock, [2,2,2,2], 10)
res = model(a)
# %%
res.shape
# %%
