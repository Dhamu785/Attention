# %% import libaries
from Res_Net import resnet, blocks
import utils

import torch as t
# %%
a = t.randn(25,3,512,512)
# %%
model = resnet.ResNet(blocks.BasicBlock, [2,2,2,2], 6)
res = model(a)
# %%
target = t.tensor([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,1,1,1,1,2,2,2])
lbl = {0: 'Aphids', 1: 'Army worm', 2: 'Bacterial Blight', 3: 'Healthy', 4: 'Powdery Mildew', 5: 'Target spot'}
utils.plot(a, t.argmax(res, 1).cpu().numpy(), target.numpy(), lbl)
# %%
