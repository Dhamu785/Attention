# %% 
from DenseNet.densenet import denseNet

import torch as t
# %%

a = t.randn(8, 3, 64, 64)
# %%
model = denseNet([2,2,2,2], 10, memory_effiency=False)
# %%
model
# %%
out = model(a)
# %%
out.shape
# %%
