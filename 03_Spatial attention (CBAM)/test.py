# %%
from CBAM import CAM

# %%
blk = CAM(8, 16)
# %%
import torch as t
# %%
x = t.randn(4,8,52,52)
x.shape
# %%
res = blk(x)
# %%
res.shape
# %%
