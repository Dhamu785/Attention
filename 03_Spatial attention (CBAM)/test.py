# %%
from CBAM import SAM

# %%
blk = SAM(False)
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
