# %%
from CBAM import cbam_net

# %%
blk = cbam_net(in_channels=3, num_class=6)
# %%
import torch as t
# %%
x = t.randn(4,3,512,512)
x.shape
# %%
res = blk(x)
# %%
res.shape
# %%
