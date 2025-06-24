# %%
from CBAM import cbam_net
from torch.utils.tensorboard import SummaryWriter
# %%
blk = cbam_net(in_channels=3, num_class=6)
# %%
import torch as t
# %%
x = t.randn(4,3,512,512)
x.shape
# %%
writer = SummaryWriter(log_dir='runs/test')
writer.add_graph(blk, x)

# %%
res = blk(x)
res.shape
# %%
type(blk.features.Conv_1.conv)
# %%
t.nn.ModuleDict()