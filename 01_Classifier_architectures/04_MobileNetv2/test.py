# %%
from mobilenet import mobilenetv2

import torch as t
# %%
x = t.randn(8, 3, 224, 224)
x.shape
# %%
model = mobilenetv2(10, 2)
# %%
model
# %%
res = model(x)
# %%
res.shape
# %%
