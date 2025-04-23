# %% imports
import torch as t
# %%
a = t.randn((10,5), requires_grad=True)
print(a.shape)
print(a)
# %%
print(a.detach().clone())
# %%
