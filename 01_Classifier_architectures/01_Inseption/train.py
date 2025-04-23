# %% imports
import googleNet
import dataset
import utils

import torch as t
from torch.utils.data import DataLoader, random_split

import os
from icecream import ic

# %%
root_pth = r'C:\Users\dhamu\Downloads\archive (1)\Main dataset-20230209T191052Z-001\Main dataset'
datasets = dataset.custom_data_prep(root_pth)

train, test = random_split(dataset=datasets, lengths=(0.8, 0.2))
train_loader = DataLoader(train, batch_size=4, shuffle=True)
test_loader = DataLoader(test, batch_size=4, shuffle=True)
# %%
print(next(iter(test_loader)))
# %%
model = googleNet.GoogleNet(6, True, False, True)
# %%
optimizer = t.optim.Adam(model.parameters(), lr=0.001)
loss = t.nn.CrossEntropyLoss()

model_set = utils.training(optimizer=optimizer, train_loader=train_loader, val_loader=test_loader, epochs=5, loss=loss, device='cuda')
# %%
model_set.train(model=model)
# %%
