# %% imports
from torchcam.utils import overlay_mask
from PIL import Image
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os

# %% img
pil_img = Image.open('C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\02\\Attention\\03_Spatial attention (CBAM)\\sample\\Armyworm.jpg').resize((256,256))

with open(r'C:\Users\dhamu\Documents\Python all\torch_works\02\Attention\03_Spatial attention (CBAM)\hooks_info\hooks_info_50.pkl', 'rb') as f:
    hooks_data = pkl.load(f)

sam_out = hooks_data['SAM']['output']
cam_out = hooks_data['CAM']['output']
inputs = hooks_data['inputs']

cam = inputs * cam_out
sam = inputs * sam_out
print(f"{cam.shape = }, {sam.shape = }")
print(f'{type(pil_img)}')
# %%
sav_loc = 'C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\02\\Attention\\03_Spatial attention (CBAM)\\sample\\CAM\\Armyworm'
for i in range(24):
    res = overlay_mask(pil_img, Image.fromarray(np.array(cam[1,i,:,:])), colormap='Dark2_r', alpha=0.1)
    res.save(os.path.join(sav_loc, f'Armyworm-{i}.png'))
# plt.imshow(np.array(cam[1,1,:,:]))
# %%
plt.imshow(np.array(cam[5,1,:,:]))

# %%
