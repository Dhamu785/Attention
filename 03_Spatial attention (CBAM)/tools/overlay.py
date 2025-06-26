# %% imports
from torchcam.utils import overlay_mask
from PIL import Image
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os

# %% img
pil_img = Image.open('C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\02\\Attention\\03_Spatial attention (CBAM)\\sample\\Targetspot.jpg').resize((256,256))

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
sav_loc = 'C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\02\\Attention\\03_Spatial attention (CBAM)\\sample\\CAM\\Targetspot'
for i in range(24):
    res = overlay_mask(pil_img, Image.fromarray(np.array(cam[5,i,:,:])), colormap='Dark2_r', alpha=0.1)
    res.save(os.path.join(sav_loc, f'Targetspot-{i}.png'))
# plt.imshow(np.array(cam[1,1,:,:]))
# %%
img_name = ['Aphids.jpg', 'Armyworm.jpg', 'Bactrialblight.jpg', 'Healthy.jpg', 'PowderyMildew.jpg', 'Targetspot.jpg']
path = 'C:\\Users\\dhamu\\Documents\\Python all\\torch_works\\02\\Attention\\03_Spatial attention (CBAM)\\sample'
for i in img_name:
    img = Image.open(os.path.join(path, i)).convert('RGB').resize((256,256))
    img.save(os.path.join(path, i.split('.')[0]+'_resized.jpg'))

# %%
