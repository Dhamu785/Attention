
# %%
import torch as t
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from CBAM import cbam_net
import matplotlib.pyplot as plt
# %%
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])

pil_lst = []
img_lst = os.listdir('./sample')
for i in img_lst:
    pil_lst.append(transform(Image.open(os.path.join('./sample', i))))

input_t = t.from_numpy(np.array(pil_lst))
# %%
model = cbam_net(in_channels=3, num_class=6)
# model.load_state_dict(t.load('a.pt', map_location=DEVICE, weights_only=True))
model
# %%
layers = {'CAM':{}, 'SAM': {}}

def cam_out(mdl, inp, out):
    layers['CAM']['output'] = out.detach().cpu()

def cam_inp(mdl, inp, out):
    layers['CAM']['input'] = inp[0][0].detach().cpu()

def sam_inp(mdl, inp, out):
    layers['SAM']['input'] = inp.detach().cpu()

def sam_inp(mdl, inp, out):
    layers['SAM']['output'] = out.detach().cpu()

# %%
cam_i = model.features.Conv_1.conv. activation.register_forward_hook(cam_inp)
cam_o = model.features.SAM_1.conv.register_forward_hook(cam_out)
model(input_t)

# %%
model.features.CAM_1.avg
# %%
layers['CAM']['output'].shape
# %%
plt.imshow(layers['CAM']['input'].permute(1,2,0).numpy()[:,:,1], cmap='gray')
plt.show()
# %%
