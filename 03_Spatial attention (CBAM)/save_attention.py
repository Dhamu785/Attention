# %%
import torch as t
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from CBAM import cbam_net
import pickle as pkl
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
def inp(mdl, inp, out):
    layers['inputs'] = inp[0].detach().cpu()

def cam_out(mdl, inp, out):
    layers['CAM']['output'] = out.detach().cpu()

def sam_out(mdl, inp, out):
    layers['SAM']['output'] = out.detach().cpu()
# %%
model = cbam_net(in_channels=3, num_class=6)
mdl_path = 'C:\\Users\\dhamu\\Downloads\\models'

for pth in os.listdir(mdl_path):
    epoch = pth.split('-')[1].split('.')[0]
    model.load_state_dict(t.load(os.path.join(mdl_path, pth), map_location=DEVICE, weights_only=True))

    layers = {'CAM':{}, 'SAM': {}}

    cam_i = model.features.Conv_1.conv.activation.register_forward_hook(inp)
    cam_o = model.features.CAM_1.sigmoid.register_forward_hook(cam_out)
    sam_o = model.features.SAM_1.sigmoid.register_forward_hook(sam_out)

    model.eval()
    with t.inference_mode():
        model(input_t)

    with open(f'./hooks_info/hooks_info_{epoch}.pkl', 'wb') as f:
        pkl.dump(layers,f)

# %%
print(f"SAM output shape = {layers['SAM']['output'].shape}")
print(f"CAM output shape = {layers['CAM']['output'].shape}")
print(f"Inputs shape = {layers['inputs'].shape}")
