# %%
import pickle as pkl
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os
import re
# %%
sorted_files = sorted(os.listdir('hooks_info'), key=lambda x: int(re.search(r'\d+', x).group()))
writer = SummaryWriter(log_dir='./runs/images')

# %%
for i, srt_file in enumerate(sorted_files):
    print(i)
    with open(f'.\hooks_info\{srt_file}', 'rb') as f:
        hooks_data = pkl.load(f)

    sam_out = hooks_data['SAM']['output']
    cam_out = hooks_data['CAM']['output']
    inputs = hooks_data['inputs']

    # print(f"SAM output shape = {sam_out.shape}")
    # print(f"CAM output shape = {cam_out.shape}")
    # print(f"Inputs shape = {inputs.shape}")

    cam = inputs * cam_out
    sam = inputs * sam_out
    print(f"{cam.shape = }, {sam.shape = }")

    for j in range(24):
        cam_grid = make_grid(cam[:,j,:,:].unsqueeze(1), 6, 1, normalize=False, scale_each=False, pad_value=2)
        sam_grid = make_grid(sam[:,j,:,:].unsqueeze(1), 6, 1, normalize=False, scale_each=False, pad_value=2)

        writer.add_image(f'CAM-channel-{j}', cam_grid, global_step=i)
        writer.add_image(f'SAM-channel-{j}', sam_grid, global_step=i)

writer.close()
# %%
grid = make_grid(cam[:,1,:,:].unsqueeze(1), 2, 3, pad_value=2).moveaxis(0, 2)
f, axs = plt.subplots(figsize=(5,10))
axs.set_axis_off()
# plt.close()
axs.imshow(grid, cmap='gray')
