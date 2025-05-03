from torch.utils.data import Dataset
from torchvision import transforms
import torch as t
from torch import Tensor

import os
from PIL import Image
from typing import Callable, Optional, Tuple

class custom_data_prep(Dataset):
    def __init__(self, root_dir: str, transform: Optional[Callable[..., Tensor]]=None) -> None:
        self.root = root_dir
        self.transforms = transform

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {idx : cls_name for idx, cls_name in enumerate(self.classes)}

        self.img_paths = []
        self.labels = []

        self.to_tensor = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        for cls_n in self.calsses:
            fldr_fls = os.listdir(os.path.join(root_dir, cls_n))
            for file in fldr_fls:
                self.img_paths.appedn(os.path.join(root_dir, cls_n, file))
                self.labels.append(self.class_to_idx[cls_n])

        def __len__(self) -> int:
            return len(self.labels)
        
        def __getitem__(self, index: int) -> Tuple[Tensor, int]:
            img_pth = self.img_paths[index]
            lbl = self.labels[index]
            pil_img = Image.open(img_pth)

            if self.transforms:
                transformed = self.transforms(pil_img)
            else:
                transformed = self.to_tensor(pil_img)
            return (transformed, t.tensor(lbl, dtype=t.long))