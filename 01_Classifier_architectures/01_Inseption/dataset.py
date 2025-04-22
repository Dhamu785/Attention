import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch import Tensor

from typing import Tuple, Optional, Union, Callable

class custom_data_prep(Dataset):
    def __init__(self, root_dir: str, transform: Union[Callable[[Image.Image], Tensor], None]):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name:idx for idx, cls_name in enumerate(self.classes)}

        self.img_paths = []
        self.labels = []

        self.to_tensor = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])

        for cls_n in self.classes:
            cls_fldr = os.path.join(root_dir, cls_n)
            for file in os.listdir(cls_fldr):
                self.img_paths.append(os.path.join(cls_fldr, file))
                self.labels.append(self.class_to_idx[cls_n])
    
    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        img_pth = self.img_paths[index]
        image = Image.open(img_pth).convert('RGB')
        lbl: int = self.labels[index]

        if self.transform:
            transformed: Tensor = self.transform(image)
        else:
            transformed: Tensor = self.to_tensor(image)

        return transformed, lbl