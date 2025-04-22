import torch as t
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split

from dataset import custom_data_prep

from tqdm import tqdm
from typing import Callable

class training:
    def __init__(self, optimizer: Optimizer, 
                    train_loader: DataLoader, val_loader: DataLoader, 
                    epochs: int, loss: Callable[[Tensor, Tensor], Tensor], device: str):
        self.optmiz = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.loss = loss
        self.DEVICE = device

    def _calc_acc(self, pred_logits: Tensor, true_val: Tensor) -> float:
        predicted = t.argmax(pred_logits, 1)
        acc = (predicted == true_val).sum() / len(true_val)
        return acc
    
    def train(self, model: t.nn.Module):
        for epoch in range(1, self.epochs+1):
            model.train()
            train_ls = 0
            val_ls = 0
            total_batchs = len(self.train_loader)
            epoch_bar = tqdm(range(total_batchs), desc="Batch processing", unit="Batchs", colour=(0, 255, 0))
            for batch in self.train_loader:
                x = batch[0]
                y = batch[1]