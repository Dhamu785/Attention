import torch as t
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

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
        self.scalar = t.GradScaler(device=device)

    def _calc_acc(self, pred_logits: Tensor, true_val: Tensor) -> float:
        predicted = t.argmax(pred_logits, 1)
        acc = (predicted == true_val).float().mean()
        return acc
    
    def train(self, model: t.nn.Module):
        for epoch in range(1, self.epochs+1):
            model.to(self.DEVICE)
            model.train()
            train_ls = 0
            train_acc = 0
            total_train_batchs = len(self.train_loader)
            total_val_batchs = len(self.val_loader)
            epoch_bar = tqdm(range(total_train_batchs), desc="Batch processing", unit="Batchs", colour='GREEN')
            for batch in self.train_loader:
                x = batch[0].to(self.DEVICE)
                y = batch[1].to(self.DEVICE)

                # 1. Forward pass
                with t.autocast(device_type=self.DEVICE):
                    preds, aux1, aux2 = model(x)
                    # 2. Calculate the loss
                    ls = self.loss(preds, y) + self.loss(aux1, y) + self.loss(aux2, y)
                acc = self._calc_acc(preds.detach().clone(), y)
                # 3. Zero grad
                self.optmiz.zero_grad()
                # 4. Backward pass
                self.scalar.scale(ls).backward()
                # 5. Step
                self.scalar.step(self.optmiz)
                self.scalar.update()
                train_ls += ls.item()
                train_acc += acc.item()
                epoch_bar.set_postfix(loss=ls.item(), accuracy=acc.item())
                epoch_bar.update(1)
            epoch_bar.close()
            train_ls /= total_train_batchs
            train_acc /= total_train_batchs

            model.eval()
            val_ls = 0
            val_acc = 0
            for batch in self.val_loader:
                x = batch[0].to(self.DEVICE)
                y = batch[1].to(self.DEVICE)
                with t.inference_mode():
                    with t.autocast(device_type=self.DEVICE):
                        test_out = model(x)
                        ls = self.loss(test_out, y)
                        acc = self._calc_acc(test_out, y)
                    val_ls += ls.item()
                    val_acc += acc.item()
            val_ls /= total_val_batchs
            val_acc /= total_val_batchs

            print(f"{epoch} / {self.epochs} | train_ls = {train_ls:.4f} | train_acc = {train_acc:.4f} | val_ls = {val_ls:.4f} | val_acc = {val_acc:.4f}")