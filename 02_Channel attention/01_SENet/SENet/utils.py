import torch as t
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from typing import Callable, Union, Dict
from inspect import isfunction
from tqdm import tqdm

import matplotlib.pyplot as plt

def get_activation(act: Union[Callable[..., Callable], str]):
    if isfunction(act):
        return act()
    elif isinstance(act, str):
        if act == 'relu':
            return nn.ReLU(inplace=True)
        elif act == 'relu6':
            return nn.ReLU6(inplace=True)
        else:
            raise NotImplementedError()
    else:
        assert isinstance(act, nn.Module)
        return act

class training:
    def __init__(self, train_dataloader: DataLoader,  test_dataloader: DataLoader, Epochs: int, optimizer: Optimizer, 
                    loss_fn: Callable[[t.Tensor, t.Tensor], t.Tensor], device: str) -> None:
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = Epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scaler = t.GradScaler(device=device)

    def calc_acc(self, predictions: t.Tensor, y: t.Tensor) -> float:
        predictions = t.argmax(predictions, 1)
        return (predictions == y).float().mean()

    def train_model(self, model: t.nn.Module) -> tuple[list[float]]:
        model.to(device=self.device)
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []

        for epoch in range(1, self.epochs+1):
            model.train()
            bth_train_ls = 0
            bth_train_acc = 0

            train_len = len(self.train_dataloader)
            test_len = len(self.test_dataloader)
            bar = tqdm(iterable=range(train_len), desc='Batch processing', unit='Batchs', colour='GREEN')
            for data, lbl in self.train_dataloader:
                data = data.to(self.device)
                lbl = lbl.to(self.device)

                with t.autocast(device_type=self.device):
                    preds = model(data)
                    ls = self.loss_fn(preds, lbl)
                acc = self.calc_acc(predictions=preds.detach().clone(), y=lbl)
                self.optimizer.zero_grad()
                self.scaler.scale(ls).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                bth_train_acc += acc.item()
                bth_train_ls += ls.item()

                bar.set_postfix(loss=ls.item(), acc=acc.item())
                bar.update(1)
            
            bar.close()
            train_acc.append(bth_train_acc/train_len)
            train_loss.append(bth_train_ls/train_len)

            model.eval()
            bth_test_ls = 0
            bth_test_acc = 0
            for data, lbl in self.test_dataloader:
                data = data.to(self.device)
                lbl = lbl.to(self.device)

                with t.inference_mode():
                    with t.autocast(device_type=self.device):
                        preds = model(data)
                        ls = self.loss_fn(preds, lbl)
                    acc = self.calc_acc(predictions=preds.detach().clone(), y=lbl)
                    bth_test_acc += acc.item()
                    bth_test_ls += ls.item()
            
            test_loss.append(bth_test_ls/test_len)
            test_acc.append(bth_test_acc/test_len)
            print(f"{epoch} / {self.epochs} | train ls = {train_loss[-1]:.4f} | train acc = {train_acc[-1]:.4f} | test ls = {test_loss[-1]:.4f} | test acc = {test_acc[-1]:.4f}")
        
        return (train_loss, train_acc, test_loss, test_acc)

def plot(imgs: list[t.Tensor], predictions: t.Tensor, true: t.Tensor, lbl: Dict[str,int]) -> None:
    plt.figure(figsize=(20, 20))
    for i in range(1, 26):
        plt.subplot(5, 5, i)
        plt.imshow(imgs[i-1].permute(1,2,0).cpu().numpy())
        true_lbl = lbl[true[i-1]]
        pred_lbl = lbl[predictions[i-1]]
        if true_lbl == pred_lbl:
            plt.title(f"{pred_lbl}", color='green')
        else:
            plt.title(f"act: {true_lbl}\npred: {pred_lbl}", color='red')
        plt.axis('off')
    plt.show()