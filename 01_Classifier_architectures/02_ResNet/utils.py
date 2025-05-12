import torch as t
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader


from tqdm import tqdm
from typing import Callable, Tuple, List, Dict

from PIL import Image
import matplotlib.pyplot as plt

class train:
    def __init__(self, optimizer: Optimizer, train_loader: DataLoader,
                    test_loader: DataLoader, epochs: int, loss: Callable[[Tensor, Tensor], Tensor],
                    device: str):
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.loss = loss
        self.DEVICE = device
        self.scalar = t.GradScaler(device=device)

    def _calc_acc(self, pred_logits: Tensor, true_val: Tensor) -> float:
        predictions = t.argmax(pred_logits, 1)
        acc = (predictions == true_val).float().mean()
        return acc
    
    def train_model(self, model: t.nn.Module, lrshed: bool=False) -> Tuple[List[float],...]:
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        learning_rates = []
        best_loss = float('inf')
        best_acc = 0
        for epoch in range(1, self.epochs+1):
            model.to(self.DEVICE)
            model.train()
            bth_train_ls = 0
            bth_train_acc = 0
            train_len = len(self.train_loader)
            test_len = len(self.test_loader)
            epoch_bar = tqdm(range(train_len), desc="Batch processing", unit="Batchs", colour='GREEN')
            for batch in self.train_loader:
                x = batch[0].to(self.DEVICE)
                y = batch[1].to(self.DEVICE)

                with t.autocast(device_type=self.DEVICE):
                    preds = model(x)
                    ls = self.loss(preds, y)
                acc = self._calc_acc(preds.detach().clone(), y)
                self.optimizer.zero_grad()
                self.scalar.scale(ls).backward()
                self.scalar.step(self.optimizer)
                self.scalar.update()
                bth_train_ls += ls.item()
                bth_train_acc += acc.item()
                epoch_bar.set_postfix(loss=ls.item(), accuracy=acc.item())
                epoch_bar.update(1)
            if lrshed:
                lrshed.step()
                learning_rates.append(lrshed.get_last_lr()[0])
            epoch_bar.close()
            bth_train_acc /= train_len
            bth_train_ls /= train_len
            train_acc.append(bth_train_acc)
            train_loss.append(bth_train_ls)

            model.eval()
            bth_val_ls = 0
            bth_val_acc = 0
            for batch in self.test_loader:
                x = batch[0].to(self.DEVICE)
                y = batch[1].to(self.DEVICE)
                with t.inference_mode():
                    with t.autocast(device_type=self.DEVICE):
                        val_preds = model(x)
                        ls = self.loss(val_preds, y)
                        acc = self._calc_acc(val_preds, y)
                    bth_val_ls += ls.item()
                    bth_val_acc += acc.item()
            bth_val_ls /= test_len
            bth_val_acc /= test_len
            test_loss.append(bth_val_ls)
            test_acc.append(bth_val_acc)
            if bth_val_ls < best_loss:
                t.save(model.state_dict(), 'best-res.pt')
            if epoch == self.epochs:
                t.save(model.state_dict(), 'last-res.pt')
            print(f"{epoch}/{self.epochs} | train_ls = {bth_train_ls:.4f} | train_acc = {bth_train_acc:.4f} | test_ls = {bth_val_ls:.4f} | test_acc = {bth_val_acc:.4f}")

        if lrshed:
            return (train_loss, train_acc, test_loss, test_acc, learning_rates)
        else:
            return (train_loss, train_acc, test_loss, test_acc)

def plot(imgs: List[Tensor], predictions: Tensor, true: Tensor, lbl: Dict[str,int]) -> None:
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