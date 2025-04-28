import torch as t
import torch.nn as nn

class basic_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 32, 3)
        self.norm = nn.BatchNorm2d(32)
    def forward(self, x):
        return self.norm(self.conv1(x))
    
class test(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv15 = nn.Conv2d(3, 32, 3)
        self.BB = basic_block()

        for m in self.modules():
            if isinstance(m, basic_block):
                print("layer >>>>", m.norm)

    def forward(self, x):
        return self.BB(self.conv1(x))
    
rand = t.randn(10, 3, 32, 32)

model = test()
print(model)
# for m in model.parameters():
#     print(m)