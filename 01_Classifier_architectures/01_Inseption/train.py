from googleNet import *
import torch as t

model = GoogleNet(num_classes=100, aux_logit=True, transform_input=False, init_weight=True)
sample = t.randn(4, 3, 244, 244)
pred = model(sample)
print(len(pred))
