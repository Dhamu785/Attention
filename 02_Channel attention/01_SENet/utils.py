import torch as t
from torch import nn
from typing import Callable, Union
from inspect import isfunction

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