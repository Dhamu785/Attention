from parts import basicConv2d, Inception, InceptionAux

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Callable, Any, Union, Tuple
import warnings

class Inception(nn.Module):
    def __init__(self, num_classes: int, aux_logit: bool, transform_input: bool,
                    init_weight: Optional[bool] = None, blocks: List[Callable[..., nn.Module]] = None,
                    drop_out: float = 0.20, drop_out_aux: float = 0.70) -> None:
        super().__init__()
        if init_weight is None:
            warnings.warn("The score behaviours are different from the original GoogleNet", FutureWarning)
            init_weight = True
        if len(blocks) != 3:
            raise ValueError(f"Blocks length should be 3 instead of {len(blocks)}")
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logit = aux_logit
        self.transform_input = transform_input
        
        