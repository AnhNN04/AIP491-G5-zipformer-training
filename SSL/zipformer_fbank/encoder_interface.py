from typing import Tuple

import torch
import torch.nn as nn

class EncoderInterface(nn.Module):
    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Please implement it in a subclass")
