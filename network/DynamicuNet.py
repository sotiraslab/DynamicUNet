from typing import Union, Type, List, Tuple

import torch
from torch import nn

from nnunetv2.network.DynamicuNet_module import Encoder
from nnunetv2.network.DynamicuNet_module import Decoder


class DUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 deep_supervision: bool = False,
                 ):
        super().__init__()

        self.encoder = Encoder(input_channels, n_stages, features_per_stage)
        self.decoder = Decoder(self.encoder, num_classes, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
