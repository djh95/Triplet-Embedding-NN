import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Define import *
from SentenceCNN import *
from NUS_WIDE_Helper import *


class TagDecoder(nn.Module):
    def __init__(self, tag_number):
        super().__init__()
        self.decoder = nn.Sequential(
            # [b, 20] => [b, 64]
            nn.Linear(Feature_Dimensions, 32),
            nn.ReLU(),
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, tag_number),
            nn.Sigmoid()
        )
 
 
    def forward(self, x):
        """
        :param [b, feature_dimensionality]:
        :return [b, tag_number]:
        """
        # decoder
        x = self.decoder(x)
        return x