import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from Define import *
from .Utils import *


class VGG16_Normalize(nn.Module): # input batchSize * 1 * tagNum * tagNum
    def __init__(self):
        super().__init__()
        self.VGG16 = models.vgg16(pretrained=True)
        
    def forward(self, x):
        out = self.VGG16(x)
        return F.normalize(out)

