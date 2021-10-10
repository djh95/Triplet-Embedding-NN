import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Define import *
from NUS_WIDE_Helper import *
from Inception import *



#IMAGE_HEIGHT = 299
#IMAGE_WIDTH = 299
#IMAGE_CHANNEL = 3
#Feature_Dimensions = 10
class TenNet_Image(nn.Module):
    def __init__(self):
        super().__init__()
        # YOUR CODE HERE
        self.feature_dimensions = Feature_Dimensions
        self.input_channels = IMAGE_CHANNEL
        self.image_H = IMAGE_HEIGHT
        self.image_W = IMAGE_WIDTH
        self.layer1 = nn.Sequential(  
            nn.Conv2d(self.input_channels, 16, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(16), 
            nn.ReLU(),	
            nn.MaxPool2d(kernel_size=2, stride=2))	

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))	
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),	
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))	

        self.fc = nn.Sequential(
            nn.Linear(int(self.image_H/8) * int(self.image_H/8) * 32, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, self.feature_dimensions))
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
