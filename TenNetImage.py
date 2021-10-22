import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

from Define import *
from NUS_WIDE_Helper import *
from Inception import *



#IMAGE_HEIGHT = 
#IMAGE_WIDTH = 
#IMAGE_CHANNEL = 3
#Feature_Dimensions = 10

class TenNet_Image(nn.Module):
    def __init__(self, dropout_probability=0.5):
        super().__init__()
        # YOUR CODE HERE
        self.feature_dimensions = Feature_Dimensions
        self.input_channels = IMAGE_CHANNEL
        self.image_H = IMAGE_HEIGHT
        self.image_W = IMAGE_WIDTH
        self.feature = models.vgg16(pretrained=True).features	

        self.fc = nn.Sequential(
            nn.Linear(int(self.image_H/8) * int(self.image_H/8) * 32, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, self.feature_dimensions))

        self.fc = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_probability, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_probability, inplace=False),
            nn.Linear(in_features=4096, out_features=self.feature_dimensions, bias=True)
        )
    
    def forward(self, x):
        out = self.feature(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return F.normalize(out)

class TenNet_Image2(nn.Module):
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
