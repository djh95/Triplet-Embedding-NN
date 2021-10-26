from math import ceil
import numpy as np
import torch
import torch.nn.functional as F
import random
import os

from Define import *

def get_tag_from_prediction(predictions, threshold=0.5):
    tags = []
    for i in range(predictions.shape[0]):
        temp_tag = []
        for j in range(predictions.shape[1]):
            if predictions[i][j] > threshold:
                temp_tag.append(1)
            else:
                temp_tag.append(0)
        tags.append(temp_tag)
    return torch.tensor(tags).to(device)

def similarity_tags(tag1, tag2):
    return len(torch.nonzero(tag1 * tag2))