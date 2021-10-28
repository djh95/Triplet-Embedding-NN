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
        if predictions[i] > threshold:
            tags.append(1)
        else:
            tags.append(0)
    return torch.tensor(tags).to(device)

def similarity_tags(tag1, tag2):
    return len(torch.nonzero(tag1 * tag2))