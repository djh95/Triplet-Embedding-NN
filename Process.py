from TripletLossFunc import TripletLossFunc
import random

import torch
import torch.nn.functional as F

import torchvision.models as models

from Utils import *
from Define import *
from NUS_WIDE_Helper import *

from jupyterplot import ProgressPlot
from tqdm.notebook import tqdm
from TenNetImage import *
from TenNetTag import *
from TagDecoder import *

def process(x_images, y_tags, image_model, tag_model, lossImageTag, lossImageImage, Lambda = 0.1):

    images_feature = image_model(x_images)
    tags_feature = tag_model(y_tags)
        
    # in feature space
    dist_image_tag_pos =  F.pairwise_distance(images_feature, tags_feature)

    # first triplet loss, an image, cor tag, and a neg image
    anchor = images_feature
    positive = tags_feature
    z_images_neg = get_one_neighbor(images_feature, dist_image_tag_pos + Margin_Distance)
    negative = torch.cat([images_feature[i].view(1,-1) for i in z_images_neg])

    lossIT, dist_image_tag_pos, dist_image_image_neg_1 = lossImageTag(anchor, positive, negative)

    # second triplet loss, an image, a pos image, a neg image
    anchor = images_feature
    z_images_pos, z_images_neg = get_pos_neg(y_tags)
    positive = torch.cat([images_feature[i].view(1,-1) for i in z_images_pos])
    negative = torch.cat([images_feature[i].view(1,-1) for i in z_images_neg])

    lossII, dist_image_image_pos, dist_image_image_neg_2 =lossImageImage(anchor, positive, negative)
    loss = lossIT +  Lambda * lossII

    return loss, dist_image_tag_pos, dist_image_image_pos, dist_image_image_neg_1, dist_image_image_neg_2