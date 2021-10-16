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

def train(image_model, tag_model, data_loader,lossIT, lossII, Lambda, optim, number, updata=True):
    train_loss = 0
    train_IT_positive_dis = 0
    train_II_positive_dis = 0
    train_negative_dis_1 = 0
    train_negative_dis_2 = 0

    image_model.train()
    tag_model.train()

    for (x_images,y_tags) in data_loader:
        
        x_images, y_tags = x_images.to(device), y_tags.to(device)
        
        loss, IT_pos_dis, II_pos_dis, II_neg_dis_1, II_neg_dis_2 = process(x_images, y_tags, image_model, tag_model, lossIT, lossII, Lambda)

        if updata:
            optim.zero_grad()
            loss.backward()
            optim.step()

        train_loss += loss.item()
        train_IT_positive_dis += IT_pos_dis.float().sum().item()
        train_II_positive_dis += II_pos_dis.float().sum().item()
        train_negative_dis_1 += II_neg_dis_1.float().sum().item()
        train_negative_dis_2 += II_neg_dis_2.float().sum().item()

    train_loss /= len(data_loader)
    train_IT_positive_dis /= number
    train_II_positive_dis /= number
    train_negative_dis_1 /= number
    train_negative_dis_2 /= number
    return train_loss, train_IT_positive_dis, train_II_positive_dis, train_negative_dis_1, train_negative_dis_2

def evalue(image_model, tag_model, data_loader, lossIT, lossII, Lambda, optim, number, epoch, min_valid_loss=-1, save_best=True):
    valid_loss = 0
    valid_IT_positive_dis = 0
    valid_II_positive_dis = 0
    valid_negative_dis_1 = 0
    valid_negative_dis_2 = 0
    
    image_model.eval()
    tag_model.eval()

    with torch.no_grad():
        for (x_images,y_tags) in data_loader:
            
            x_images, y_tags = x_images.to(device), y_tags.to(device)

            loss, IT_pos_dis, II_pos_dis, II_neg_dis_1, II_neg_dis_2 = process(x_images, y_tags, image_model, tag_model, lossIT, lossII, Lambda)
       
            valid_loss += loss.item()
            valid_IT_positive_dis += IT_pos_dis.float().sum().item()
            valid_II_positive_dis += II_pos_dis.float().sum().item()
            valid_negative_dis_1 += II_neg_dis_1.float().sum().item()
            valid_negative_dis_2 += II_neg_dis_2.float().sum().item()

    valid_loss /= len(data_loader)
    valid_IT_positive_dis /= number
    valid_II_positive_dis /= number
    valid_negative_dis_1 /= number
    valid_negative_dis_2 /= number

    if save_best and (min_valid_loss == -1 or min_valid_loss > valid_loss):
        min_valid_loss = valid_loss
        torch.save({
            'epoch': epoch,
            'image_model_state_dict': image_model.state_dict(),
            'tag_model_state_dict': tag_model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'valid_loss': valid_loss,
            }, "best_val.ckpt")
    return valid_loss, valid_IT_positive_dis, valid_II_positive_dis, valid_negative_dis_1, valid_negative_dis_2, min_valid_loss