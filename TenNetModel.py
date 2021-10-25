from TripletLossFunc import TripletLossFunc
import random

import torch
import torch.nn.functional as F

import torchvision.models as models
from tqdm.notebook import tqdm

from Utils import *
from Define import *
from NUS_WIDE_Helper import *

from jupyterplot import ProgressPlot
from TenNetImage import *
from TenNetTag import *
from TagDecoder import *

def compute_loss(x_images, y_tags, image_model, tag_model, triplet_loss, Lambda = 0.5):

    image_features = image_model(x_images)
    tag_features = tag_model(y_tags)
        
    # in feature space
    IT_dist =  F.pairwise_distance(image_features, tag_features)

    # first triplet loss, an image, cor tag, and a neg image
    anchor_image = image_features
    positive_tag = tag_features

    similarity_matrix = get_similarity_matrix(y_tags)
    z_tag_indexes = get_one_neighbor(tag_features, similarity_matrix, IT_dist + Margin_Distance)
    negative_tag = torch.cat([tag_features[i].view(1,-1) for i in z_tag_indexes])

    z_images_pos, z_images_neg = get_pos_neg(y_tags, similarity_matrix)
    positive_image = torch.cat([image_features[i].view(1,-1) for i in z_images_pos])
    negative_image = torch.cat([image_features[i].view(1,-1) for i in z_images_neg])

    lossIT, dist_image_tag_pos, dist_image_tag_neg = triplet_loss(anchor_image, positive_tag, negative_tag)

    # second triplet loss, an image, a pos image, a neg image
    lossII, dist_image_image_pos, dist_image_image_neg =triplet_loss(anchor_image, positive_image, negative_image)
    loss = lossIT +  Lambda * lossII

    return loss, dist_image_tag_pos, dist_image_image_pos, dist_image_tag_neg, dist_image_image_neg

def single_epoch_computation(image_model, tag_model, loader, triplet_loss, Lambda, optim, updata):
    loss = 0
    IT_positive_dis = 0
    II_positive_dis = 0
    IT_negative_dis = 0
    II_negative_dis = 0

    for (x_images,y_tags) in loader:
        
        x_images, y_tags = x_images.to(device), y_tags.to(device)    
        res = compute_loss(x_images, y_tags, image_model, tag_model, triplet_loss, Lambda)

        if updata:
            optim.zero_grad()
            res[0].backward()
            optim.step()

        loss += res[0].item()
        IT_positive_dis += res[1].item()
        II_positive_dis += res[2].item()
        IT_negative_dis += res[3].item()
        II_negative_dis += res[4].item()

    loss /= len(loader)
    IT_positive_dis /= len(loader)
    II_positive_dis /= len(loader)
    IT_negative_dis /= len(loader)
    II_negative_dis /= len(loader)

    return loss, IT_positive_dis, II_positive_dis, IT_negative_dis, II_negative_dis

def train(image_model, tag_model, loader, triplet_loss, Lambda, optim):

    image_model.train()
    tag_model.train()

    res = single_epoch_computation(image_model, tag_model, loader, triplet_loss, Lambda, optim, updata=True)

    return res

def evalue(image_model, tag_model, loader, triplet_loss, Lambda, optim, epoch, min_loss, save_best=True):
    
    image_model.eval()
    tag_model.eval()

    with torch.no_grad():
        res = single_epoch_computation(image_model, tag_model, loader, triplet_loss, Lambda, optim, updata=False)

    if save_best and (min_loss == -1 or min_loss > res[0]):
        min_loss = res[0]
        torch.save({
            'epoch': epoch,
            'image_model_state_dict': image_model.state_dict(),
            'tag_model_state_dict': tag_model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'loss': res[0],
            }, "best_val.ckpt")
            
    return res + (min_loss,)

def output_loss_dis(s, loss_dis):
    print(  s + "\n" +
            f"loss: {loss_dis[0]:.2f},  " +
            f"IT_pos_dis: {loss_dis[1]:.2f},  " +
            f"II_pos_dis: {loss_dis[2]:.2f},  " + 
            f"IT_neg_dis: {loss_dis[3]:.2f},  " + 
            f"II_neg_dis: {loss_dis[4]:.2f}\n " )

def getTenModel(tag_model, image_model, name = "best_val.ckpt"):
    try:
        checkpoint = torch.load(name)
        image_model.load_state_dict(checkpoint['image_model_state_dict'])   
        tag_model.load_state_dict(checkpoint['tag_model_state_dict'])        # 从字典中依次读取
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("loss: ", loss)
        print("epoch: ",epoch)
        print("Load last checkpoint data")
    except FileNotFoundError:
        print("Can\'t found " + name)

def updataProgressPlot(pp, loss_dis_train, loss_dis_valid, max_v):
    pp.update([ [min(loss_dis_train[0], max_v), min(loss_dis_valid[0], max_v), 0, 0],
                [min(loss_dis_train[1], max_v), 
                 min(loss_dis_train[2], max_v), 
                 min(loss_dis_train[3], max_v), 
                 min(loss_dis_train[4], max_v)]])



def printResult(res, n_epochs, max_v):

    pbar = tqdm(range(min(len(res), n_epochs)))
    pp = ProgressPlot(plot_names=["loss", "train distance"],
                  line_names=["train/pos_IT", "valid/pos_II", "0/neg_IT", "0/neg_II"],
                  x_lim=[0, n_epochs-1], 
                  y_lim=[0, max_v])

    for e in pbar:
    
        loss_dis_train =  res[e][0]
        output_loss_dis(f"epoch:{e}: 1-train dataset with train model", loss_dis_train)
        
        loss_dis_valid = res[e][1]   
        output_loss_dis(f"epoch:{e}: 2-valid dataset with evalue model", loss_dis_valid)
    
        updataProgressPlot(pp, loss_dis_train, loss_dis_valid, max_v)

    pp.finalize()


